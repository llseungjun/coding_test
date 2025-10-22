# imgcls_pipeline.py
# ----------------------------------------------------------
# 1) 데이터 전처리  2) 모델 선택  3) 모델 구현  4) 모델 평가
# ----------------------------------------------------------
import os, random, argparse, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------
# 공통 유틸
# ---------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def accuracy_top1(logits, targets):
    pred = logits.argmax(1)
    return (pred == targets).float().mean().item()

@torch.no_grad()
def precision_recall_f1_macro(logits, targets, n_classes: int):
    pred = logits.argmax(1).cpu().numpy()
    y = targets.cpu().numpy()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    # per-class precision, recall
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
        rec  = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        f1   = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    # NaN -> 0
    prec = np.nan_to_num(prec)
    rec  = np.nan_to_num(rec)
    f1   = np.nan_to_num(f1)
    return float(prec.mean()), float(rec.mean()), float(f1.mean())

def save_checkpoint(model, path="best.pt"):
    torch.save(model.state_dict(), path)

# ==========================================================
# 1) 데이터 전처리
# ==========================================================
def get_transforms(img_size=224, aug=True, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    if aug:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf

def make_loaders(args):
    train_tf, test_tf = get_transforms(args.img_size, aug=not args.no_aug)
    if args.dataset.lower() == "cifar10":
        # 자동 다운로드
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
        test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
        # train/val 분할
        n = len(train_ds)
        val_size = int(n*args.val_ratio)
        train_size = n - val_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        n_classes = 10
    elif args.dataset.lower() == "imagefolder":
        assert os.path.isdir(args.data_dir), "imagefolder는 --data_dir에 데이터 경로 필요 (train/val/test)"
        train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=test_tf)
        test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, "test"),  transform=test_tf)
        n_classes = len(train_ds.dataset.classes) if hasattr(train_ds, "dataset") else len(train_ds.classes)
    else:
        raise ValueError("dataset은 cifar10 또는 imagefolder만 지원")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, n_classes

# ==========================================================
# 2) 모델 선택
#    - --model smallcnn | resnet18
#    - --pretrained 로 전이학습 on/off
# ==========================================================
def pick_model(name: str, n_classes: int, pretrained: bool, device: str):
    name = name.lower()
    if name == "smallcnn":
        model = SmallCNN(n_classes)
    elif name == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
        # 분류기 헤드 교체
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError("--model 은 smallcnn 또는 resnet18")
    return model.to(device)

# ==========================================================
# 3) 모델 구현
#    - 경량 CNN (SmallCNN)
#    - 학습/평가 루프 (AMP + EarlyStopping + Checkpoint)
# ==========================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def train_one_epoch(model, loader, optimizer, lossf, device, scaler=None):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device, dtype=torch.float16 if device=='cuda' else torch.bfloat16):
                out = model(x)
                loss = lossf(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = lossf(out, y)
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, n_classes: int):
    model.eval()
    lossf = nn.CrossEntropyLoss()
    loss_sum, total = 0.0, 0
    preds_all, targets_all = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = lossf(out, y)
        loss_sum += loss.item()*y.size(0)
        total    += y.size(0)
        preds_all.append(out.detach().cpu())
        targets_all.append(y.detach().cpu())
    logits = torch.cat(preds_all, 0)
    targets = torch.cat(targets_all, 0)
    acc = accuracy_top1(logits, targets)
    prec, rec, f1 = precision_recall_f1_macro(logits, targets, n_classes)
    return loss_sum/total, acc, prec, rec, f1

def fit(model, train_loader, val_loader, device, epochs=10, lr=3e-4, wd=1e-4, patience=3, ckpt_path="best.pt"):
    lossf = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
    best_acc, no_improve = -1.0, 0

    for ep in range(1, epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, lossf, device, scaler)
        va_loss, va_acc, va_prec, va_rec, va_f1 = evaluate(model, val_loader, device, n_classes=model.classifier.out_features if hasattr(model, "classifier") else model.fc.out_features)
        dt = time.time()-t0
        print(f"E{ep:02d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f}  ({dt:.1f}s)")

        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint(model, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep}. Best val acc={best_acc:.4f}")
                break
    # best 로드
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model

# ==========================================================
# 4) 모델 평가 (최종 테스트)
#    - Accuracy / Precision / Recall / F1 (macro)
# ==========================================================
def evaluate_test(model, test_loader, device, n_classes: int):
    te_loss, te_acc, te_prec, te_rec, te_f1 = evaluate(model, test_loader, device, n_classes)
    print("\n[TEST RESULTS]")
    print(f" Acc={te_acc:.4f} | Prec(macro)={te_prec:.4f} | Rec(macro)={te_rec:.4f} | F1(macro)={te_f1:.4f}")
    return dict(acc=te_acc, prec=te_prec, rec=te_rec, f1=te_f1)

# ---------------------------
# Main
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","imagefolder"])
    ap.add_argument("--data_dir", type=str, default="./data")   # imagefolder일 때: data/train, data/val, data/test
    ap.add_argument("--model", type=str, default="resnet18", choices=["smallcnn","resnet18"])
    ap.add_argument("--pretrained", action="store_true", help="resnet 전이학습 사용")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)     # CIFAR10에서만 사용
    ap.add_argument("--no_aug", action="store_true", help="학습 증강 비활성화")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_arg_
