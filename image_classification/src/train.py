import os, yaml, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils import set_seed, get_device, AverageMeter, EarlyStopping, save_ckpt
from .models.scratch_cnn import SmallCNN
from .models.resnet_transfer import make_resnet18
from .models.resnet_scratch import resnet18_scratch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument(
        "--model",
        type=str,
        choices=["scratch", "resnet18", "resnet18_scratch"],
        default=None,
    )
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--val_dir", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--freeze_backbone", type=str, default=None)  # "true"/"false"
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="runs")
    return ap.parse_args()


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def str2bool(x):
    if x is None:
        return None
    return x.lower() in ["1", "true", "t", "yes", "y"]


def get_transforms(size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_model(name, num_classes, freeze_backbone=True):
    if name == "scratch":
        return SmallCNN(num_classes)
    elif name == "resnet18":
        return make_resnet18(num_classes=num_classes, freeze_backbone=freeze_backbone)
    elif name == "resnet18_scratch":
        return resnet18_scratch(num_classes=num_classes)
    else:
        raise ValueError("unknown model")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), imgs.size(0))
        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    acc = correct / max(1, total)
    return loss_meter.avg, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), imgs.size(0))
        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    acc = correct / max(1, total)
    return loss_meter.avg, acc


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # override from CLI (if provided)
    for k in [
        "model",
        "img_size",
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "seed",
        "num_workers",
    ]:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    fb = str2bool(args.freeze_backbone)
    if fb is not None:
        cfg["freeze_backbone"] = fb

    set_seed(cfg.get("seed", 42))
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    train_tf, eval_tf = get_transforms(cfg["img_size"])
    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(args.val_dir, transform=eval_tf)
    class_names = train_ds.classes
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    model = build_model(
        cfg["model"], num_classes, freeze_backbone=cfg.get("freeze_backbone", True)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    early = EarlyStopping(patience=5, mode="max")
    best_acc, best_path = -1, os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        print(
            f"[{epoch:03d}/{cfg['epochs']}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}"
        )

        if early.step(va_acc):
            # save best
            save_ckpt(
                {"model": model.state_dict(), "class_names": class_names, "cfg": cfg},
                best_path,
            )
            best_acc = va_acc
        if early.should_stop:
            print("Early stopping.")
            break

    print(f"Best val acc: {best_acc:.4f} | saved: {best_path}")


if __name__ == "__main__":
    main()
