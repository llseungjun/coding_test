import os, argparse, json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .utils import get_device, save_metrics, plot_confusion_matrix


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--val_dir", type=str, required=True)
    ap.add_argument("--test_dir", type=str, default=None)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    class_names = ckpt["class_names"]
    cfg = ckpt["cfg"]
    model_name = cfg["model"]

    # build model
    if model_name == "scratch":
        from .models.scratch_cnn import SmallCNN

        model = SmallCNN(num_classes=len(class_names))
    elif model_name == "resnet18_scratch":
        from .models.resnet_scratch import resnet18_scratch

        model = resnet18_scratch(num_classes=len(class_names))
    else:
        from .models.resnet_transfer import make_resnet18

        model = make_resnet18(num_classes=len(class_names), freeze_backbone=False)

    model.load_state_dict(ckpt["model"])
    device = get_device()
    model = model.to(device).eval()

    tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def evaluate_split(split_dir, split_name):
        ds = datasets.ImageFolder(split_dir, transform=tf)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        y_true, y_pred = [], []
        total, correct = 0, 0
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        acc = correct / max(1, total)
        return acc, y_true, y_pred

    val_acc, val_true, val_pred = evaluate_split(args.val_dir, "val")
    print(f"VAL acc: {val_acc:.4f}")
    metrics = {"val_acc": val_acc}

    if args.test_dir and os.path.isdir(args.test_dir):
        test_acc, test_true, test_pred = evaluate_split(args.test_dir, "test")
        print(f"TEST acc: {test_acc:.4f}")
        metrics["test_acc"] = test_acc
        plot_confusion_matrix(
            test_true,
            test_pred,
            class_names,
            path=os.path.join(args.out_dir, "confusion_matrix.png"),
        )
    else:
        plot_confusion_matrix(
            val_true,
            val_pred,
            class_names,
            path=os.path.join(args.out_dir, "confusion_matrix.png"),
        )

    save_metrics(metrics, os.path.join(args.out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
