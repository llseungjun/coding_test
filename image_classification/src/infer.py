import argparse, os
import torch
from PIL import Image
from torchvision import transforms
from .utils import get_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    class_names = ckpt["class_names"]
    cfg = ckpt["cfg"]
    model_name = cfg["model"]

    if model_name == "scratch":
        from .models.scratch_cnn import SmallCNN

        model = SmallCNN(num_classes=len(class_names))
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

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0].cpu().tolist()
    pred_idx = int(torch.argmax(logits, dim=1).item())
    print(f"Pred: {class_names[pred_idx]} (idx={pred_idx})")
    print("Prob:", {class_names[i]: round(p, 4) for i, p in enumerate(prob)})


if __name__ == "__main__":
    main()
