# 이미지 분류(Image Classification) 프로젝트 템플릿 (KR)
> (마지막 업데이트: 2025-10-22 10:55)

## ✅ 핵심 특징
- **두 가지 학습 방식** 제공
  1) `scratch_cnn.py` : **완전 스크래치 CNN** (작은 데이터셋/과제용)
  2) `resnet_transfer.py` : **ResNet18 전이학습** (실전 베이스라인)
- **ImageFolder** 형식 데이터 사용: `data/train/class_x/...`, `data/val/class_y/...`
- **학습/평가/추론** 스크립트 분리: `train.py`, `evaluate.py`, `infer.py`
- **재현성**: 시드 고정, 체크포인트 저장, 로그 출력
- **편의 유틸**: 조기종료(EarlyStopping), 혼동행렬 저장, best model 저장

## 📁 폴더 구조
```
image_classification_project_ko/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ models/
│  │  ├─ scratch_cnn.py
│  │  └─ resnet_transfer.py
│  ├─ utils.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ infer.py
└─ scripts/
   ├─ train_scratch.sh
   └─ train_transfer.sh
```

## 🧩 데이터 준비 (ImageFolder)
```
data/
├─ train/
│  ├─ class_a/ xxx.jpg ...
│  └─ class_b/ yyy.jpg ...
└─ val/
   ├─ class_a/ ...
   └─ class_b/ ...
```
- 클래스 이름이 **라벨**로 사용됩니다.
- 테스트가 따로 있으면 `--test_dir data/test`로 평가/추론 시 지정.

## ⚙️ 설치
```
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 학습
### 1) 스크래치 CNN
```
python -m src.train --model scratch --train_dir data/train --val_dir data/val \
  --epochs 20 --batch_size 64 --lr 1e-3 --img_size 224
```

### 2) ResNet18 전이학습
```
python -m src.train --model resnet18 --train_dir data/train --val_dir data/val \
  --epochs 10 --batch_size 64 --lr 3e-4 --img_size 224 --freeze_backbone true
```

## 📊 평가
```
python -m src.evaluate --ckpt runs/best.pt --val_dir data/val --img_size 224
# (선택) --test_dir data/test
```

- `runs/metrics.json`에 정량 지표 저장, `runs/confusion_matrix.png` 저장.

## 🔎 단일 이미지 추론
```
python -m src.infer --ckpt runs/best.pt --image path/to/image.jpg --img_size 224
```

## 🧪 빠른 샘플 실행(더미)
더미 폴더 구조만 있어도 커맨드/에러 흐름을 확인할 수 있습니다. 실제 학습에는 이미지가 필요합니다.

## 🧰 팁
- 작은 데이터셋: `scratch`가 오히려 과적합/수렴이 빠를 수 있음
- 일반 베이스라인: `resnet18` 고정 → `--freeze_backbone false`로 풀고 미세 튜닝
- 데이터 증강: `train.py` 내부 `get_transforms`에서 간단히 조절 가능  
  
---  
  
# 🧠 CNN 이미지 분류 코드 스니펫
> 딥러닝 코딩테스트나 실무 테스트에서 **바로 복붙 가능한 핵심 구조 요약집**  
> 마지막 업데이트: 2025-10-22 11:29

---

## 📌 기본 구조
CNN 기반 이미지 분류 문제는 대부분 아래 순서로 구성됩니다.

1. **데이터셋 정의 (Dataset / DataLoader)**
2. **모델 정의 (CNN 구조)**
3. **손실함수 & 옵티마이저 정의**
4. **학습 루프 (train / validation)**
5. **평가 및 예측**

---

## 1️⃣ 데이터셋 & 로더 설정
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

test_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_ds = datasets.ImageFolder("data/train", transform=train_tf)
test_ds = datasets.ImageFolder("data/test", transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

num_classes = len(train_ds.classes)
```

---

## 2️⃣ CNN 모델 기본 구조 (가벼운 SmallCNN)
```python
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

✅ **Tip:** `AdaptiveAvgPool2d(1,1)` 덕분에 입력 크기가 달라도 동작.

---

## 3️⃣ 학습 준비
```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

---

## 4️⃣ 학습 루프 (Train + Validate)
```python
for epoch in range(10):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total
    print(f"[{{epoch+1}}] loss={{train_loss/total:.4f}} acc={{acc:.4f}}")
```

---

## 5️⃣ 평가 & 예측
```python
from sklearn.metrics import accuracy_score, classification_report

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        preds = out.argmax(1).cpu().tolist()
        y_pred += preds
        y_true += y.tolist()

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=train_ds.classes))
```

---

## 6️⃣ 개선용 스니펫 모음

### ✅ BatchNorm 추가
```python
nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
```

### ✅ Data Augmentation 다양화
```python
transforms.RandomRotation(10),
transforms.ColorJitter(brightness=0.2, contrast=0.2),
transforms.RandomErasing(),
```

### ✅ Scheduler / Early Stopping
```python
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    # train loop ...
    scheduler.step()
```

### ✅ 모델 저장 / 로드
```python
torch.save(model.state_dict(), "best.pt")
model.load_state_dict(torch.load("best.pt", map_location=device))
```

### ✅ 예측 확률 보기
```python
import torch.nn.functional as F

with torch.no_grad():
    img = next(iter(test_loader))[0][0].unsqueeze(0).to(device)
    probs = F.softmax(model(img), dim=1)[0]
    for cls, p in zip(train_ds.classes, probs):
        print(f"{{cls}}: {{p.item():.3f}}")
```

---

## 7️⃣ 빠른 템플릿 (10줄 컷)
```python
model = SmallCNN(num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
for e in range(10):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        l = crit(model(x), y); l.backward(); opt.step()
```

---

## 🧩 면접/시험에서 자주 나오는 질문
| 질문 | 핵심 요약 |
|------|-------------|
| CNN에서 **커널 크기 3**의 의미는? | 입력 특징을 3×3 영역 단위로 보는 필터 |
| **Padding=1**을 주는 이유는? | 출력 크기를 입력과 동일하게 유지 |
| **Pooling**을 하는 이유는? | 공간 크기 감소, translation-invariance 확보 |
| **ReLU**를 쓰는 이유는? | 비선형성 부여 + gradient vanishing 완화 |
| **Dropout**은 언제 쓰나? | 과적합 방지, Fully Connected 층에 주로 사용 |
| **CrossEntropyLoss**는 Softmax를 포함하나? | ✅ 네, 내부적으로 `log_softmax`가 포함됨 |
| **AdaptiveAvgPool2d(1,1)**의 역할은? | 입력 크기에 상관없이 1×1로 전역 요약 |

---

## 🎯 한 줄 요약
> “Conv + ReLU + Pool 반복 → 전역요약(GAP) → Fully Connected → Softmax(Class)”  
> 이게 이미지 분류 CNN의 기본 뼈대다! 💪

---

### 📘 추천 연습 문제 (코딩테스트 연습용)
- [Dacon: 식물 잎 질병 분류](https://dacon.io/competitions/official/235862)
- [Kaggle: CIFAR-10 Image Classification](https://www.kaggle.com/c/cifar-10)
- [AI Stage: Tiny Image Classification]()

---