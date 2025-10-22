# 시계열 예측(Time Series Forecasting) 프로젝트 템플릿 (KR)
> 선형회귀/이미지 분류 템플릿과 같은 톤으로 **바로 실행 가능한 구조** + **코딩테스트 치트시트** 포함  
> 마지막 업데이트: 2025-10-22 02:44

## ✅ 핵심 특징
- **데이터 형식**: CSV (`date,value`) 기본 (헤더명 커스터마이즈 가능)
- **베이스라인**: Naive, Seasonal-Naive, Moving Average
- **통계 모델**: ARIMA (선택)
- **DL 모델**: LSTM (PyTorch) — 슬라이딩 윈도우
- **학습/평가/추론** 분리: `train.py`, `evaluate.py`, `infer.py`
- **지표**: MAE, RMSE, MAPE
- **전략**: 홀드아웃 또는 **워크-포워드**(rolling) 예측 지원

## 📁 폴더 구조
```
time_series_forecasting_project_ko/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ data.py
│  ├─ metrics.py
│  ├─ baselines.py
│  ├─ models.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ infer.py
└─ scripts/
   ├─ train_lstm.sh
   ├─ evaluate.sh
   └─ infer.sh
```

## 🧩 데이터 예시
```
date,value
2023-01-01,123
2023-01-02,130
2023-01-03,128
...
```
- 다른 컬럼명일 경우 `--date_col`, `--value_col`로 지정
- 결측은 선형보간(`data.py`)으로 간단 처리

## ⚙️ 설치
```
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 학습 (LSTM)
```
python -m src.train --data_csv data/train.csv --model lstm \
  --window 24 --horizon 1 --epochs 10 --lr 1e-3 --batch_size 64
```
- `--window`: 입력 윈도우 길이, `--horizon`: 예측 스텝(1 스텝 ahead)
- 표준화는 `StandardScaler` 자동 적용(훈련 세트에만 `fit`)

## 📊 평가
```
python -m src.evaluate --ckpt runs/best.pt --data_csv data/val.csv --window 24 --horizon 1
```
- `runs/metrics.json`에 MAE/RMSE/MAPE 저장

## 🔎 단일/연속 추론
```
python -m src.infer --ckpt runs/best.pt --recent_csv data/recent.csv --window 24 --horizon 7
```
- `recent.csv`는 **가장 최신 구간**만 포함하면 됨(최소 `window` 길이)

---

# 🧾 코딩테스트 치트시트 (복붙용 스니펫)

## 1) 슬라이딩 윈도우 만들기 (넘파이)
```python
import numpy as np

def make_windows(arr, window, horizon=1):
    X, y = [], []
    for i in range(len(arr) - window - horizon + 1):
        X.append(arr[i:i+window])
        y.append(arr[i+window:i+window+horizon])
    return np.array(X), np.array(y).squeeze()
```

## 2) Torch Dataset
```python
import torch
from torch.utils.data import Dataset

class SeriesDS(Dataset):
    def __init__(self, series, window, horizon=1):
        self.X, self.y = make_windows(series, window, horizon)
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)  # (N, W, 1)
        self.y = torch.tensor(self.y, dtype=torch.float32)                # (N,) or (N,H)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]
```

## 3) LSTM 모델 (Univariate)
```python
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden=64, layers=1, out=1, drop=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True, dropout=drop if layers>1 else 0.0)
        self.fc = nn.Linear(hidden, out)
    def forward(self, x):                  # x: (B, W, 1)
        o, _ = self.lstm(x)               # o: (B, W, H)
        h_last = o[:, -1, :]              # (B, H)
        return self.fc(h_last)            # (B, out)
```

## 4) 학습 루프 요약
```python
for e in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(dev), yb.to(dev)
        pred = model(xb).squeeze()
        loss = crit(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
```

## 5) 지표 함수
```python
import numpy as np
def rmse(y, p): return float(np.sqrt(np.mean((y-p)**2)))
def mae(y, p):  return float(np.mean(np.abs(y-p)))
def mape(y, p): return float(np.mean(np.abs((y-p)/np.maximum(np.abs(y), 1e-8))) * 100)
```

## 6) 베이스라인 3종
```python
def naive(y): return y[-1]                 # 마지막 값 그대로
def seasonal_naive(y, m): return y[-m]     # m-주기 전 값
def moving_avg(y, k): return np.mean(y[-k:])
```

## 7) 워크-포워드 예측(1-step)
```python
def walk_forward_predict(model, series, window, steps):
    buf = series[-window:].copy().tolist()
    preds = []
    for _ in range(steps):
        x = np.array(buf[-window:], dtype=np.float32).reshape(1, window, 1)
        x = torch.tensor(x).to(dev)
        p = model(x).detach().cpu().numpy().ravel()[0]
        preds.append(p)
        buf.append(p)
    return np.array(preds)
```

---

# 📌 팁
- 스케일러는 **훈련 데이터로만 fit** → 검증/테스트는 transform만 (데이터 누수 방지)
- `window↑`는 장기 의존 포착에 도움 but 데이터 부족 시 과적합↑
- H>1 다중 스텝이면: (a) 디렉트 아웃 `out=H`, (b) 워크-포워드 반복, (c) Seq2Seq

행운을 빕니다! 🚀
