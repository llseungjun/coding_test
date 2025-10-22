# Linear Regression Project (Scikit-learn & From-Scratch)

이 프로젝트는 **선형 회귀(Linear Regression)** 를  
① **Scikit-learn 버전**과  
② **NumPy 기반 완전 스크래치 버전**  
두 가지로 구현하고 비교할 수 있는 구조입니다.

---

## 📂 폴더 구조 및 파일별 기능

linreg-project/  
├─ README.md # 프로젝트 개요 및 구조 설명  
├─ requirements.txt # 필요한 라이브러리 목록  
├─ configs/ # 설정 파일 폴더  
│ ├─ sklearn.yaml # scikit-learn 버전 설정  
│ └─ scratch.yaml # 스크래치 버전 설정  
│  
├─ common/ # 공용 유틸리티 모듈  
│ └─ utils.py # 시드 설정, YAML 로드, Pickle 저장/로딩, 디렉터리 생성 등  
│  
├─ linreg_sklearn/ # scikit-learn 기반 구현  
│ ├─ init.py  
│ ├─ data.py # CSV 또는 sklearn 데이터셋 로드  
│ ├─ model.py # StandardScaler + LinearRegression 파이프라인 정의   
│ ├─ train.py # 데이터 분할, 모델 학습, 저장 및 평가 출력  
│ └─ evaluate.py # 학습된 모델 로드 후 새 데이터 평가  
│  
├─ linreg_scratch/ # 완전 스크래치(NumPy) 구현  
│ ├─ init.py  
│ ├─ data.py # CSV 또는 합성 데이터 생성 (y = Xw + noise)  
│ ├─ features.py # 표준화(스케일링), bias 추가 기능  
│ ├─ model.py # LinearRegressionGD / LinearRegressionNE 구현  
│ ├─ metrics.py # RMSE, MAE, R² 등 회귀 성능 지표 함수  
│ ├─ train.py # 학습 파이프라인 (train/val/test 분할, 학습, 저장)  
│ └─ evaluate.py # 저장된 모델 로드 후 CSV 기반 평가  
│  
└─ scripts/ # CLI(명령줄 실행) 스크립트  
├─ train_sklearn.py # scikit-learn 버전 학습 실행  
├─ eval_sklearn.py # scikit-learn 버전 평가 실행  
├─ train_scratch.py # 스크래치 버전 학습 실행  
└─ eval_scratch.py # 스크래치 버전 평가 실행  


---

## ⚙️ 주요 구성 설명

### 🧩 1. configs/
- YAML 포맷 설정 파일로 학습 파라미터 및 경로 관리.
- 데이터 분할 비율, 학습률, 에폭 수, 모델 저장 경로 등을 지정.

### 🧰 2. common/utils.py
- 공통 유틸 함수 모음:
  - `set_seed()`: 랜덤 시드 고정  
  - `load_yaml()`: YAML 설정 불러오기  
  - `save_pickle()/load_pickle()`: 모델 저장 및 로드  
  - `ensure_dir()`: 디렉터리 자동 생성  

### 🤖 3. linreg_sklearn/
- Scikit-learn 기반의 파이프라인 학습 구현.
- `train.py`는 `train_test_split`으로 분할 후 `LinearRegression` 학습 및 저장.
- `evaluate.py`는 저장된 모델로 CSV 입력 평가.

### 🧮 4. linreg_scratch/
- NumPy만으로 구현된 완전 스크래치 버전.
- `model.py`  
  - `LinearRegressionGD`: 경사하강법(GD) 기반 학습  
  - `LinearRegressionNE`: 정규방정식(Closed-form) 기반 학습
- `metrics.py`: RMSE, MAE, R² 등 지표 계산.
- `train.py`: 합성 데이터 생성, 스케일링, 학습, 저장.
- `evaluate.py`: 저장된 모델 불러와 CSV 데이터로 성능 평가.

### 🧩 5. scripts/
- CLI 환경에서 손쉽게 학습/평가 실행 가능:
  ```bash
  python scripts/train_sklearn.py --config configs/sklearn.yaml
  python scripts/eval_scratch.py --model artifacts/scratch/model.pkl --csv your.csv --target target

---

# 선형회귀(Linear Regression) 코드 스니펫 모음
> 시험 중 빠르게 참고하는 **필살 치트시트(README)** — 핵심만 간단·짧게!  
> 마지막 업데이트: 2025-10-22 10:22

---

## 📌 바로 사용 요약
- **필수 패키지**: `numpy`, `pandas`, `scikit-learn`, (선택) `statsmodels`
- **데이터 분리** → **모델 학습** → **평가** 순서만 기억!
- **정규화/표준화**: `Pipeline`과 `ColumnTransformer` 조합이 가장 깔끔.

---

## 0) 데이터 예시 만들기 (복붙용)
```python
import numpy as np, pandas as pd
np.random.seed(42)

n = 200
X = pd.DataFrame({
    "x1": np.random.randn(n),
    "x2": np.random.randn(n) * 2 + 1,
    "cat": np.random.choice(["A","B","C"], size=n)
})
y = 3.0 + 2.0*X["x1"] - 0.5*X["x2"] + (X["cat"]=="B")*1.5 + np.random.randn(n)*0.8
```

---

## 1) 가장 기본: scikit-learn 선형회귀
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_num = X[["x1","x2"]]                # 수치형만 우선 사용
X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_tr, y_tr)
pred = lr.predict(X_te)

mse  = mean_squared_error(y_te, pred)
rmse = mse ** 0.5
mae  = mean_absolute_error(y_te, pred)
r2   = r2_score(y_te, pred)

print("coef_:", lr.coef_, "intercept_:", lr.intercept_)
print(f"RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")
```

---

## 2) 원핫인코딩 + 스케일링 + 파이프라인 (실전형)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

num_cols = ["x1","x2"]
cat_cols = ["cat"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols),
])

pipe = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression())
])

pipe.fit(X, y)
yhat = pipe.predict(X)
print("학습 R2:", r2_score(y, yhat))
```

---

## 3) 다항 특성(비선형성) 포함
```python
from sklearn.preprocessing import PolynomialFeatures

poly_pipe = Pipeline([
    ("prep", preprocess),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])
poly_pipe.fit(X, y)
print("다항 R2:", r2_score(y, poly_pipe.predict(X)))
```

---

## 4) 정규화 회귀: Ridge / Lasso / ElasticNet
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Pipeline([("prep", preprocess), ("model", Ridge(alpha=1.0))]).fit(X, y)
lasso = Pipeline([("prep", preprocess), ("model", Lasso(alpha=0.05, max_iter=10000))]).fit(X, y)
enet  = Pipeline([("prep", preprocess), ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))]).fit(X, y)

print("Ridge R2:", r2_score(y, ridge.predict(X)))
print("Lasso R2:", r2_score(y, lasso.predict(X)))
print("ElasticNet R2:", r2_score(y, enet.predict(X)))
```

**Tip**: 규제가 강할수록(큰 `alpha`) 계수가 줄고 과적합 방지에 도움.  
**Lasso**는 일부 계수를 **0으로** 만들어 특성 선택 효과.

---

## 5) 교차검증 & 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# 5-1) 교차검증 점수
scores = cross_val_score(pipe, X, y, scoring="r2", cv=5)
print("CV R2 mean:", scores.mean(), "±", scores.std())

# 5-2) GridSearch로 Ridge alpha 찾기
param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0]}
ridge_gs = GridSearchCV(Pipeline([("prep", preprocess), ("model", Ridge())]),
                        param_grid=param_grid, scoring="r2", cv=5, n_jobs=-1)
ridge_gs.fit(X, y)
print("Best alpha:", ridge_gs.best_params_, "Best R2:", ridge_gs.best_score_)
```

---

## 6) 지표 모음 (복붙용 함수)
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def eval_reg(y_true, y_pred, prefix="VAL"):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    print(f"[{prefix}] RMSE={{rmse:.4f}} | MAE={{mae:.4f}} | R2={{r2:.4f}} | MAPE={{mape:.2f}}%")
    return dict(rmse=rmse, mae=mae, r2=r2, mape=mape)
```

---

## 7) 순수 넘파이 구현(정규방정식, Normal Equation)
```python
import numpy as np

def fit_normal_eq(X, y):
    # X: (n, d), y: (n,)
    X_ = np.c_[np.ones(len(X)), X]          # bias 추가
    theta = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
    return theta  # [intercept, w1, w2, ...]

def predict_normal_eq(X, theta):
    X_ = np.c_[np.ones(len(X)), X]
    return X_ @ theta

X_arr = X_num.values
theta = fit_normal_eq(X_arr, y)
pred  = predict_normal_eq(X_arr, theta)
print("theta:", theta)
print("R2:", r2_score(y, pred))
```

---

## 8) 경사하강법(GD) 기초 구현
```python
import numpy as np

def fit_gd(X, y, lr=0.05, epochs=1000):
    X_ = np.c_[np.ones(len(X)), X]      # bias 포함
    w  = np.zeros(X_.shape[1])
    for _ in range(epochs):
        grad = (2/len(X_)) * X_.T @ (X_ @ w - y)
        w   -= lr * grad
    return w

w = fit_gd(X_num.values, y.values if hasattr(y, "values") else np.array(y), lr=0.05, epochs=2000)
print("GD w:", w)
```

---

## 9) Statsmodels로 계수/유의성 보고서
```python
import statsmodels.api as sm

X_sm = sm.add_constant(X_num)  # 상수항
model = sm.OLS(y, X_sm).fit()
print(model.summary())  # 계수, 표준오차, t-값, p-값, 신뢰구간 등
```

---

## 10) 잔차 진단(간단 체크)
```python
import numpy as np

residuals = y - lr.predict(X_num)
print("잔차 평균≈0?:", np.mean(residuals))
print("잔차 분산 일정성(대략):", np.var(residuals[:len(residuals)//2]), np.var(residuals[len(residuals)//2:]))
# 정교한 검정: Breusch-Pagan, Durbin-Watson 등은 statsmodels 참고
```

---

## 11) 자주 하는 실수 체크리스트
- [ ] **데이터 누수**: 스케일링/인코딩은 반드시 **train에 fit → test에 transform**.
- [ ] **범주형 드랍 기준**: `OneHotEncoder(drop="first")`로 완전 다중공선성 방지.
- [ ] **스케일링 누락**: Lasso/ElasticNet은 **스케일링 필수**(Pipeline 권장).
- [ ] **평가 일관성**: 항상 동일한 지표(RMSE/R2/MAE 등)로 비교.
- [ ] **외적 타당성**: 교차검증으로 과적합 여부 확인.

---

## 12) 최소 예제 템플릿 (시험장 10줄 컷)
```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), num_cols),
                                            ("cat", OneHotEncoder(drop="first"), cat_cols)])),
                 ("model", Ridge(alpha=1.0))])
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)
print(r2_score(y_te, pred))
```

---

### 참고
- 선형성 위배·이상치가 심하면: **로버스트 회귀(HuberRegressor/RANSACRegressor)** 검토
- 목표가 0 근처에서 상대오차 중요하면: **SMAPE, MAPE** 지표 검토
- 특성 수가 매우 많고 희소: **ElasticNet** 우선 고려

