# linear_reg_pipeline.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1) 데이터 전처리 -----------------------------------------------------------
def preprocess_df(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    df: 원본 DataFrame
    target: 타깃 컬럼명 (str)
    return: X_train, X_val, y_train, y_val, preprocessor (ColumnTransformer)
    """
    y = df[target].values
    X = df.drop(columns=[target])

    # 컬럼 타입 구분
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 수치형: 결측치 평균대치 → 표준화 / 범주형: 최빈값 대치 → 원핫
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols)
        ],
        remainder="drop"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val, preprocessor


# 2) 모델 선택 ---------------------------------------------------------------
def select_model(preprocessor: ColumnTransformer, cv_splits: int = 5, n_jobs: int = -1):
    """
    Linear / Ridge / Lasso 중에서 일반화 성능(neg MSE) 기준으로 선택.
    return: GridSearchCV 객체 (fit 전 상태)
    """
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("reg", LinearRegression())  # placeholder, grid에서 바뀜
    ])

    alphas = np.logspace(-4, 3, 15)  # 1e-4 ~ 1e3
    param_grid = [
        {"reg": [LinearRegression()]},  # 하이퍼파라미터 없음
        {"reg": [Ridge()], "reg__alpha": alphas},
        {"reg": [Lasso(max_iter=20000)], "reg__alpha": alphas},
    ]

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # 낮을수록 좋음 → neg 값이므로 클수록 좋음
        cv=kf,
        n_jobs=n_jobs,
        verbose=0,
        return_train_score=False
    )
    return gs


# 3) 모델 평가 ---------------------------------------------------------------
def evaluate_regression(model, X_val, y_val, prefix: str = "VAL"):
    """
    RMSE / MAE / R^2 출력 및 딕셔너리 반환
    """
    pred = model.predict(X_val)
    mse = mean_squared_error(y_val, pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)

    print(f"[{prefix}] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# 3) 모델 평가 (확장형 구현 버전) ---------------------------------------------
import numpy as np

def evaluate_regression_v2(model, X_val, y_val, prefix: str = "VAL"):
    """
    RMSE / MAE / MAPE / MSLE / R² / Adjusted R²
    모두 수식으로 직접 구현
    """
    pred = model.predict(X_val)
    n = len(y_val)
    p = X_val.shape[1]  # feature 수

    # ---- 기본 오차 지표 ----
    mse = np.sum((y_val - pred) ** 2) / n
    rmse = np.sqrt(mse)
    mae = np.sum(np.abs(y_val - pred)) / n

    # ---- 추가 지표 ----
    # 평균 절대 백분율 오차 (MAPE)
    eps = 1e-8  # 0 나누기 방지
    mape = np.mean(np.abs((y_val - pred) / (y_val + eps))) * 100

    # 로그 스케일 MSE (MSLE)
    pred_clip = np.clip(pred, 0, None)  # 로그 음수 방지
    y_clip = np.clip(y_val, 0, None)
    msle = np.mean((np.log1p(y_clip) - np.log1p(pred_clip)) ** 2)

    # ---- 결정계수(R²) 및 보정된 R² ----
    ss_res = np.sum((y_val - pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    # ---- 출력 ----
    print(f"[{prefix}]")
    print(f" RMSE = {rmse:.4f}")
    print(f" MAE  = {mae:.4f}")
    print(f" MAPE = {mape:.2f}%")
    print(f" MSLE = {msle:.4f}")
    print(f" R²   = {r2:.4f}")
    print(f" Adj R² = {adj_r2:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "msle": msle,
        "r2": r2,
        "adj_r2": adj_r2
    }


# ---- 예시 구동 스니펫 (Diabetes 회귀 데이터로 데모) -------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    df = X.copy()
    df["target"] = y

    # 1) 전처리
    X_tr, X_va, y_tr, y_va, prep = preprocess_df(df, target="target")

    # 2) 모델 선택(CV로 Linear vs Ridge vs Lasso 비교)
    grid = select_model(prep, cv_splits=5)
    grid.fit(X_tr, y_tr)

    print("Best model:", grid.best_estimator_.named_steps["reg"].__class__.__name__)
    if hasattr(grid.best_estimator_.named_steps["reg"], "alpha"):
        print("Best alpha:", grid.best_estimator_.named_steps["reg"].alpha)

    # 3) 홀드아웃 평가
    evaluate_regression(grid.best_estimator_, X_va, y_va)
