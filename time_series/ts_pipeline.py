# ts_pipeline.py
# -----------------------------
# 시계열 예측: 전처리 → 모델 선택 → 모델 구현 → 모델 평가 (실행 가능)
# 데이터: statsmodels.datasets.sunspots (연간 흑점수)
# -----------------------------
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# 모델(특징기반)에서만 sklearn 사용
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.datasets import sunspots


# ---------------------------------------------------------------------
# 0) 공통 유틸: 지표(전부 직접 구현)
# ---------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, m: int = 1, eps: float = 1e-8) -> float:
    """
    MASE: Mean Absolute Scaled Error
    - insample: 학습(인샘플) 실제값 시퀀스
    - m: 계절 주기 (연간 월데이터면 12, 여기 sunspots(연간)는 1)
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    insample = np.asarray(insample)
    # 계절 naive의 평균 절대 변화량
    scale = np.mean(np.abs(insample[m:] - insample[:-m])) + eps
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


# ---------------------------------------------------------------------
# 1) 데이터 전처리
# ---------------------------------------------------------------------
@dataclass
class TSPreprocessResult:
    y: pd.Series
    train: pd.Series
    val: pd.Series
    val_horizon: int
    # 특징기반 모델용 데이터
    X_tr: Optional[pd.DataFrame] = None
    y_tr: Optional[pd.Series] = None
    X_va: Optional[pd.DataFrame] = None
    y_va: Optional[pd.Series] = None
    scaler: Optional[StandardScaler] = None
    used_lags: Optional[List[int]] = None
    used_rolls: Optional[List[int]] = None
    seasonal_m: int = 1  # MASE 계산용


def load_sunspots() -> pd.Series:
    data = sunspots.load_pandas().data
    # YEAR (int), SUNACTIVITY (float)
    y = pd.Series(data["SUNACTIVITY"].values, index=pd.Index(data["YEAR"].astype(int), name="year"))
    y = y.astype(float)
    return y


def make_features(y: pd.Series, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    df = pd.DataFrame({"y": y}).copy()
    for l in lags:
        df[f"lag{l}"] = df["y"].shift(l)
    for r in rolls:
        df[f"roll{r}"] = df["y"].shift(1).rolling(r).mean()
    # 연간 데이터라 간단한 날짜 파생
    if isinstance(df.index, pd.DatetimeIndex):
        df["year"] = df.index.year
    else:
        df["year"] = df.index.values
    return df


def preprocess_ts(
    y: pd.Series,
    val_ratio: float = 0.2,
    lags: List[int] = [1, 2, 3, 11],
    rolls: List[int] = [3, 11],
    seasonal_m: int = 1
) -> TSPreprocessResult:
    """시계열을 시간 순서 유지한 채 train/val 분할 + (옵션) 특징 생성"""
    y = y.sort_index()
    n = len(y)
    val_h = max(1, int(n * val_ratio))
    split = n - val_h
    train, val = y.iloc[:split], y.iloc[split:]

    # 특징 기반 모델용 테이블
    # index를 datetime으로 바꿔도 무방 (여기선 그대로 사용)
    Xy = make_features(y, lags=lags, rolls=rolls).dropna()
    X = Xy.drop(columns=["y"])
    y_supervised = Xy["y"]

    # 특징 테이블도 동일 분할
    split_feat = len(Xy) - val_h
    X_tr, y_tr = X.iloc[:split_feat], y_supervised.iloc[:split_feat]
    X_va, y_va = X.iloc[split_feat:], y_supervised.iloc[split_feat:]

    # 스케일링(수치 스케일 안정화를 위해 특징에만 적용)
    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    X_va_scaled = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=X_va.columns)

    return TSPreprocessResult(
        y=y,
        train=train,
        val=val,
        val_horizon=val_h,
        X_tr=X_tr_scaled,
        y_tr=y_tr,
        X_va=X_va_scaled,
        y_va=y_va,
        scaler=scaler,
        used_lags=lags,
        used_rolls=rolls,
        seasonal_m=seasonal_m
    )


# ---------------------------------------------------------------------
# 2) 모델 선택 (간단 튜닝)
#   - 특징기반 Ridge: alpha를 TSCV로 선택
#   - ARIMA: (p,d,q) 후보 중 AIC 우선 탐색 → 이후 VAL 성능 비교
# ---------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    params: dict


def select_ridge_alpha(X_tr: pd.DataFrame, y_tr: pd.Series, alphas: np.ndarray = None, splits: int = 3) -> float:
    if alphas is None:
        alphas = np.logspace(-4, 3, 12)
    tscv = TimeSeriesSplit(n_splits=splits)
    best_alpha = None
    best_score = np.inf
    for a in alphas:
        scores = []
        for tr_idx, va_idx in tscv.split(X_tr):
            Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            ytr, yva = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            model = Ridge(alpha=a).fit(Xtr, ytr)
            pred = model.predict(Xva)
            scores.append(rmse(yva.values, pred))
        cv_rmse = np.mean(scores)
        if cv_rmse < best_score:
            best_score = cv_rmse
            best_alpha = float(a)
    return best_alpha


def select_arima_order(train: pd.Series, pmax: int = 3, d_values: List[int] = [0,1], qmax: int = 3) -> Tuple[int,int,int]:
    best_aic = np.inf
    best_order = (1,1,1)
    for p in range(0, pmax+1):
        for d in d_values:
            for q in range(0, qmax+1):
                try:
                    model = ARIMA(train, order=(p,d,q)).fit(method_kwargs={"warn_convergence":False})
                    aic = model.aic
                    if np.isfinite(aic) and aic < best_aic:
                        best_aic = aic
                        best_order = (p,d,q)
                except Exception:
                    continue
    return best_order


def model_selection(prep: TSPreprocessResult) -> Tuple[ModelSpec, ModelSpec]:
    # Ridge alpha 선택
    alpha = select_ridge_alpha(prep.X_tr, prep.y_tr)
    ridge_spec = ModelSpec(name="ridge", params={"alpha": alpha})

    # ARIMA (p,d,q) 선택 (AIC 기반)
    p,d,q = select_arima_order(prep.train)
    arima_spec = ModelSpec(name="arima", params={"order": (p,d,q)})

    return ridge_spec, arima_spec


# ---------------------------------------------------------------------
# 3) 모델 구현 (클래스화)
# ---------------------------------------------------------------------
class FeatureRidgeModel:
    def __init__(self, alpha: float, scaler: StandardScaler, feature_cols: List[str]):
        self.alpha = alpha
        self.scaler = scaler
        self.model = Ridge(alpha=alpha)
        self.feature_cols = feature_cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def forecast(self, y_full: pd.Series, horizon: int, lags: List[int], rolls: List[int]) -> np.ndarray:
        """
        고정된 모델/스케일러로 한 스텝씩 롤링 예측 (피처기반)
        - y_full: 학습까지 포함한 전체 실제 시계열
        - horizon: 예측 길이
        """
        preds = []
        y_hist = y_full.copy().astype(float)
        for _ in range(horizon):
            # 다음 시점의 피처 생성
            idx_next = y_hist.index[-1] + 1 if not isinstance(y_hist.index, pd.DatetimeIndex) else y_hist.index[-1] + (y_hist.index[-1]-y_hist.index[-2])
            tmp = make_features(y_hist.append(pd.Series([np.nan], index=[idx_next])), lags, rolls).iloc[[-1]]
            X_next = tmp.drop(columns=["y"])
            X_next = pd.DataFrame(self.scaler.transform(X_next), columns=X_next.columns, index=X_next.index)
            y_next = float(self.model.predict(X_next)[0])
            preds.append(y_next)
            y_hist.loc[idx_next] = y_next
        return np.array(preds)


class ARIMAModel:
    def __init__(self, order: Tuple[int,int,int]):
        self.order = order
        self.model_fit = None

    def fit(self, y_train: pd.Series):
        self.model_fit = ARIMA(y_train, order=self.order).fit(method_kwargs={"warn_convergence":False})
        return self

    def predict_val(self, steps: int) -> np.ndarray:
        # 학습끝 시점 이후 steps 스텝 예측
        return self.model_fit.forecast(steps=steps).values

    def forecast(self, steps: int) -> np.ndarray:
        return self.predict_val(steps)


# ---------------------------------------------------------------------
# 4) 모델 평가 (VAL 구간에서 두 모델 비교 + 베이스라인 포함)
# ---------------------------------------------------------------------
def naive_last_value_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], horizon)

def moving_average_forecast(train: pd.Series, horizon: int, window: int = 3) -> np.ndarray:
    last_ma = float(train.iloc[-window:].mean()) if len(train) >= window else float(train.mean())
    return np.repeat(last_ma, horizon)

def evaluate_on_val(
    prep: TSPreprocessResult,
    ridge_spec: ModelSpec,
    arima_spec: ModelSpec
) -> Dict[str, Dict[str, float]]:
    # 특징기반 Ridge
    ridge = FeatureRidgeModel(ridge_spec.params["alpha"], prep.scaler, list(prep.X_tr.columns))
    ridge.fit(prep.X_tr, prep.y_tr)
    ridge_pred_val = ridge.predict(prep.X_va).astype(float).ravel()

    # ARIMA
    arima = ARIMAModel(arima_spec.params["order"])
    arima.fit(prep.train)
    arima_pred_val = arima.predict_val(prep.val_horizon)

    # 베이스라인들
    naive_pred = naive_last_value_forecast(prep.train, prep.val_horizon)
    ma_pred = moving_average_forecast(prep.train, prep.val_horizon, window=3)

    y_true = prep.val.values
    results = {}
    def pack(yhat, name):
        return {
            "RMSE": rmse(y_true, yhat),
            "MAE": mae(y_true, yhat),
            "MAPE(%)": mape(y_true, yhat),
            "sMAPE(%)": smape(y_true, yhat),
            "MASE": mase(y_true, yhat, prep.train.values, m=prep.seasonal_m)
        }

    results["Ridge"] = pack(ridge_pred_val, "Ridge")
    results["ARIMA"] = pack(arima_pred_val, "ARIMA")
    results["NaiveLast"] = pack(naive_pred, "NaiveLast")
    results["MovingAvg3"] = pack(ma_pred, "MovingAvg3")

    # 콘솔 출력
    print("\n[VAL RESULTS]")
    for k, v in results.items():
        print(f" {k:>10} | RMSE={v['RMSE']:.3f}  MAE={v['MAE']:.3f}  MAPE={v['MAPE(%)']:.2f}%  sMAPE={v['sMAPE(%)']:.2f}%  MASE={v['MASE']:.3f}")

    # 승자 정보
    best_model = min(results.items(), key=lambda kv: kv[1]["RMSE"])[0]
    print(f"\n>> 선택: RMSE 기준 Best = {best_model}")
    return results


# ---------------------------------------------------------------------
# 메인 실행 데모
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) 데이터 전처리
    y = load_sunspots()  # 연간 흑점수
    prep = preprocess_ts(y, val_ratio=0.2, lags=[1,2,3,11], rolls=[3,11], seasonal_m=1)

    # 2) 모델 선택(간단 튜닝)
    ridge_spec, arima_spec = model_selection(prep)
    print(f"[Model candidates] Ridge(alpha={ridge_spec.params['alpha']:.5f}), ARIMA(order={arima_spec.params['order']})")

    # 3) 모델 구현 & 4) 평가
    _ = evaluate_on_val(prep, ridge_spec, arima_spec)
