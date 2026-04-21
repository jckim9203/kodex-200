import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from imblearn.over_sampling import SMOTE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

warnings.filterwarnings('ignore')

# ── 1. 데이터 로드 및 전처리 ─────────────────────────────────────────
data = pd.read_csv(
    'C:/Users/user/OneDrive/바탕 화면/재무/알고리즘투자/kodex200_data.csv',
    encoding='euc-kr'
)
data = data[~data.isin(['#DIV/0!', '#REF!']).any(axis=1)].reset_index(drop=True)
data['vr_10'] = data['vr_10'].astype(float)
data['rsi']   = data['rsi'].astype(float)
data['date']  = pd.to_datetime(data['date'].astype(str), format='%Y%m%d')
# 예: 200일 이동평균 위에 있을 때만 매수 신호 유효
data['trend_filter'] = (data['sma_10'] > data['sma_10'].rolling(200).mean()).astype(int)
data = data.drop(columns=['stochastic_k_10', 'stochastic_d_3', '1d_after_price'])
data = data.sort_values('date').reset_index(drop=True)

# 숫자형 변환 (문자열로 읽힌 컬럼 처리)
numeric_cols = [c for c in data.columns if c not in ['date', '5d_after_price']]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna().reset_index(drop=True)

# ── 2. 피처 엔지니어링 ───────────────────────────────────────────────
def add_features(df):
    df = df.copy()

    # Lag 피처 (1~5일 전 값)
    lag_cols = ['sma_10', 'rsi', 'macd', 'obv', 'cci_10', 'atr_10', 'disparity']
    for col in lag_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 5]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Rolling 통계 (3일, 5일)
    rolling_cols = ['rsi', 'macd', 'disparity', 'cci_10']
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_roll3_mean'] = df[col].shift(1).rolling(3).mean()
            df[f'{col}_roll5_std']  = df[col].shift(1).rolling(5).std()

    # 가격 모멘텀: SMA 대비 현재 가격 변화율
    if 'sma_10' in df.columns and 'vwma_10' in df.columns:
        df['sma_vwma_diff'] = df['sma_10'] - df['vwma_10']

    # RSI 과매수/과매도 구간
    if 'rsi' in df.columns:
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold']   = (df['rsi'] < 30).astype(int)

    # Bollinger Band Width
    if 'bollinger_upper_10' in df.columns and 'bollinger_lower_10' in df.columns:
        df['bb_width'] = df['bollinger_upper_10'] - df['bollinger_lower_10']
        df['bb_pos']   = (df['sma_10'] - df['bollinger_lower_10']) / (df['bb_width'] + 1e-9)

    # MACD 방향
    if 'macd' in df.columns:
        df['macd_diff'] = df['macd'].diff()
        df['macd_sign'] = (df['macd'] > 0).astype(int)

    # OBV 변화율
    if 'obv' in df.columns:
        df['obv_change'] = df['obv'].diff()

    # 월/분기 (계절성)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    # add_features 함수 안, return df 바로 위
    if 'sma_10' in df.columns:
        df['trend_filter'] = (
            df['sma_10'] > df['sma_10'].rolling(20).mean()
        ).astype(int)
    return df

data = add_features(data)
data = data.dropna().reset_index(drop=True)

# ── 3. 피처 / 타겟 분리 ──────────────────────────────────────────────
date   = data['date']
X      = data.drop(columns=['date', '5d_after_price'])
y      = data['5d_after_price']

encoder = LabelEncoder()
y_enc   = encoder.fit_transform(y)

# ── 4. TimeSeriesSplit Train/Test ────────────────────────────────────
split_idx = int(len(X) * 0.8)
X_train, X_test   = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test   = y_enc[:split_idx], y_enc[split_idx:]
dates_test        = date.iloc[split_idx:].reset_index(drop=True)
X_tomorrow        = X.iloc[[-1]]

# ── 5. 클래스 불균형 처리 (SMOTE) ────────────────────────────────────
print("클래스 분포 (원본):", np.bincount(y_train))
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("클래스 분포 (SMOTE):", np.bincount(y_train_res))

# ── 6. Optuna 하이퍼파라미터 튜닝 ────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

def objective_rf(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 300, 1000),
        'max_depth'        : trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features'     : trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state'     : 42,
        'n_jobs'           : -1,
    }
    clf = RandomForestClassifier(**params)
    scores = cross_val_score(clf, X_train_res, y_train_res, cv=tscv, scoring='f1', n_jobs=-1)
    return scores.mean()

print("\nOptuna RF 튜닝 중 (10 trials)...")
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=10, show_progress_bar=False)
best_rf_params = study_rf.best_params
print(f"Best RF params: {best_rf_params}")

sample_weight = np.linspace(0.5, 1.0, len(X_train_res))

rf_model = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res, sample_weight=sample_weight)

# XGBoost 튜닝
if XGB_AVAILABLE:
    def objective_xgb(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 200, 800),
            'max_depth'       : trial.suggest_int('max_depth', 3, 8),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'use_label_encoder': False,
            'eval_metric'     : 'logloss',
            'random_state'    : 42,
            'n_jobs'          : -1,
        }
        clf = xgb.XGBClassifier(**params)
        scores = cross_val_score(clf, X_train_res, y_train_res, cv=tscv, scoring='f1', n_jobs=-1)
        return scores.mean()

    print("\nOptuna XGBoost 튜닝 중 (10 trials)...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=10, show_progress_bar=False)
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({'use_label_encoder': False, 'eval_metric': 'logloss',
                             'random_state': 42, 'n_jobs': -1})
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train_res, y_train_res, sample_weight=sample_weight)
    print(f"Best XGB params: {best_xgb_params}")

# LightGBM 튜닝
if LGB_AVAILABLE:
    def objective_lgb(trial):
        params = {
            'n_estimators'  : trial.suggest_int('n_estimators', 200, 800),
            'max_depth'     : trial.suggest_int('max_depth', 3, 8),
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves'    : trial.suggest_int('num_leaves', 20, 100),
            'subsample'     : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state'  : 42,
            'n_jobs'        : -1,
            'verbose'       : -1,
        }
        clf = lgb.LGBMClassifier(**params)
        scores = cross_val_score(clf, X_train_res, y_train_res, cv=tscv, scoring='f1', n_jobs=-1)
        return scores.mean()

    print("\nOptuna LightGBM 튜닝 중 (10 trials)...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=10, show_progress_bar=False)
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    lgb_model = lgb.LGBMClassifier(**best_lgb_params)
    lgb_model.fit(X_train_res, y_train_res, sample_weight=sample_weight)
    print(f"Best LGB params: {best_lgb_params}")

# ── 7. Voting 앙상블 ──────────────────────────────────────────────────
estimators = [('rf', rf_model)]
if XGB_AVAILABLE:
    estimators.append(('xgb', xgb_model))
if LGB_AVAILABLE:
    estimators.append(('lgb', lgb_model))

if len(estimators) > 1:
    voting_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting_model.fit(X_train_res, y_train_res)
    final_model = voting_model
    print(f"\n앙상블 모델 사용: {[name for name, _ in estimators]}")
else:
    final_model = rf_model
    print("\nRF 단일 모델 사용")

# ── 7.5 Calibration + Threshold 조정 ────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV

# Calibration (train 일부를 validation으로 사용)
val_split = int(len(X_train_res) * 0.8)
X_cal = X_train_res.iloc[val_split:]
y_cal = y_train_res[val_split:]

calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_cal, y_cal)

# Threshold 조정 (0.65로 설정, 0.6~0.75 사이에서 조절)
THRESHOLD = 0.65
y_pred_prob = calibrated_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= THRESHOLD).astype(int)  # ← 기존 y_pred 덮어쓰기
print(f"Threshold {THRESHOLD} 적용 후 UP 예측 수: {y_pred.sum()} / {len(y_pred)}")

# ── 8. 전체 테스트셋 평가 ─────────────────────────────────────────────
y_pred      = final_model.predict(X_test)
y_pred_prob = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else None

print("\n" + "="*60)
print("  전체 테스트셋 평가")
print("="*60)
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (UP=1): {precision_score(y_test, y_pred, pos_label=1):.4f}")
print(f"Recall    (UP=1): {recall_score(y_test, y_pred, pos_label=1):.4f}")
print(f"F1-Score  (UP=1): {f1_score(y_test, y_pred, pos_label=1):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=encoder.classes_.astype(str)))

# ── 9. 월별 정확도 / Precision / Recall ──────────────────────────────
result = pd.DataFrame({
    'Date'      : dates_test.values,
    'real'      : y_test,
    'pred'      : y_pred,
}).assign(
    YearMonth=lambda df: pd.to_datetime(df['Date']).dt.to_period('M')
)

print("\n" + "="*60)
print("  월별 성능 지표 (test set)")
print("="*60)

monthly_rows = []
for ym, grp in result.groupby('YearMonth'):
    real = grp['real'].values
    pred = grp['pred'].values
    n    = len(real)
    acc  = accuracy_score(real, pred)
    # 해당 월에 예측값/실제값에 클래스 1이 있어야 precision/recall 계산 가능
    prec = precision_score(real, pred, pos_label=1, zero_division=0)
    rec  = recall_score(real, pred, pos_label=1, zero_division=0)
    f1   = f1_score(real, pred, pos_label=1, zero_division=0)
    pred_up   = int((pred == 1).sum())
    actual_up = int((real == 1).sum())
    monthly_rows.append({
        '월'           : str(ym),
        '샘플수'       : n,
        'Accuracy'     : round(acc, 4),
        'Precision(↑)' : round(prec, 4),
        'Recall(↑)'    : round(rec, 4),
        'F1(↑)'        : round(f1, 4),
        '예측UP'       : pred_up,
        '실제UP'       : actual_up,
    })

monthly_df = pd.DataFrame(monthly_rows)
print(monthly_df.to_string(index=False))

# 요약 통계
print("\n[월별 평균]")
for col in ['Accuracy', 'Precision(↑)', 'Recall(↑)', 'F1(↑)']:
    print(f"  {col}: {monthly_df[col].mean():.4f}  (std: {monthly_df[col].std():.4f})")


# ── 8.5 모델 성능 vs 시장 편향 분리 분석 ─────────────────────────────
from sklearn.dummy import DummyClassifier

print("\n" + "="*60)
print("  모델 성능 vs 시장 편향 분리 분석")
print("="*60)

# 1) 기준선: 항상 UP만 예측하는 Dummy
dummy = DummyClassifier(strategy='constant', constant=1)
dummy.fit(X_train_res, y_train_res)
y_dummy = dummy.predict(X_test)

print("\n[Dummy: 항상 UP 예측]")
print(f"  Accuracy  : {accuracy_score(y_test, y_dummy):.4f}")
print(f"  Precision : {precision_score(y_test, y_dummy, pos_label=1, zero_division=0):.4f}  ← 실제 UP 비율 (base rate)")
print(f"  Recall    : {recall_score(y_test, y_dummy, pos_label=1, zero_division=0):.4f}")

print("\n[우리 모델]")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")

# 2) Precision Lift: 모델이 dummy 대비 얼마나 더 잘하는지
base_precision = precision_score(y_test, y_dummy, pos_label=1, zero_division=0)
model_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
lift = model_precision / base_precision if base_precision > 0 else float('inf')

print(f"\n[Precision Lift]")
print(f"  Base rate (항상UP precision) : {base_precision:.4f}")
print(f"  모델 precision               : {model_precision:.4f}")
print(f"  Lift                         : {lift:.4f}x")
print(f"  → {'모델 실력 있음 (Lift > 1.1)' if lift > 1.1 else '시장 상승 효과가 대부분 (Lift ≈ 1.0)'}")

# 3) 월별 UP 비율 vs 모델 precision 비교
print("\n[월별 실제 UP 비율 vs 모델 Precision 비교]")
monthly_compare = []
for ym, grp in result.groupby('YearMonth'):
    real = grp['real'].values
    pred = grp['pred'].values
    actual_up_rate = real.mean()
    model_prec = precision_score(real, pred, pos_label=1, zero_division=0)
    diff = model_prec - actual_up_rate
    monthly_compare.append({
        '월'           : str(ym),
        '실제UP비율'   : round(actual_up_rate, 4),
        '모델Precision': round(model_prec, 4),
        '차이(+면모델우위)': round(diff, 4),
    })

mc_df = pd.DataFrame(monthly_compare)
print(mc_df.to_string(index=False))

avg_diff = mc_df['차이(+면모델우위)'].mean()
print(f"\n평균 차이: {avg_diff:+.4f}")
print(f"→ {'모델이 시장 편향 이상의 실력 있음' if avg_diff > 0 else '시장 상승이 precision을 끌어올린 효과가 큼'}")


# ── 10. 피처 중요도 ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  피처 중요도 Top 20 (RF 기준)")
print("="*60)
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
print(feat_imp.sort_values(ascending=False).head(20).to_string())

# ── 11. 내일 예측 ─────────────────────────────────────────────────────
# final_model → calibrated_model 로 변경
tomorrow_proba = calibrated_model.predict_proba(X_tomorrow)[0]
tomorrow_pred  = int(tomorrow_proba[1] >= THRESHOLD)
invest_guide   = "IN" if tomorrow_pred == 1 else "STAY"
target_date    = date.iloc[-1]

print("\n" + "="*60)
print("  내일 예측")
print("="*60)
print(f"Target date     : {target_date.date()}")
print(f"Prediction      : {encoder.classes_[tomorrow_pred]} → {invest_guide}")
print(f"Class prob (UP) : {tomorrow_proba[1]:.4f}")
print(f"Class prob (DN) : {tomorrow_proba[0]:.4f}")

# ── 12. 결과 저장 ─────────────────────────────────────────────────────
result_out = result.copy()
result_out['real_label'] = encoder.inverse_transform(result_out['real'])
result_out['pred_label'] = encoder.inverse_transform(result_out['pred'])
result_out['correct']    = (result_out['real'] == result_out['pred']).astype(int)
result_out.to_csv(
    'C:/Users/user/OneDrive/바탕 화면/재무/알고리즘투자/test_result_v2.csv',
    index=False, encoding='utf-8-sig'
)
print("\n결과 저장 완료: test_result_v1.csv")