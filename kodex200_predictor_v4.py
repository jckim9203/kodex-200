import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
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
data['date'] = pd.to_datetime(data['date'].astype(str), format='%Y%m%d')
data = data.drop(columns=['stochastic_k_10', 'stochastic_d_3', '1d_after_price'])
data = data.sort_values('date').reset_index(drop=True)

numeric_cols = [c for c in data.columns if c not in ['date', '5d_after_price']]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna().reset_index(drop=True)

# ── 2. 피처 엔지니어링 ───────────────────────────────────────────────
def add_features(df):
    df = df.copy()

    # Lag 피처
    lag_cols = ['sma_10', 'rsi', 'macd', 'obv', 'cci_10', 'atr_10', 'disparity',
                'vr_10', 'ema_10']
    for col in lag_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 5]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Rolling 통계
    for col in ['rsi', 'macd', 'disparity', 'cci_10', 'vr_10']:
        if col in df.columns:
            df[f'{col}_roll3_mean'] = df[col].shift(1).rolling(3).mean()
            df[f'{col}_roll5_mean'] = df[col].shift(1).rolling(5).mean()
            df[f'{col}_roll5_std']  = df[col].shift(1).rolling(5).std()

    # Bollinger Band
    if 'bollinger_upper_10' in df.columns and 'bollinger_lower_10' in df.columns:
        df['bb_width'] = df['bollinger_upper_10'] - df['bollinger_lower_10']
        df['bb_pos']   = (df['sma_10'] - df['bollinger_lower_10']) / (df['bb_width'] + 1e-9)
        df['bb_width_lag1'] = df['bb_width'].shift(1)

    # MACD
    if 'macd' in df.columns:
        df['macd_diff'] = df['macd'].diff()
        df['macd_sign'] = (df['macd'] > 0).astype(int)
        df['macd_cross'] = (
            (df['macd'] > 0) & (df['macd'].shift(1) <= 0)
        ).astype(int)  # MACD 골든크로스

    # RSI
    if 'rsi' in df.columns:
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold']   = (df['rsi'] < 30).astype(int)
        df['rsi_diff']       = df['rsi'].diff()

    # OBV
    if 'obv' in df.columns:
        df['obv_change']       = df['obv'].diff()
        df['obv_change_ratio'] = df['obv'].pct_change().clip(-5, 5)

    # SMA/EMA 괴리
    if 'sma_10' in df.columns and 'ema_10' in df.columns:
        df['sma_ema_diff'] = df['sma_10'] - df['ema_10']

    # 추세 필터 (20일, 60일 이동평균 대비)
    if 'sma_10' in df.columns:
        df['trend_20'] = (df['sma_10'] > df['sma_10'].rolling(20).mean()).astype(int)
        df['trend_60'] = (df['sma_10'] > df['sma_10'].rolling(60).mean()).astype(int)

    # 월/분기 계절성
    df['month']   = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['weekday'] = df['date'].dt.dayofweek  # 요일 효과

    return df

data = add_features(data)
data = data.dropna().reset_index(drop=True)
print(f"데이터 shape: {data.shape}")
print(f"기간: {data['date'].min().date()} ~ {data['date'].max().date()}")

# ── 3. 피처 / 타겟 분리 ──────────────────────────────────────────────
date = data['date']
X    = data.drop(columns=['date', '5d_after_price'])
y    = data['5d_after_price']

encoder = LabelEncoder()
y_enc   = encoder.fit_transform(y)
print(f"클래스: {encoder.classes_}  (UP=1, DOWN=0)")
print(f"전체 UP 비율: {y_enc.mean():.4f}")

# ── 4. Train/Validation/Test Split ───────────────────────────────────
# train: 80%, validation(calibration용): 10%, test: 10%
# 시계열 순서 유지
n = len(X)
train_end = int(n * 0.7)
val_end   = int(n * 0.8)


X_train = X.iloc[:train_end]
y_train = y_enc[:train_end]
X_val   = X.iloc[train_end:val_end]
y_val   = y_enc[train_end:val_end]
X_test  = X.iloc[val_end:]
y_test  = y_enc[val_end:]
dates_test = date.iloc[val_end:].reset_index(drop=True)
X_tomorrow = X.iloc[[-1]]

print(f"\nTrain: {len(X_train)}일 ({date.iloc[0].date()} ~ {date.iloc[train_end-1].date()})")
print(f"Val  : {len(X_val)}일  ({date.iloc[train_end].date()} ~ {date.iloc[val_end-1].date()})")
print(f"Test : {len(X_test)}일  ({date.iloc[val_end].date()} ~ {date.iloc[-1].date()})")
print(f"Train UP 비율: {y_train.mean():.4f}")
print(f"Test  UP 비율: {y_test.mean():.4f}")

# ── 5. Optuna: Precision 기준으로 튜닝 ───────────────────────────────
# F1 대신 precision을 최적화 목표로 변경
tscv = TimeSeriesSplit(n_splits=5)

def objective_rf(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 300, 1000),
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features'     : trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight'     : trial.suggest_categorical('class_weight', [None, 'balanced']),
        'random_state': 42, 'n_jobs': -1,
    }
    clf = RandomForestClassifier(**params)
    # precision 기준으로 튜닝 (핵심 변경)
    scores = cross_val_score(clf, X_train, y_train,
                             cv=tscv, scoring='precision', n_jobs=-1)
    return scores.mean()

print("\nOptuna RF 튜닝 중 (10 trials, precision 기준)...")
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=10, show_progress_bar=False)
best_rf_params = study_rf.best_params
print(f"Best RF params: {best_rf_params}")

rf_model = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

if XGB_AVAILABLE:
    def objective_xgb(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 200, 800),
            'max_depth'       : trial.suggest_int('max_depth', 3, 7),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
            'use_label_encoder': False, 'eval_metric': 'logloss',
            'random_state': 42, 'n_jobs': -1,
        }
        clf = xgb.XGBClassifier(**params)
        scores = cross_val_score(clf, X_train, y_train,
                                 cv=tscv, scoring='precision', n_jobs=-1)
        return scores.mean()

    print("\nOptuna XGBoost 튜닝 중 (10 trials, precision 기준)...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=10, show_progress_bar=False)
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({'use_label_encoder': False, 'eval_metric': 'logloss',
                             'random_state': 42, 'n_jobs': -1})
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    print(f"Best XGB params: {best_xgb_params}")

if LGB_AVAILABLE:
    def objective_lgb(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 200, 800),
            'max_depth'       : trial.suggest_int('max_depth', 3, 7),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves'      : trial.suggest_int('num_leaves', 20, 80),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'is_unbalance'    : trial.suggest_categorical('is_unbalance', [True, False]),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        clf = lgb.LGBMClassifier(**params)
        scores = cross_val_score(clf, X_train, y_train,
                                 cv=tscv, scoring='precision', n_jobs=-1)
        return scores.mean()

    print("\nOptuna LightGBM 튜닝 중 (10 trials, precision 기준)...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=10, show_progress_bar=False)
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    lgb_model = lgb.LGBMClassifier(**best_lgb_params)
    lgb_model.fit(X_train, y_train)
    print(f"Best LGB params: {best_lgb_params}")

# ── 6. Voting 앙상블 ─────────────────────────────────────────────────
estimators = [('rf', rf_model)]
if XGB_AVAILABLE:
    estimators.append(('xgb', xgb_model))
if LGB_AVAILABLE:
    estimators.append(('lgb', lgb_model))

if len(estimators) > 1:
    voting_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting_model.fit(X_train, y_train)
    base_model = voting_model
    print(f"\n앙상블: {[n for n, _ in estimators]}")
else:
    base_model = rf_model

# ── 7. Calibration (validation set으로) ──────────────────────────────
print("\nCalibration 적용 중...")
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_val, y_val)

# ── 8. Threshold 최적화 (validation set에서 precision 최대화) ─────────
print("\nValidation set에서 최적 Threshold 탐색...")
val_proba = calibrated_model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_precision = 0.0
threshold_results = []

for thresh in np.arange(0.50, 0.75, 0.01):
    val_pred_t = (val_proba >= thresh).astype(int)
    n_pred_up  = val_pred_t.sum()
    if n_pred_up < 10:  # 예측이 너무 적으면 스킵
        continue
    prec = precision_score(y_val, val_pred_t, pos_label=1, zero_division=0)
    rec  = recall_score(y_val, val_pred_t, pos_label=1, zero_division=0)
    threshold_results.append({'threshold': thresh, 'precision': prec,
                               'recall': rec, 'n_pred_up': n_pred_up})
    if prec > best_precision:
        best_precision = prec
        best_threshold = thresh

thresh_df = pd.DataFrame(threshold_results)
print("\n[Threshold별 Validation 성능 (주요 구간)]")
print(thresh_df[thresh_df['threshold'].isin([0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
               ].to_string(index=False))
print(f"\n→ 최적 Threshold: {best_threshold:.2f}  (Precision: {best_precision:.4f})")

THRESHOLD = best_threshold

# ── 9. Test set 평가 ──────────────────────────────────────────────────
y_pred_prob = calibrated_model.predict_proba(X_test)[:, 1]
y_pred      = (y_pred_prob >= THRESHOLD).astype(int)

print("\n" + "="*60)
print(f"  전체 테스트셋 평가 (Threshold={THRESHOLD:.2f})")
print("="*60)
print(f"UP 예측 수    : {y_pred.sum()} / {len(y_pred)}")
print(f"Accuracy      : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision(UP) : {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"Recall(UP)    : {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"F1(UP)        : {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=encoder.classes_.astype(str)))

# ── 10. 시장 편향 분석 ───────────────────────────────────────────────
base_prec  = y_test.mean()
model_prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
lift       = model_prec / base_prec if base_prec > 0 else float('nan')

print("\n" + "="*60)
print("  시장 편향 vs 모델 실력 분석")
print("="*60)
print(f"Test 기간 실제 UP 비율 (base rate) : {base_prec:.4f}")
print(f"모델 Precision                     : {model_prec:.4f}")
print(f"Lift                               : {lift:.4f}x")
print(f"→ {'UP 예측시 base rate 대비 우위 → 투자 활용 가능' if model_prec > base_prec else 'base rate 하회 → 모델 개선 필요'}")

# ── 11. 월별 성능 ────────────────────────────────────────────────────
result = pd.DataFrame({
    'Date'   : dates_test.values,
    'real'   : y_test,
    'pred'   : y_pred,
    'prob_up': y_pred_prob,
}).assign(YearMonth=lambda df: pd.to_datetime(df['Date']).dt.to_period('M'))

print("\n" + "="*60)
print("  월별 성능 지표")
print("="*60)
monthly_rows = []
for ym, grp in result.groupby('YearMonth'):
    real = grp['real'].values
    pred = grp['pred'].values
    up_rate   = real.mean()
    n_pred_up = pred.sum()
 
    if n_pred_up == 0:
        # UP 예측이 단 1건도 없는 달 → 성능 평가 불가
        monthly_rows.append({
            '월'       : str(ym),
            'N'        : len(real),
            'UP비율'   : round(up_rate, 3),
            'N_pred_UP': 0,
            'Precision': '-',
            'Recall'   : 0.0,
            'F1'       : '-',
            '차이'     : '-',
            '판정'     : '- 신호없음',
        })
    else:
        prec    = precision_score(real, pred, pos_label=1, zero_division=0)
        rec     = recall_score(real, pred, pos_label=1, zero_division=0)
        f1      = f1_score(real, pred, pos_label=1, zero_division=0)
        diff    = prec - up_rate
        verdict = '✓ 모델우위' if diff > 0 else '✗ 시장편향'
        monthly_rows.append({
            '월'       : str(ym),
            'N'        : len(real),
            'UP비율'   : round(up_rate, 3),
            'N_pred_UP': int(n_pred_up),
            'Precision': round(prec, 3),
            'Recall'   : round(rec, 3),
            'F1'       : round(f1, 3),
            '차이'     : round(diff, 3),
            '판정'     : verdict,
        })

monthly_df = pd.DataFrame(monthly_rows)
print(monthly_df.to_string(index=False))

# 요약 통계: 신호없음 달은 제외하고 집계
scored_df   = monthly_df[monthly_df['판정'] != '- 신호없음']
no_sig_cnt  = len(monthly_df) - len(scored_df)
avg_diff    = scored_df['차이'].mean()   if len(scored_df) > 0 else float('nan')
win_months  = (scored_df['차이'] > 0).sum()
win_rate    = (scored_df['차이'] > 0).mean() if len(scored_df) > 0 else float('nan')
 
print(f"\n평균 차이      : {avg_diff:+.4f}  (신호없음 {no_sig_cnt}개월 제외)")
print(f"모델우위 월수  : {int(win_months)} / {len(scored_df)}개월 ({win_rate:.1%})  (신호없음 제외)")
if no_sig_cnt > 0:
    print(f"신호없음 월수  : {no_sig_cnt}개월  (해당 월 UP 예측 0건 → 성능 평가 불가)")
print(f"→ {'모델이 base rate 이상의 실력 있음' if avg_diff > 0 else 'base rate 하회 → 개선 필요'}")

# ── 12. 피처 중요도 ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  피처 중요도 Top 20 (RF 기준)")
print("="*60)
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
print(feat_imp.sort_values(ascending=False).head(20).to_string())

# ── 13. 내일 예측 ────────────────────────────────────────────────────
# 전체 데이터로 재학습
print("\n전체 데이터로 최종 재학습 중...")
rf_model.fit(X, y_enc)
if XGB_AVAILABLE:
    xgb_model.fit(X, y_enc)
if LGB_AVAILABLE:
    lgb_model.fit(X, y_enc)
if len(estimators) > 1:
    voting_model.fit(X, y_enc)
    final_base = voting_model
else:
    final_base = rf_model

final_calibrated = CalibratedClassifierCV(final_base, method='isotonic', cv='prefit')
final_calibrated.fit(X_val, y_val)  # calibration은 val set 유지

tomorrow_proba = final_calibrated.predict_proba(X_tomorrow)[0]
tomorrow_pred  = int(tomorrow_proba[1] >= THRESHOLD)
invest_guide   = "IN" if tomorrow_pred == 1 else "STAY"

# ── 14. 결과 저장 ────────────────────────────────────────────────────
result_out = result.copy()
result_out['real_label'] = encoder.inverse_transform(result_out['real'])
result_out['pred_label'] = encoder.inverse_transform(result_out['pred'])
result_out['correct']    = (result_out['real'] == result_out['pred']).astype(int)
result_out.to_csv(
    'C:/Users/user/OneDrive/바탕 화면/재무/알고리즘투자/test_result_v3.csv',
    index=False, encoding='utf-8-sig'
)
today_prob = result_out.iloc[-1]['prob_up']

print("\n" + "="*60)
print("  내일 예측")
print("="*60)
print(f"Target date     : {date.iloc[-1].date()}")
print(f"Prediction      : {encoder.classes_[tomorrow_pred]} → {invest_guide}")
print(f"Prob UP         : {today_prob:.4f}")
print(f"Threshold       : {THRESHOLD:.2f}")
print("\n결과 저장 완료: test_result_v4.csv")