import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
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

# ── 페이지 설정 ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="KODEX200 ML 예측기",
    page_icon="📈",
    layout="wide"
)
st.title("📈 KODEX200 5일 후 방향 예측")
st.caption("CSV 파일을 업로드하면 모델을 학습하고 내일 매매 신호를 알려드립니다.")

# ── 피처 엔지니어링 함수 ──────────────────────────────────────────────
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
        ).astype(int)
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
    # 추세 필터
    if 'sma_10' in df.columns:
        df['trend_20'] = (df['sma_10'] > df['sma_10'].rolling(20).mean()).astype(int)
        df['trend_60'] = (df['sma_10'] > df['sma_10'].rolling(60).mean()).astype(int)
    # 월/분기/요일 계절성
    df['month']   = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['weekday'] = df['date'].dt.dayofweek
    return df

# ── 메인 로직 ─────────────────────────────────────────────────────────
uploaded = st.file_uploader("📂 CSV 파일 업로드 (kodex200_data.csv)", type=["csv"])
if uploaded is None:
    st.info("👆 위에서 CSV 파일을 업로드해 주세요.")
    st.stop()

# ── 데이터 로드 ───────────────────────────────────────────────────────
with st.spinner("데이터 로드 중..."):
    try:
        try:
            data = pd.read_csv(uploaded, encoding='euc-kr')
        except UnicodeDecodeError:
            uploaded.seek(0)
            data = pd.read_csv(uploaded, encoding='utf-8')
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        st.stop()

    data = data[~data.isin(['#DIV/0!', '#REF!']).any(axis=1)].reset_index(drop=True)
    data['date'] = pd.to_datetime(data['date'].astype(str), format='%Y%m%d')
    drop_cols = [c for c in ['stochastic_k_10', 'stochastic_d_3', '1d_after_price'] if c in data.columns]
    data = data.drop(columns=drop_cols)
    data = data.sort_values('date').reset_index(drop=True)
    numeric_cols = [c for c in data.columns if c not in ['date', '5d_after_price']]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna().reset_index(drop=True)
    data = add_features(data)
    data = data.dropna().reset_index(drop=True)

col1, col2, col3 = st.columns(3)
col1.metric("총 데이터", f"{len(data):,}일")
col2.metric("시작일", str(data['date'].min().date()))
col3.metric("종료일", str(data['date'].max().date()))

# ── 피처/타겟 분리 ────────────────────────────────────────────────────
date_series = data['date']
X    = data.drop(columns=['date', '5d_after_price'])
y    = data['5d_after_price']
encoder = LabelEncoder()
y_enc   = encoder.fit_transform(y)

n         = len(X)
train_end = int(n * 0.7)
val_end   = int(n * 0.8)
X_train = X.iloc[:train_end];       y_train = y_enc[:train_end]
X_val   = X.iloc[train_end:val_end]; y_val  = y_enc[train_end:val_end]
X_test  = X.iloc[val_end:];          y_test = y_enc[val_end:]
dates_test  = date_series.iloc[val_end:].reset_index(drop=True)
X_tomorrow  = X.iloc[[-1]]

# ── 학습 ─────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
progress = st.progress(0, text="모델 학습을 시작합니다...")

def tune_rf():
    def objective(trial):
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
        scores = cross_val_score(clf, X_train, y_train, cv=tscv, scoring='precision', n_jobs=-1)
        return scores.mean()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, show_progress_bar=False)
    return study.best_params

progress.progress(10, text="🌲 RandomForest 튜닝 중... (1/3)")
best_rf = tune_rf()
rf_model = RandomForestClassifier(**best_rf, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
estimators = [('rf', rf_model)]

if XGB_AVAILABLE:
    progress.progress(35, text="⚡ XGBoost 튜닝 중... (2/3)")
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
        scores = cross_val_score(clf, X_train, y_train, cv=tscv, scoring='precision', n_jobs=-1)
        return scores.mean()
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=10, show_progress_bar=False)
    best_xgb = study_xgb.best_params
    best_xgb.update({'use_label_encoder': False, 'eval_metric': 'logloss',
                     'random_state': 42, 'n_jobs': -1})
    xgb_model = xgb.XGBClassifier(**best_xgb)
    xgb_model.fit(X_train, y_train)
    estimators.append(('xgb', xgb_model))

if LGB_AVAILABLE:
    progress.progress(60, text="💡 LightGBM 튜닝 중... (3/3)")
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
        scores = cross_val_score(clf, X_train, y_train, cv=tscv, scoring='precision', n_jobs=-1)
        return scores.mean()
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=10, show_progress_bar=False)
    best_lgb = study_lgb.best_params
    best_lgb.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    lgb_model = lgb.LGBMClassifier(**best_lgb)
    lgb_model.fit(X_train, y_train)
    estimators.append(('lgb', lgb_model))

progress.progress(75, text="🔗 앙상블 및 Calibration 중...")
if len(estimators) > 1:
    voting_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting_model.fit(X_train, y_train)
    base_model = voting_model
else:
    base_model = rf_model

calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_val, y_val)

# ── Threshold 최적화 ──────────────────────────────────────────────────
progress.progress(88, text="🎯 Threshold 최적화 중...")
val_proba = calibrated_model.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_precision = 0.0
threshold_results = []
for thresh in np.arange(0.50, 0.75, 0.01):
    val_pred_t = (val_proba >= thresh).astype(int)
    if val_pred_t.sum() < 10:
        continue
    prec = precision_score(y_val, val_pred_t, pos_label=1, zero_division=0)
    rec  = recall_score(y_val, val_pred_t, pos_label=1, zero_division=0)
    threshold_results.append({'threshold': round(thresh, 2), 'precision': round(prec, 4),
                               'recall': round(rec, 4), 'n_pred_up': int(val_pred_t.sum())})
    if prec > best_precision:
        best_precision = prec
        best_threshold = thresh
THRESHOLD = best_threshold

# ── 테스트 평가 ───────────────────────────────────────────────────────
progress.progress(95, text="📊 테스트셋 평가 중...")
y_pred_prob = calibrated_model.predict_proba(X_test)[:, 1]
y_pred      = (y_pred_prob >= THRESHOLD).astype(int)
acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec   = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1    = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
base  = y_test.mean()
lift  = prec / base if base > 0 else float('nan')

# ── 월별 성능 ─────────────────────────────────────────────────────────
result = pd.DataFrame({
    'Date': dates_test.values, 'real': y_test, 'pred': y_pred, 'prob_up': y_pred_prob
}).assign(YearMonth=lambda df: pd.to_datetime(df['Date']).dt.to_period('M'))

monthly_rows = []
for ym, grp in result.groupby('YearMonth'):
    real      = grp['real'].values
    pred      = grp['pred'].values
    up_rate   = real.mean()
    n_pred_up = pred.sum()

    if n_pred_up == 0:
        # UP 예측이 단 1건도 없는 달 → 성능 평가 불가
        monthly_rows.append({
            '월': str(ym), 'N': len(real), 'UP비율': round(up_rate, 3),
            'N_pred_UP': 0,
            'Precision': None, 'Recall': 0.0, 'F1': None,
            '차이': None, '판정': '⬜ 신호없음'
        })
    else:
        p    = precision_score(real, pred, pos_label=1, zero_division=0)
        r    = recall_score(real, pred, pos_label=1, zero_division=0)
        f    = f1_score(real, pred, pos_label=1, zero_division=0)
        diff = p - up_rate
        monthly_rows.append({
            '월': str(ym), 'N': len(real), 'UP비율': round(up_rate, 3),
            'N_pred_UP': int(n_pred_up),
            'Precision': round(p, 3), 'Recall': round(r, 3), 'F1': round(f, 3),
            '차이': round(diff, 3), '판정': '✅ 모델우위' if diff > 0 else '❌ 시장편향'
        })
monthly_df = pd.DataFrame(monthly_rows)

# 요약 통계: 신호없음 달 제외
scored_df    = monthly_df[monthly_df['판정'] != '⬜ 신호없음']
no_sig_cnt   = len(monthly_df) - len(scored_df)
avg_diff     = scored_df['차이'].mean()        if len(scored_df) > 0 else float('nan')
win_months   = (scored_df['차이'] > 0).sum()
win_rate_val = (scored_df['차이'] > 0).mean() if len(scored_df) > 0 else float('nan')

# ── 최종 재학습 및 내일 예측 ──────────────────────────────────────────
rf_model.fit(X, y_enc)
if XGB_AVAILABLE: xgb_model.fit(X, y_enc)
if LGB_AVAILABLE: lgb_model.fit(X, y_enc)
if len(estimators) > 1:
    voting_model.fit(X, y_enc)
    final_base = voting_model
else:
    final_base = rf_model
final_calibrated = CalibratedClassifierCV(final_base, method='isotonic', cv='prefit')
final_calibrated.fit(X_val, y_val)

tomorrow_proba   = final_calibrated.predict_proba(X_tomorrow)[0]
tomorrow_pred    = int(tomorrow_proba[1] >= THRESHOLD)
tomorrow_prob_up = tomorrow_proba[1]
invest_guide     = "IN 🟢" if tomorrow_pred == 1 else "STAY 🔴"
tomorrow_label   = encoder.classes_[tomorrow_pred]

progress.progress(100, text="✅ 완료!")
progress.empty()

# ═══════════════════════════════════════════════════════════════════════
# 결과 화면
# ═══════════════════════════════════════════════════════════════════════
st.divider()


# ── 섹션 2: 테스트셋 성능 지표 ───────────────────────────────────────
st.subheader("📊 테스트셋 성능 지표")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Accuracy",       f"{acc:.3f}")
m2.metric("Precision(UP)",  f"{prec:.3f}")
m3.metric("Recall(UP)",     f"{rec:.3f}")
m4.metric("F1(UP)",         f"{f1:.3f}")
m5.metric("Base Rate",      f"{base:.3f}")
m6.metric("Lift",           f"{lift:.2f}x",
          delta="모델 우위" if prec > base else "개선 필요",
          delta_color="normal" if prec > base else "inverse")
st.caption(f"Test 기간: {dates_test.iloc[0].date()} ~ {dates_test.iloc[-1].date()}  |  "
           f"UP 예측 수: {y_pred.sum()} / {len(y_pred)}일")
with st.expander("Classification Report 보기"):
    report_str = classification_report(y_test, y_pred,
                                       target_names=encoder.classes_.astype(str))
    st.code(report_str)
st.divider()

# ── 섹션 3: 월별 성능 테이블 ──────────────────────────────────────────
st.subheader("📅 월별 성능 테이블")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric(
    "평균 Precision 차이",
    f"{avg_diff:+.4f}" if not np.isnan(avg_diff) else "N/A",
    delta="모델 우위" if (not np.isnan(avg_diff) and avg_diff > 0) else "개선 필요",
    delta_color="normal" if (not np.isnan(avg_diff) and avg_diff > 0) else "inverse"
)
mc2.metric("모델우위 월 수", f"{win_months} / {len(scored_df)}개월",
           help="신호없음 달 제외 기준")
mc3.metric("모델우위 비율",
           f"{win_rate_val:.1%}" if not np.isnan(win_rate_val) else "N/A",
           help="신호없음 달 제외 기준")
mc4.metric("신호없음 달 수", f"{no_sig_cnt} / {len(monthly_df)}개월",
           help="해당 월 UP 예측 0건 → 성능 평가 불가")

def highlight_row(row):
    if row['판정'] == '✅ 모델우위':
        color = '#d4edda'
    elif row['판정'] == '❌ 시장편향':
        color = '#f8d7da'
    else:
        color = '#f0f0f0'   # 신호없음 → 회색
    return [f'background-color: {color}'] * len(row)

def fmt_optional(val, fmt):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '-'
    return fmt.format(val)

styled = monthly_df.style.apply(highlight_row, axis=1).format({
    'UP비율'   : lambda v: fmt_optional(v, '{:.3f}'),
    'Precision': lambda v: fmt_optional(v, '{:.3f}'),
    'Recall'   : lambda v: fmt_optional(v, '{:.3f}'),
    'F1'       : lambda v: fmt_optional(v, '{:.3f}'),
    '차이'     : lambda v: fmt_optional(v, '{:+.3f}'),
})
st.dataframe(styled, use_container_width=True, hide_index=True)
st.divider()


# ── 섹션 5: 결과 CSV 다운로드 ─────────────────────────────────────────
st.subheader("💾 결과 저장")
result_out = result.copy()
result_out['real_label'] = encoder.inverse_transform(result_out['real'])
result_out['pred_label'] = encoder.inverse_transform(result_out['pred'])
result_out['correct']    = (result_out['real'] == result_out['pred']).astype(int)
csv_bytes = result_out.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

today_prob = result_out.iloc[-1]['prob_up']

# ── 섹션 1: 내일 예측 ─────────────────────────────────────────────────
st.subheader("🔮 매매 신호")
sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)

signal_color = "normal" if tomorrow_pred == 1 else "inverse"
sig_col1.metric("매매 신호", invest_guide)
sig_col2.metric("예측 방향", tomorrow_label)
sig_col3.metric("UP 확률", f"{today_prob:.1%}")
sig_col4.metric("적용 Threshold", f"{THRESHOLD:.2f}")

st.caption(f"기준일: {date_series.iloc[-1].date()}  |  사용 모델: {[n for n, _ in estimators]}")

st.divider()

