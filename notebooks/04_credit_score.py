"""
Project 04 — 신용 점수 예측 모델 (Credit Score Model)
=====================================================
기법: RandomForest + Logistic Regression + SHAP
분석: WOE/IV, Calibration, Feature Importance
"""
import sys
sys.path.append('/home/claude/financial_portfolio')

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    roc_curve, classification_report, confusion_matrix
)
import shap
import warnings
warnings.filterwarnings('ignore')

from utils.data_generator import generate_credit_data

print("=" * 60)
print("  Project 04: 신용 점수 예측 모델")
print("=" * 60)

# ── 1. 데이터 ─────────────────────────────────────────────
print("\n[1/6] 데이터 생성 & EDA...")
df = generate_credit_data(n_samples=30000)
print(f"  총 고객: {len(df):,}명")
print(f"  부도율: {df['default'].mean():.2%}")

print("\n  교육 수준별 부도율:")
edu_map = {1: '대학원', 2: '대학교', 3: '고등학교', 4: '기타'}
df['Education_label'] = df['EDUCATION'].map(edu_map)
print(df.groupby('Education_label')['default'].agg(['mean', 'count']).rename(
    columns={'mean': '부도율', 'count': '고객수'}
).to_string())

print("\n  신용한도 구간별 부도율:")
df['LimitBand'] = pd.cut(df['LIMIT_BAL'],
    bins=[0, 100000, 300000, 500000, 800000],
    labels=['10만이하', '30만이하', '50만이하', '80만이하']
)
print(df.groupby('LimitBand', observed=True)['default'].agg(['mean', 'count']).to_string())

# ── 2. Feature Engineering ────────────────────────────────
print("\n[2/6] Feature Engineering...")

# 연체 관련 파생 변수
pay_cols = [f'PAY_{i}' for i in range(1, 7)]
bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]

df['avg_delay'] = df[pay_cols].clip(lower=0).mean(axis=1)
df['max_delay'] = df[pay_cols].clip(lower=0).max(axis=1)
df['n_delayed'] = (df[pay_cols] > 0).sum(axis=1)
df['avg_bill'] = df[bill_cols].mean(axis=1)
df['avg_payment'] = df[pay_amt_cols].mean(axis=1)
df['pay_ratio'] = df['avg_payment'] / (df['avg_bill'] + 1)
df['util_rate'] = df['avg_bill'] / (df['LIMIT_BAL'] + 1)
df['payment_trend'] = df['PAY_AMT1'] - df['PAY_AMT6']  # 최근 - 6개월전

features = (
    ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
    + pay_cols + bill_cols + pay_amt_cols
    + ['avg_delay', 'max_delay', 'n_delayed', 'avg_bill',
       'avg_payment', 'pay_ratio', 'util_rate', 'payment_trend']
)

X = df[features]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"  Feature 수: {len(features)}")
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── 3. 모델 비교 ──────────────────────────────────────────
print("\n[3/6] 모델 학습 & 비교...")

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=500, C=0.1, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42,
        class_weight='balanced', n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
}

results = {}
for name, m in models.items():
    if name == 'Logistic Regression':
        m.fit(X_train_sc, y_train)
        prob = m.predict_proba(X_test_sc)[:, 1]
    else:
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(m, X_train if name != 'Logistic Regression' else X_train_sc,
                                 y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    results[name] = {
        'prob': prob,
        'auc': roc_auc_score(y_test, prob),
        'brier': brier_score_loss(y_test, prob),
        'logloss': log_loss(y_test, prob),
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores))
    }
    print(f"  {name:<22} AUC={results[name]['auc']:.4f}  "
          f"CV={results[name]['cv_mean']:.4f}±{results[name]['cv_std']:.4f}")

# ── 4. Calibration ─────────────────────────────────────────
print("\n[4/6] 모델 Calibration (Platt Scaling)...")
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = models[best_model_name]

calibrated = CalibratedClassifierCV(best_model, cv='prefit', method='sigmoid')
calibrated.fit(
    X_test if best_model_name != 'Logistic Regression' else X_test_sc,
    y_test
)
cal_prob = calibrated.predict_proba(
    X_test if best_model_name != 'Logistic Regression' else X_test_sc
)[:, 1]
print(f"  최고 모델: {best_model_name}")
print(f"  Calibrated AUC:   {roc_auc_score(y_test, cal_prob):.4f}")
print(f"  Calibrated Brier: {brier_score_loss(y_test, cal_prob):.4f}")

# Calibration curve 데이터
frac_pos_orig, mean_pred_orig = calibration_curve(y_test, results[best_model_name]['prob'], n_bins=10)
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, cal_prob, n_bins=10)

# ── 5. SHAP + WOE/IV ──────────────────────────────────────
print("\n[5/6] SHAP 분석...")
rf_model = models['Random Forest']
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test.iloc[:1000])

if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

shap_imp = pd.DataFrame({
    'feature': features,
    'shap_mean_abs': np.abs(sv).mean(axis=0)
}).sort_values('shap_mean_abs', ascending=False).head(20)
print(shap_imp.head(10).to_string(index=False))

# WOE 분석 (LIMIT_BAL)
print("\n[6/6] WOE/IV 분석 (신용한도)...")
df['LimitBand_woe'] = pd.qcut(df['LIMIT_BAL'], q=10, duplicates='drop')
woe_table = df.groupby('LimitBand_woe', observed=True)['default'].agg(
    Events=lambda x: x.sum(),
    NonEvents=lambda x: (x == 0).sum(),
    Total='count'
).reset_index()

total_events = woe_table['Events'].sum()
total_non_events = woe_table['NonEvents'].sum()
woe_table['pct_events'] = woe_table['Events'] / total_events
woe_table['pct_non_events'] = woe_table['NonEvents'] / total_non_events
woe_table['WOE'] = np.log(woe_table['pct_non_events'] / (woe_table['pct_events'] + 1e-8))
woe_table['IV'] = (woe_table['pct_non_events'] - woe_table['pct_events']) * woe_table['WOE']
print(woe_table[['LimitBand_woe', 'WOE', 'IV']].to_string(index=False))
print(f"  LIMIT_BAL IV: {woe_table['IV'].sum():.4f}")

# 신용 점수 스케일 변환 (300~850)
best_prob = results[best_model_name]['prob']
credit_scores = 850 - (best_prob * 550)
score_df = pd.DataFrame({'prob': best_prob, 'credit_score': credit_scores, 'actual': y_test.values})

print(f"\n  신용 점수 분포:")
print(score_df['credit_score'].describe().round(1).to_string())

# 결과 저장
roc_data = {}
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['prob'])
    idx = np.linspace(0, len(fpr)-1, 200, dtype=int)
    roc_data[name] = {'fpr': fpr[idx].tolist(), 'tpr': tpr[idx].tolist(), 'auc': r['auc']}

score_hist, score_bins = np.histogram(credit_scores, bins=20)
output = {
    'model_comparison': {
        name: {
            'auc': round(r['auc'], 4),
            'brier': round(r['brier'], 4),
            'logloss': round(r['logloss'], 4),
            'cv_mean': round(r['cv_mean'], 4),
        }
        for name, r in results.items()
    },
    'roc_data': roc_data,
    'calibration': {
        'original': {
            'frac_pos': frac_pos_orig.tolist(),
            'mean_pred': mean_pred_orig.tolist()
        },
        'calibrated': {
            'frac_pos': frac_pos_cal.tolist(),
            'mean_pred': mean_pred_cal.tolist()
        }
    },
    'shap_importance': shap_imp.to_dict('records'),
    'woe_table': woe_table[['WOE', 'IV', 'Events', 'NonEvents']].to_dict('records'),
    'credit_score_hist': {
        'counts': score_hist.tolist(),
        'bins': score_bins.tolist()
    },
    'default_rate': float(df['default'].mean()),
    'default_by_education': df.groupby('EDUCATION')['default'].mean().round(4).to_dict(),
    'best_model': best_model_name,
    'best_auc': round(results[best_model_name]['auc'], 4),
}

with open('/home/claude/financial_portfolio/outputs/credit_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n✓ 완료! outputs/credit_results.json 저장됨")
print(f"\n{'='*60}")
print(f"  Best Model: {best_model_name}")
print(f"  Test AUC: {results[best_model_name]['auc']:.4f}")
print(f"{'='*60}")
