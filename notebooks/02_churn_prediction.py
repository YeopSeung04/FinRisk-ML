"""
Project 02 — 고객 이탈 예측 (Churn Prediction)
===============================================
기법: LightGBM + SHAP + Survival Analysis
평가: AUC-ROC, F1, Cohort Analysis
"""
import sys
sys.path.append('/home/claude/financial_portfolio')

import numpy as np
import pandas as pd
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report,
    roc_curve, f1_score, precision_score, recall_score
)
import shap
import warnings
warnings.filterwarnings('ignore')

from utils.data_generator import generate_churn_data

print("=" * 60)
print("  Project 02: 고객 이탈 예측 (Churn Prediction)")
print("=" * 60)

# ── 1. 데이터 ─────────────────────────────────────────────
print("\n[1/5] 데이터 생성...")
df = generate_churn_data(n_samples=7000)
print(f"  총 고객: {len(df):,}명")
print(f"  이탈율: {df['Churn'].mean():.2%}")
print(f"\n  카드 유형별 이탈율:")
print(df.groupby('CardType')['Churn'].agg(['mean', 'count']).rename(
    columns={'mean': '이탈율', 'count': '고객수'}
).to_string())

# ── 2. 전처리 ──────────────────────────────────────────────
print("\n[2/5] 전처리 & Feature Engineering...")

# 인코딩
le_gender = LabelEncoder()
le_card = LabelEncoder()
df['Gender_enc'] = le_gender.fit_transform(df['Gender'])
df['CardType_enc'] = le_card.fit_transform(df['CardType'])

# 파생 변수
df['SpendPerMonth'] = df['MonthlySpend'] / (df['Tenure'] + 1)
df['LimitUtilAmount'] = df['CreditLimit'] * df['UtilizationRate']
df['CSperYear'] = df['CSContacts'] / (df['Tenure'] / 12 + 0.1)

features = [
    'Age', 'Tenure', 'CreditLimit', 'MonthlySpend', 'UtilizationRate',
    'NumProducts', 'CSContacts', 'OnlineBanking', 'AutoPay',
    'Gender_enc', 'CardType_enc', 'SpendPerMonth', 'LimitUtilAmount', 'CSperYear'
]
X = df[features]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── 3. LightGBM 학습 ──────────────────────────────────────
print("\n[3/5] LightGBM 학습 (5-Fold CV)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        class_weight='balanced', random_state=42,
        verbose=-1
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])

    val_prob = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_prob)
    oof_scores.append(score)
    print(f"  Fold {fold+1}: AUC = {score:.4f}")

print(f"  CV Mean AUC: {np.mean(oof_scores):.4f} ± {np.std(oof_scores):.4f}")

# 최종 모델
final_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=20,
    class_weight='balanced', random_state=42, verbose=-1
)
final_model.fit(X_train, y_train)
test_prob = final_model.predict_proba(X_test)[:, 1]

# 임계값 0.5
y_pred = (test_prob >= 0.5).astype(int)
test_auc = roc_auc_score(y_test, test_prob)
print(f"\n  Test AUC: {test_auc:.4f}")
print(f"  Test F1:  {f1_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['유지', '이탈'])}")

# ── 4. SHAP 분석 ──────────────────────────────────────────
print("\n[4/5] SHAP 분석...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# SHAP summary (feature importance)
shap_importance = pd.DataFrame({
    'feature': features,
    'shap_mean_abs': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_mean_abs', ascending=False)
print(shap_importance.to_string(index=False))

# ── 5. 코호트 분석 (Tenure별 이탈율) ──────────────────────
print("\n[5/5] 코호트 분석...")
df['TenureBand'] = pd.cut(df['Tenure'],
    bins=[0, 12, 24, 36, 48, 72],
    labels=['0-12개월', '12-24개월', '24-36개월', '36-48개월', '48개월+']
)
cohort = df.groupby('TenureBand', observed=True)['Churn'].agg(
    ['mean', 'count']
).rename(columns={'mean': '이탈율', 'count': '고객수'}).reset_index()
print(cohort.to_string(index=False))

# 결과 저장
fpr, tpr, _ = roc_curve(y_test, test_prob)
idx = np.linspace(0, len(fpr)-1, 200, dtype=int)

output = {
    'cv_scores': [round(s, 4) for s in oof_scores],
    'cv_mean': round(float(np.mean(oof_scores)), 4),
    'test_auc': round(test_auc, 4),
    'test_f1': round(float(f1_score(y_test, y_pred)), 4),
    'test_precision': round(float(precision_score(y_test, y_pred)), 4),
    'test_recall': round(float(recall_score(y_test, y_pred)), 4),
    'roc': {
        'fpr': fpr[idx].tolist(),
        'tpr': tpr[idx].tolist()
    },
    'shap_importance': shap_importance.to_dict('records'),
    'cohort': cohort.to_dict('records'),
    'churn_rate': float(df['Churn'].mean()),
    'churn_by_card': df.groupby('CardType')['Churn'].mean().round(4).to_dict(),
    'churn_by_tenure': cohort.to_dict('records'),
}

# SHAP waterfall 데이터 (상위 5개 고객 샘플)
sample_shap = []
for i in range(min(5, len(X_test))):
    sample_shap.append({
        'customer_idx': i,
        'shap_values': shap_values[i].tolist(),
        'features': features,
        'feature_values': X_test.iloc[i].tolist(),
        'prediction': float(test_prob[i])
    })
output['sample_shap'] = sample_shap

with open('/home/claude/financial_portfolio/outputs/churn_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n✓ 완료! outputs/churn_results.json 저장됨")
print(f"\n{'='*60}")
print(f"  Best Model: LightGBM (5-Fold CV)")
print(f"  CV AUC: {np.mean(oof_scores):.4f} ± {np.std(oof_scores):.4f}")
print(f"  Test AUC: {test_auc:.4f}  F1: {f1_score(y_test, y_pred):.4f}")
print(f"{'='*60}")
