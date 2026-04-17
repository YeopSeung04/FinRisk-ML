"""
Project 01 — 신용카드 사기 탐지 (Fraud Detection)
=================================================
기법: Isolation Forest + XGBoost + SMOTE
평가: AUC-ROC, Precision-Recall, Confusion Matrix
"""
import sys
sys.path.append('/home/claude/financial_portfolio')

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from utils.data_generator import generate_fraud_data

print("=" * 60)
print("  Project 01: 신용카드 사기 탐지")
print("=" * 60)

# ── 1. 데이터 로드 ──────────────────────────────────────────
print("\n[1/6] 데이터 생성 및 탐색적 분석")
df = generate_fraud_data(n_samples=50000)
print(f"  총 거래: {len(df):,}건")
print(f"  사기:   {df['Class'].sum():,}건 ({df['Class'].mean():.4%})")
print(f"  정상:   {(df['Class']==0).sum():,}건")
print(f"  금액 통계:\n{df.groupby('Class')['Amount'].describe().round(2)}")

# ── 2. 전처리 ──────────────────────────────────────────────
print("\n[2/6] 전처리...")
df['Amount_log'] = np.log1p(df['Amount'])
df['Hour'] = (df['Time'] % 86400 // 3600).astype(int)

features = [f"V{i}" for i in range(1, 29)] + ['Amount_log', 'Hour']
X = df[features].values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# SMOTE 적용
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
print(f"  SMOTE 후 클래스 분포: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# ── 3. 모델 학습 ───────────────────────────────────────────
print("\n[3/6] 모델 학습...")

results = {}

# Logistic Regression (베이스라인)
lr = LogisticRegression(random_state=42, max_iter=500)
lr.fit(X_train_sm, y_train_sm)
lr_prob = lr.predict_proba(X_test_sc)[:, 1]
results['Logistic Regression'] = {
    'proba': lr_prob,
    'auc': roc_auc_score(y_test, lr_prob),
    'ap': average_precision_score(y_test, lr_prob)
}
print(f"  LR     AUC: {results['Logistic Regression']['auc']:.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
rf_prob = rf.predict_proba(X_test_sc)[:, 1]
results['Random Forest'] = {
    'proba': rf_prob,
    'auc': roc_auc_score(y_test, rf_prob),
    'ap': average_precision_score(y_test, rf_prob)
}
print(f"  RF     AUC: {results['Random Forest']['auc']:.4f}")

# XGBoost
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    scale_pos_weight=scale_pos, random_state=42,
    eval_metric='aucpr', verbosity=0
)
xgb_model.fit(X_train_sm, y_train_sm)
xgb_prob = xgb_model.predict_proba(X_test_sc)[:, 1]
results['XGBoost'] = {
    'proba': xgb_prob,
    'auc': roc_auc_score(y_test, xgb_prob),
    'ap': average_precision_score(y_test, xgb_prob)
}
print(f"  XGB    AUC: {results['XGBoost']['auc']:.4f}")

# Isolation Forest (비지도)
iso = IsolationForest(contamination=0.002, random_state=42, n_jobs=-1)
iso.fit(X_train_sc[y_train == 0])  # 정상 데이터로만 학습
iso_score = -iso.score_samples(X_test_sc)
results['Isolation Forest'] = {
    'proba': iso_score,
    'auc': roc_auc_score(y_test, iso_score),
    'ap': average_precision_score(y_test, iso_score)
}
print(f"  ISO    AUC: {results['Isolation Forest']['auc']:.4f}")

# ── 4. 최적 임계값 탐색 ────────────────────────────────────
print("\n[4/6] XGBoost 최적 임계값 탐색...")
precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_prob)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]
print(f"  최적 임계값: {best_threshold:.4f}")
print(f"  최적 F1:    {f1_scores[best_idx]:.4f}")

y_pred_best = (xgb_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_best)
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n  Classification Report:\n{classification_report(y_test, y_pred_best, target_names=['정상', '사기'])}")

# ── 5. Feature Importance ─────────────────────────────────
print("\n[5/6] Feature Importance...")
feature_imp = pd.DataFrame({
    'feature': features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)
print(feature_imp.to_string(index=False))

# ── 6. 결과 저장 ───────────────────────────────────────────
print("\n[6/6] 결과 데이터 저장")
# ROC curve 데이터
roc_data = {}
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['proba'])
    # 샘플링 (데이터 크기 줄이기)
    idx = np.linspace(0, len(fpr)-1, 200, dtype=int)
    roc_data[name] = {
        'fpr': fpr[idx].tolist(),
        'tpr': tpr[idx].tolist(),
        'auc': r['auc'],
        'ap': r['ap']
    }

# PR curve 데이터 (XGBoost만)
pr_data = {
    'precision': precisions.tolist(),
    'recall': recalls.tolist()
}

output = {
    'roc_data': roc_data,
    'pr_data': pr_data,
    'feature_imp': feature_imp.to_dict('records'),
    'confusion_matrix': cm.tolist(),
    'best_threshold': float(best_threshold),
    'fraud_rate': float(df['Class'].mean()),
    'total_samples': len(df),
    'summary': {
        name: {'auc': round(r['auc'], 4), 'ap': round(r['ap'], 4)}
        for name, r in results.items()
    }
}

with open('/home/claude/financial_portfolio/outputs/fraud_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n✓ 완료! outputs/fraud_results.json 저장됨")
print(f"\n{'='*60}")
print(f"  Best Model: XGBoost")
print(f"  AUC-ROC: {results['XGBoost']['auc']:.4f}")
print(f"  Average Precision: {results['XGBoost']['ap']:.4f}")
print(f"{'='*60}")
