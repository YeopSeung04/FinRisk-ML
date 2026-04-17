"""
Project 03 — 소비 패턴 분석 & 세그멘테이션
==========================================
기법: RFM 분석 + K-Means + PCA
평가: Silhouette Score, 세그먼트별 프로파일링
"""
import sys
sys.path.append('/home/claude/financial_portfolio')

import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from utils.data_generator import generate_transaction_data

print("=" * 60)
print("  Project 03: 소비 패턴 분석 & 세그멘테이션")
print("=" * 60)

# ── 1. 데이터 ─────────────────────────────────────────────
print("\n[1/5] 데이터 생성...")
tx_df, cust_df = generate_transaction_data(n_customers=2000, n_transactions=50000)
print(f"  거래 건수: {len(tx_df):,}건")
print(f"  고객 수: {tx_df['CustomerID'].nunique():,}명")
print(f"  기간: {tx_df['Date'].min().date()} ~ {tx_df['Date'].max().date()}")
print(f"\n  카테고리별 지출:")
cat_summary = tx_df.groupby('Category')['Amount'].agg(['sum', 'count', 'mean']).round(0)
print(cat_summary.to_string())

# ── 2. RFM 계산 ────────────────────────────────────────────
print("\n[2/5] RFM 지표 계산...")
reference_date = pd.Timestamp('2025-01-01')

rfm = tx_df.groupby('CustomerID').agg(
    Recency=('Date', lambda x: (reference_date - x.max()).days),
    Frequency=('CustomerID', 'count'),
    Monetary=('Amount', 'sum')
).reset_index()

# 카테고리별 지출 비중 (pivot)
cat_pivot = tx_df.pivot_table(
    index='CustomerID', columns='Category', values='Amount',
    aggfunc='sum', fill_value=0
)
cat_pivot_pct = cat_pivot.div(cat_pivot.sum(axis=1), axis=0)

rfm = rfm.merge(cat_pivot_pct.add_prefix('pct_'), on='CustomerID')

print(f"  RFM 기술통계:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2).to_string())

# RFM 점수 (1-5 등급)
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['RFM_score'] = rfm['R_score'].astype(int) + rfm['F_score'].astype(int) + rfm['M_score'].astype(int)

# ── 3. K-Means 클러스터링 ─────────────────────────────────
print("\n[3/5] K-Means 클러스터링...")
cluster_features = ['Recency', 'Frequency', 'Monetary']
X_cluster = rfm[cluster_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 최적 k 탐색 (Elbow + Silhouette)
k_range = range(2, 9)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

best_k = k_range.start + np.argmax(sil_scores)
print(f"  최적 클러스터 수 (Silhouette): k={best_k}")
print(f"  Silhouette Scores: {[round(s, 4) for s in sil_scores]}")

# 최종 모델
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['Cluster'] = km_final.fit_predict(X_scaled)

# ── 4. PCA 시각화 데이터 ──────────────────────────────────
print("\n[4/5] PCA 2D 투영...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
rfm['PCA1'] = X_pca[:, 0]
rfm['PCA2'] = X_pca[:, 1]
print(f"  설명 분산: {pca.explained_variance_ratio_ * 100}")

# ── 5. 세그먼트 프로파일링 ────────────────────────────────
print("\n[5/5] 세그먼트 프로파일링...")
profile = rfm.groupby('Cluster').agg(
    고객수=('CustomerID', 'count'),
    평균_최근성=('Recency', 'mean'),
    평균_빈도=('Frequency', 'mean'),
    평균_금액=('Monetary', 'mean'),
    RFM_평균=('RFM_score', 'mean'),
).round(2)

# 세그먼트 라벨 부여 (RFM 기준)
profile = profile.sort_values('RFM_평균', ascending=False)
segment_labels = ['VIP 고객', '활성 고객', '일반 고객', '휴면 고객',
                  '이탈 위험', '신규 고객', '잠재 고객', '저관여 고객']
profile['라벨'] = segment_labels[:len(profile)]
print(profile.to_string())

# 클러스터별 라벨 매핑
cluster_label_map = profile['라벨'].to_dict()
rfm['SegmentLabel'] = rfm['Cluster'].map(cluster_label_map)

# ── 결과 저장 ─────────────────────────────────────────────
# PCA scatter 데이터 (샘플링)
sample_idx = np.random.choice(len(rfm), min(1000, len(rfm)), replace=False)
scatter_data = rfm.iloc[sample_idx][['PCA1', 'PCA2', 'Cluster', 'SegmentLabel',
                                       'Recency', 'Frequency', 'Monetary']].round(4)

output = {
    'k_range': list(k_range),
    'inertias': [round(v, 2) for v in inertias],
    'sil_scores': [round(v, 4) for v in sil_scores],
    'best_k': int(best_k),
    'pca_variance': pca.explained_variance_ratio_.tolist(),
    'scatter': scatter_data.to_dict('records'),
    'segment_profile': profile.reset_index().to_dict('records'),
    'category_spend': tx_df.groupby('Category')['Amount'].sum().round(0).to_dict(),
    'rfm_stats': rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2).to_dict(),
    'cluster_label_map': {str(k): v for k, v in cluster_label_map.items()},
    'total_customers': int(rfm['CustomerID'].nunique()),
    'total_transactions': len(tx_df),
}

with open('/home/claude/financial_portfolio/outputs/segment_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n✓ 완료! outputs/segment_results.json 저장됨")
print(f"\n{'='*60}")
print(f"  클러스터 수: {best_k}")
print(f"  최적 Silhouette Score: {max(sil_scores):.4f}")
print(f"{'='*60}")
