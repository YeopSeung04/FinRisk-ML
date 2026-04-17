# 금융 데이터 분석 포트폴리오

> 카드사·금융 도메인 핵심 4개 분석 프로젝트 — Python · scikit-learn · LightGBM · SHAP

---

## 프로젝트 구성

| # | 프로젝트 | 기법 | 핵심 지표 |
|---|---------|------|---------|
| 01 | 신용카드 사기 탐지 | XGBoost · SMOTE · Isolation Forest | AUC-ROC **0.9997** |
| 02 | 고객 이탈 예측 | LightGBM · SHAP · 5-Fold CV | CV AUC **0.6991** |
| 03 | 소비 패턴 세그멘테이션 | K-Means · RFM · PCA | Silhouette **0.6118** |
| 04 | 신용 점수 예측 | RandomForest · SHAP · WOE/IV | AUC **0.6541** |

---

## 폴더 구조

```
financial_portfolio/
├── utils/
│   └── data_generator.py      # 합성 데이터 생성 (Fraud/Churn/Segment/Credit)
├── notebooks/
│   ├── 01_fraud_detection.py  # 사기 탐지 분석
│   ├── 02_churn_prediction.py # 이탈 예측 분석
│   ├── 03_segmentation.py     # 세그멘테이션 분석
│   └── 04_credit_score.py     # 신용 점수 분석
├── outputs/
│   ├── dashboard.html         # 인터랙티브 대시보드
│   ├── fraud_results.json
│   ├── churn_results.json
│   ├── segment_results.json
│   └── credit_results.json
└── README.md
```

---

## Project 01 — 신용카드 사기 탐지

**문제 정의:** 거래 데이터에서 0.17%에 불과한 사기 거래를 탐지하는 극단적 불균형 분류 문제

**데이터:** 50,000건 거래 (사기 85건 / 정상 49,915건)

**핵심 접근:**
- SMOTE로 소수 클래스 오버샘플링 (1:1 균형 맞춤)
- Isolation Forest로 비지도 이상 탐지 병행
- XGBoost + `scale_pos_weight`로 불균형 가중치 적용
- Precision-Recall 곡선 기반 최적 임계값 탐색

**결과:**
```
Model             AUC-ROC   AP
Logistic Reg      1.0000    0.9967
Random Forest     0.9997    0.9441
XGBoost           0.9995    0.8993
Isolation Forest  0.9535    0.0273
```

---

## Project 02 — 고객 이탈 예측

**문제 정의:** 체류 기간, 이용 패턴, CS 접촉 이력으로 향후 이탈 고객 사전 예측

**데이터:** 7,000명 카드사 고객 (이탈율 4.99%)

**핵심 접근:**
- LightGBM + Stratified 5-Fold CV
- SHAP TreeExplainer로 이탈 원인 개인별 해석
- 코호트 분석: 신규 12개월 이내 이탈율 15.6% (위험 구간 식별)
- Feature Engineering: SpendPerMonth, CSperYear, LimitUtilAmount

**인사이트:**
- 가입 0-12개월 신규 고객 이탈율이 **15.6%**로 가장 높음
- Tenure (체류 기간)이 이탈 예측 가장 중요 변수 (SHAP)
- AutoPay 미등록 고객이 이탈 위험 1.4배 높음

---

## Project 03 — 소비 패턴 세그멘테이션

**문제 정의:** 거래 이력 기반 RFM 분석으로 마케팅 타겟 세그먼트 발굴

**데이터:** 2,000 고객 · 21,146 거래 (2024년 1년 치)

**핵심 접근:**
- RFM (Recency · Frequency · Monetary) 지표 계산
- Elbow + Silhouette Score로 최적 K=2 선정
- PCA 2D 투영으로 클러스터 시각화
- 카테고리별 지출 패턴 분석 (식비 1위, 쇼핑 2위)

**세그먼트 프로파일:**

| 세그먼트 | 고객 수 | 평균 빈도 | 평균 금액 |
|--------|--------|---------|---------|
| VIP 고객 | 204명 | 25회 | 3,677만원 |
| 활성 고객 | 1,796명 | 9회 | 322만원 |

---

## Project 04 — 신용 점수 예측

**문제 정의:** 대출·카드 신청자의 향후 부도 가능성 예측 및 신용 점수 산출

**데이터:** 10,000명 (부도율 5.4%)

**핵심 접근:**
- 6개월 납부 이력 Feature Engineering (avg_delay, n_delayed, pay_ratio)
- RandomForest + SHAP 해석
- WOE/IV 분석으로 신용한도의 변별력 측정
- 예측 확률 → 신용 점수 (300-850) 스케일 변환

---

## 기술 스택

```python
Python 3.11+
pandas · numpy                  # 데이터 처리
scikit-learn                    # ML 파이프라인
lightgbm · xgboost              # 트리 부스팅
imbalanced-learn                # SMOTE 불균형 처리
shap                            # 모델 해석
faker                           # 합성 데이터 생성
plotly / Chart.js               # 시각화
```

## 실행 방법

```bash
# 환경 설치
pip install pandas numpy scikit-learn lightgbm xgboost plotly imbalanced-learn shap faker

# 프로젝트 루트에서 순서대로 실행
python notebooks/01_fraud_detection.py
python notebooks/02_churn_prediction.py
python notebooks/03_segmentation.py
python notebooks/04_credit_score.py

# 대시보드 열기
open outputs/dashboard.html
```
