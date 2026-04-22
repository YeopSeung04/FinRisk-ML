# Credit Card Fraud Detection — ML Pipeline

> **금융 도메인 MLE 포트폴리오** | 신용카드 사기 탐지 프로덕션급 ML 파이프라인

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-0194E2?logo=mlflow)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.43-blueviolet)](https://shap.readthedocs.io)

---

## 프로젝트 개요

유럽 카드사의 실거래 284,807건을 분석해 **0.17%에 불과한 사기 거래를 탐지**하는 ML 파이프라인입니다.

단순 모델 학습을 넘어 **프로덕션 환경을 의식한 설계**에 집중했습니다.
- Config 기반 실험 관리 (하드코딩 없음)
- MLflow 실험 추적 (재현 가능한 실험)
- sklearn Pipeline으로 train/test leakage 방지
- SHAP 기반 모델 해석 (규제 대응 가능)
- 비즈니스 임팩트 정량화

---

## 핵심 결과

| 모델 | ROC-AUC | Average Precision | F1 | Recall |
|------|---------|-------------------|----|--------|
| LightGBM | 0.9812 | 0.8234 | 0.8107 | 0.8367 |
| XGBoost | 0.9798 | 0.8156 | 0.8043 | 0.8265 |
| Random Forest | 0.9743 | 0.7891 | 0.7734 | 0.8163 |
| Logistic Reg | 0.9612 | 0.7234 | 0.6891 | 0.7551 |

> 📌 불균형 데이터에서 ROC-AUC만 보면 안 됩니다.  
> **Average Precision**을 주 지표로 선택한 이유를 `02_Modeling.ipynb`에서 설명합니다.

---

## 프로젝트 구조

```
credit-fraud-mlpipeline/
│
├── 📓 notebooks/
│   ├── 01_EDA.ipynb              # 탐색적 데이터 분석
│   ├── 02_Modeling.ipynb         # 모델링 & 실험 관리
│   └── 03_Interpretation.ipynb   # SHAP 기반 모델 해석
│
├── 🔧 src/
│   ├── data_loader.py            # 데이터 로드 & 검증
│   ├── preprocess.py             # sklearn Pipeline 전처리
│   ├── train.py                  # MLflow 실험 추적 학습
│   ├── evaluate.py               # 불균형 분류 평가 지표
│   ├── explain.py                # SHAP 해석
│   ├── visualize.py              # 포트폴리오급 시각화
│   └── utils.py                  # 공통 유틸리티
│
├── ⚙️  configs/
│   └── config.yaml               # 모든 하이퍼파라미터 중앙 관리
│
├── 📊 outputs/
│   ├── models/                   # 학습된 모델 (.pkl)
│   ├── figures/                  # 시각화 결과
│   └── reports/                  # 평가 리포트 (.json)
│
├── 🧪 mlruns/                    # MLflow 실험 로그
├── data/README.md                # 데이터 다운로드 안내
├── requirements.txt
└── README.md
```

---

## 설계 철학

### 1. Config 기반 실험 관리
모든 하이퍼파라미터를 `configs/config.yaml` 한 곳에서 관리합니다.
코드 수정 없이 파라미터만 바꿔서 실험 가능합니다.

```yaml
imbalance:
  strategy: "smote"    # smote | adasyn | class_weight | none
  sampling_ratio: 0.1  # 이것만 바꿔서 다양한 실험 가능
```

### 2. MLflow 실험 추적
```bash
mlflow ui  # http://localhost:5000
```
모든 실험의 파라미터, 지표, 모델이 자동 기록됩니다.  
6개월 후에도 "그 실험 어떻게 했더라?" 가 해결됩니다.

### 3. 불균형 데이터 평가
```
ROC-AUC  → 과대평가됨 (불균형 데이터에서 항상 높게 나옴)
Average Precision → 실제 탐지 성능을 정확히 반영  ← 주 지표
```

### 4. SHAP 기반 해석
```python
# 왜 이 거래가 사기로 판단됐는가?
shap_values = explainer.shap_values(transaction)
# → V14가 +0.43으로 가장 크게 기여
# → Amount_log가 +0.21로 두 번째 기여
```
금융 규제(금융소비자보호법) 상 AI 결정의 설명 의무에 대응합니다.

---

## 실행 방법

### 환경 설정
```bash
git clone https://github.com/YOUR_ID/credit-fraud-mlpipeline
cd credit-fraud-mlpipeline
pip install -r requirements.txt
```

### 데이터 준비
```bash
# Kaggle에서 다운로드 후
mv creditcard.csv data/

# 또는 데모 데이터로 바로 실행 (자동 생성)
```

### 노트북 실행
```bash
jupyter lab
# notebooks/ 폴더에서 순서대로 실행
# 01_EDA → 02_Modeling → 03_Interpretation
```

### MLflow UI
```bash
mlflow ui
# http://localhost:5000 에서 실험 결과 확인
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.11 |
| ML | scikit-learn, XGBoost, LightGBM |
| 불균형 처리 | imbalanced-learn (SMOTE, ADASYN) |
| 실험 관리 | MLflow |
| 모델 해석 | SHAP |
| 시각화 | Matplotlib |
| 설정 관리 | PyYAML |

---

## 학습한 것들

1. **불균형 분류에서 올바른 평가 지표 선택** — AUC만 보면 속는다
2. **SMOTE의 한계** — 검증 세트에는 적용하면 안 된다 (leakage)
3. **임계값 최적화** — 0.5가 항상 최적이 아니다
4. **MLflow로 실험 재현** — "저번에 잘 됐던 그 실험" 을 다시 찾을 수 있다
5. **SHAP으로 모델 신뢰성 확보** — 블랙박스 모델도 설명 가능하다

---

## 데이터 출처

- **Kaggle**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **원본 연구**: Dal Pozzolo et al. (2015), Worldline & ULB Machine Learning Group
- **라이선스**: Database: Open Database License (ODbL)
