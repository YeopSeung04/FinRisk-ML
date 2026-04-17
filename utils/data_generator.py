"""
금융 포트폴리오 - 공개 데이터셋 로더 & 합성 데이터 생성기
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from faker import Faker

fake = Faker('ko_KR')
np.random.seed(42)


# ─────────────────────────────────────────
# 1. 신용카드 사기 탐지 데이터 (Kaggle 유사 합성)
# ─────────────────────────────────────────
def generate_fraud_data(n_samples=50000):
    """
    실제 Kaggle Credit Card Fraud 데이터셋 구조를 모방한 합성 데이터.
    - V1~V28: PCA 변환 특징 (정규)
    - Amount: 거래 금액
    - Time: 경과 시간(초)
    - Class: 0=정상, 1=사기 (0.17% 불균형)
    """
    n_fraud = int(n_samples * 0.0017)
    n_normal = n_samples - n_fraud

    # 정상 거래
    normal = pd.DataFrame(
        np.random.randn(n_normal, 28),
        columns=[f"V{i}" for i in range(1, 29)]
    )
    normal['Amount'] = np.random.exponential(scale=80, size=n_normal).round(2)
    normal['Time'] = np.sort(np.random.uniform(0, 172800, n_normal))
    normal['Class'] = 0

    # 사기 거래 (다른 분포)
    fraud_features = np.random.randn(n_fraud, 28)
    fraud_features[:, [0, 1, 2, 3]] += np.array([-3.5, 3.0, -2.5, 2.8])  # 편향
    fraud = pd.DataFrame(fraud_features, columns=[f"V{i}" for i in range(1, 29)])
    fraud['Amount'] = np.random.exponential(scale=200, size=n_fraud).round(2)
    fraud['Time'] = np.random.uniform(0, 172800, n_fraud)
    fraud['Class'] = 1

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────
# 2. 고객 이탈 예측 데이터 (Telco Churn 유사)
# ─────────────────────────────────────────
def generate_churn_data(n_samples=7000):
    """
    카드사 고객 이탈 예측 데이터.
    특징: 이용 금액, 이용 빈도, 체류 기간, 상품 수, CS 접촉 횟수 등
    """
    rows = []
    for _ in range(n_samples):
        tenure = np.random.randint(1, 72)
        products = np.random.randint(1, 5)
        monthly_spend = max(0, np.random.normal(350000, 200000))
        cs_contacts = np.random.poisson(1.5)
        credit_limit = np.random.randint(1000000, 30000000)
        utilization = np.random.beta(2, 5)
        age = np.random.randint(22, 70)
        gender = np.random.choice(['M', 'F'])
        card_type = np.random.choice(['일반', '골드', '플래티넘'], p=[0.5, 0.35, 0.15])
        online_banking = np.random.choice([0, 1], p=[0.3, 0.7])
        auto_pay = np.random.choice([0, 1], p=[0.4, 0.6])

        # 이탈 확률 계산 (논리적 관계 반영)
        churn_prob = (
            0.05
            + (0.15 if tenure < 12 else 0)
            - (0.02 * products)
            + (0.03 * cs_contacts)
            - (0.01 * online_banking)
            - (0.02 * auto_pay)
            + (0.1 if utilization > 0.8 else 0)
            - (0.005 * min(monthly_spend / 100000, 5))
        )
        churn = int(np.random.random() < np.clip(churn_prob, 0.01, 0.95))

        rows.append({
            'CustomerID': fake.uuid4()[:8],
            'Age': age,
            'Gender': gender,
            'Tenure': tenure,
            'CardType': card_type,
            'CreditLimit': credit_limit,
            'MonthlySpend': round(monthly_spend),
            'UtilizationRate': round(utilization, 4),
            'NumProducts': products,
            'CSContacts': cs_contacts,
            'OnlineBanking': online_banking,
            'AutoPay': auto_pay,
            'Churn': churn
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 3. 소비 패턴 & 세그멘테이션 데이터 (RFM)
# ─────────────────────────────────────────
def generate_transaction_data(n_customers=2000, n_transactions=50000):
    """
    카드사 거래 데이터. RFM 분석 + 세그멘테이션에 사용.
    카테고리: 식비, 쇼핑, 교통, 의료, 문화, 여행, 기타
    """
    categories = ['식비', '쇼핑', '교통', '의료', '문화', '여행', '기타']
    cat_weights = [0.30, 0.25, 0.15, 0.08, 0.10, 0.05, 0.07]

    # 세그먼트별 고객 프로필
    segments = {
        'VIP': {'spend_mean': 1500000, 'freq_lambda': 25, 'n': int(n_customers * 0.10)},
        '활성': {'spend_mean': 500000,  'freq_lambda': 15, 'n': int(n_customers * 0.30)},
        '일반': {'spend_mean': 200000,  'freq_lambda': 8,  'n': int(n_customers * 0.40)},
        '휴면': {'spend_mean': 50000,   'freq_lambda': 2,  'n': int(n_customers * 0.20)},
    }

    customers = []
    transactions = []
    cid = 1

    for seg_name, props in segments.items():
        for _ in range(props['n']):
            customer_id = f"C{cid:05d}"
            cid += 1
            n_tx = max(1, np.random.poisson(props['freq_lambda']))

            for _ in range(n_tx):
                amount = max(1000, np.random.normal(props['spend_mean'], props['spend_mean'] * 0.5))
                days_ago = np.random.randint(1, 365)
                transactions.append({
                    'CustomerID': customer_id,
                    'Segment_True': seg_name,
                    'Amount': round(amount, -2),
                    'Category': np.random.choice(categories, p=cat_weights),
                    'DaysAgo': days_ago,
                    'Date': pd.Timestamp('2024-12-31') - pd.Timedelta(days=int(days_ago)),
                    'MerchantCity': np.random.choice(['서울', '부산', '대구', '인천', '광주'], p=[0.5, 0.2, 0.1, 0.1, 0.1]),
                })
            customers.append({'CustomerID': customer_id, 'Segment_True': seg_name})

    tx_df = pd.DataFrame(transactions)
    return tx_df, pd.DataFrame(customers)


# ─────────────────────────────────────────
# 4. 신용 점수 예측 데이터 (UCI 유사)
# ─────────────────────────────────────────
def generate_credit_data(n_samples=30000):
    """
    신용 점수 예측 (default 여부 분류).
    특징: 인구통계, 신용 한도, 청구/납부 이력, 연체 현황
    """
    rows = []
    for _ in range(n_samples):
        age = np.random.randint(21, 75)
        education = np.random.choice([1, 2, 3, 4], p=[0.15, 0.45, 0.30, 0.10])
        marriage = np.random.choice([0, 1, 2], p=[0.15, 0.55, 0.30])
        limit_bal = np.random.choice(
            range(10000, 800001, 10000),
            p=np.ones(80) / 80
        )
        sex = np.random.choice([1, 2])

        # 결제 상태 (-2: 잔액 없음, -1: 정상, 0~9: 연체 개월)
        pay_status = [
            np.random.choice([-2, -1, 0, 1, 2, 3], p=[0.15, 0.40, 0.25, 0.10, 0.07, 0.03])
            for _ in range(6)
        ]

        # 청구/납부 금액
        bill_amts = [max(0, np.random.normal(limit_bal * 0.4, limit_bal * 0.3)) for _ in range(6)]
        pay_amts = [max(0, b * np.random.uniform(0.05, 1.2)) for b in bill_amts]

        # 부도 확률
        avg_delay = np.mean([max(0, p) for p in pay_status])
        default_prob = (
            0.10
            + (0.25 * avg_delay / 3)
            - (0.001 * (age - 21))
            - (0.02 * (education - 1))
            - (0.00000005 * limit_bal)
        )
        default = int(np.random.random() < np.clip(default_prob, 0.01, 0.90))

        row = {
            'ID': _, 'LIMIT_BAL': limit_bal, 'SEX': sex,
            'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
        }
        for i, (ps, ba, pa) in enumerate(zip(pay_status, bill_amts, pay_amts), 1):
            row[f'PAY_{i}'] = ps
            row[f'BILL_AMT{i}'] = round(ba)
            row[f'PAY_AMT{i}'] = round(pa)
        row['default'] = default
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("데이터 생성 테스트...")
    df1 = generate_fraud_data(10000)
    print(f"Fraud:  {df1.shape}, 사기율: {df1['Class'].mean():.4%}")

    df2 = generate_churn_data(1000)
    print(f"Churn:  {df2.shape}, 이탈율: {df2['Churn'].mean():.4%}")

    df3, _ = generate_transaction_data(500, 5000)
    print(f"Trans:  {df3.shape}")

    df4 = generate_credit_data(5000)
    print(f"Credit: {df4.shape}, 부도율: {df4['default'].mean():.4%}")
    print("완료!")
