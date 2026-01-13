# Customer Segmentation & Causal Targeting - Dunnhumby

## Executive Summary

본 프로젝트는 Dunnhumby "The Complete Journey" 리테일 데이터셋을 활용하여 고객 분석을 위한 **2-Track Framework**를 구현하였다. 고객 세그멘테이션과 인과 추론을 결합하여 두 가지 핵심 마케팅 문제에 접근했다:

- **Track 1** (Factor modeling / Clustering): "우리 고객은 누구인가?" → 7개의 고유한 고객 세그먼트
- **Track 2** (Causal inference, HTE, optimal policy): "누구를 타겟팅해야 하는가?" → 125% ROI 개선을 위한 최적 31.3% 타겟팅

### Key Results

| Track | 핵심 발견 | 비즈니스 임팩트 |
|-------|----------|-----------------|
| Track 1 | 92.44% 분산 설명의 7개 세그먼트 | CRM을 위한 실행 가능한 고객 프로필 |
| Track 2 | 100% Baseline 대비 31.3% 최적 타겟팅 | $7,000+ 수익 개선 |
| Track 2 | VIP Heavy, Bulk Shoppers 음수 CATE | 해당 세그먼트 타겟팅 축소 |

---

## Motivation

본 프로젝트에서는 전통적인 세그멘테이션의 "우리 고객은 누구인가?"의 질문과 인과추론 기반의 "이 캠페인이 누구에게 얼마나 효과가 있을까?"의 문제에 대한 접근이다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         2-TRACK FRAMEWORK                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRACK 1: CUSTOMER UNDERSTANDING                                            │
│  ════════════════════════════════                                           │
│                                                                             │
│  "우리 고객은 누구인가?"                                                         │
│                                                                             │
│  • Latent Factor Modeling (NMF)   → 행동 차원 발견                             │
│  • Clustering (K-Means)           → 실행 가능한 세그먼트 도출                     │
│  • Stability Validation           → 세그먼트 신뢰성 보장                         │
│                                                                             │
│  Output: 마케팅 전략을 위한 고객 프로필                                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRACK 2: CAUSAL TARGETING                                                  │
│  ═════════════════════════                                                  │
│                                                                             │
│  "누구를 타겟팅해야 하는가?"                                                      │
│                                                                             │
│  • Heterogeneous Treatment Effects → 고객별 캠페인 효과                         │
│  • Policy Learning                 → 최적 타겟팅 규칙                           │
│  • ROI Optimization                → 캠페인 수익 극대화                         │
│                                                                             │
│  Output: 데이터 기반 타겟팅 정책                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**왜 두 Track 모두 필요한가?**

| 측면 | Track 1 (Descriptive) | Track 2 (Causal) |
|------|----------------------|------------------|
| 핵심 질문 | "이 고객은 누구인가?" | "이것이 그들에게 효과가 있을까?" |
| 주요 사용자 | 마케팅, CRM, 전략 | Data Science, 최적화 |
| 설명 가능성 | "Premium Fresh Lover 세그먼트" | "이 고객의 CATE = +$34" |
| 조직 요구사항 | 기본 분석 역량 | Causal thinking 문화 |

---

## Methodology

### Track 1: Customer Segmentation

**접근법**: Latent Factor Modeling + Clustering

1. **Feature Engineering**: 33개 고객 수준 Feature
   - RFM Feature (Recency, Frequency, Monetary)
   - Behavioral Feature (할인 사용, 다양성 추구)
   - Category 선호 (grocery, fresh, health & beauty)

2. **Latent Factor Modeling (NMF)**: 잠재 행동 차원 발견
   - 5개의 해석 가능한 Factor 추출
   - 92.44% 누적 분산 설명

3. **Clustering (K-Means)**: 고객 세그먼트 도출
   - Elbow method + Silhouette 분석으로 7개 세그먼트
   - Bootstrap 안정성 검증 (ARI = 0.617)

**Track 1 결과**:

| 메트릭 | 값 |
|--------|-----|
| Latent Factor | 5개 |
| Variance Explained | 92.44% |
| Customer Segment | 7개 |
| Bootstrap ARI | 0.617 |
| Davies-Bouldin Index | 0.90 |

**주요 시각화:**

![NMF Factor Loadings](results/figures/factor_loadings_heatmap.png)
*33개 고객 Feature에서 추출된 5개 행동 차원의 NMF Factor Loadings*

![Segment Analysis](results/figures/bubble_a_loyal_vs_deal.png)
*Loyalty (구매 빈도)와 Deal Sensitivity (할인 사용률) 기준 고객 세그먼트 포지셔닝*

### Track 2: Causal Targeting

**접근법**: Heterogeneous Treatment Effects + Policy Learning

1. **Study Design**: First TypeA Campaign Only
   - 2,430명 고객 (Treatment: 1,511, Control: 919)
   - 각 고객이 정확히 한 번만 등장 (clean causal identification)
   - Outcome: 캠페인 후 4주간 구매 금액

2. **Positivity Diagnostics**: Causal identification 품질 평가
   - Propensity Score AUC: 0.989 (심각한 positivity violation)
   - Overlap 영역 [0.1, 0.9]에서 17%만 존재
   - 의미: 결과 해석에 주의 필요

3. **CATE Estimation**: 다양한 Meta-learner
   - S-Learner, T-Learner, X-Learner
   - (R-Learner, DML) LinearDML, CausalForestDML
   - AUUC 기반 모델 선택 (CausalForestDML: 396.3)

4. **Policy Learning**: 최적 타겟팅 규칙
   - Breakeven CATE: $42.43 (비용 $12.73 / 마진 30%)
   - Threshold Policy: CATE > Breakeven이면 타겟팅
   - 보수적 배포를 위한 Risk-adjusted Policy

**Track 2 결과**:

| 메트릭 | 값 |
|--------|-----|
| PS AUC | 0.989 (positivity violation) |
| ATE (trimmed) | $21-41/고객 |
| Best CATE Model | CausalForestDML (AUUC: 396.3) |
| Optimal Targeting | 31.3% 고객 |
| Expected Profit | $2,426 (125% ROI) |

**세그먼트별 전략 / 권고**:

| 세그먼트 | Mean CATE | 액션 |
|----------|-----------|------|
| Regular + H&B | +$34 | 타겟팅 유지 |
| Active Loyalists | +$33 | 타겟팅 유지 |
| VIP Heavy | -$38 | **타겟팅 축소** |
| Bulk Shoppers | -$40 | **타겟팅 축소** |

**주요 시각화:**

![Uplift Curves](results/figures/uplift_auuc_purchase_amount.png)
*CATE 모델별 AUUC 비교 - CausalForestDML이 최고 Uplift 달성*

![Segment CATE](results/figures/segment_bubble.png)
*세그먼트 및 Outcome 차원별 Treatment Effect*

![ROI Optimization](results/figures/roi_curves.png)
*타겟팅 비율별 ROI - 31.3%에서 최적*

---

## Project Structure

```
projects/kr_segmentation_causal_targeting_dunnhumby/
├── notebook/           # 분석 노트북
├── src/                # Python 모듈 (11개 파일)
├── docs/               # 기술 보고서
└── results/
    ├── figures/        # 65개 시각화 파일
    └── *.csv           # 28개 결과 테이블
```

---

## Notebooks

### Track 1: Customer Segmentation

| 노트북 | 설명 |
|--------|------|
| [00_study_design.ipynb](notebook/00_study_design.ipynb) | Study Design 및 2-Track Framework |
| [01_feature_engineering.ipynb](notebook/01_feature_engineering.ipynb) | 33개 고객 Feature 구성 |
| [02_customer_profiling.ipynb](notebook/02_customer_profiling.ipynb) | NMF Latent Factor + K-Means Segmentation |

### Track 2: Causal Targeting

| 노트북 | 설명 |
|--------|------|
| [03a_hte_estimation.ipynb](notebook/03a_hte_estimation.ipynb) | 다양한 방법을 통한 ATE/CATE 추정 |
| [03b_hte_validation.ipynb](notebook/03b_hte_validation.ipynb) | 검증, Refutation Test, Bounds |
| [04_optimal_policy.ipynb](notebook/04_optimal_policy.ipynb) | Policy Learning 및 ROI 최적화 |

---

## Technical Reports

자세한 방법론, 결과, 비즈니스 해석은 다음을 참조한다:

- **[Track 1 Report](docs/track1_report.md)**: Customer Segmentation Analysis
  - NMF Factor 해석, 세그먼트 프로필, 세그먼트별 마케팅 액션

- **[Track 2 Report](docs/track2_report.md)**: Causal Targeting Analysis
  - Positivity Diagnostics, CATE 추정, Policy 비교, A/B Test 설계

각 보고서는 요약, 서론, 방법론, 결과, 논의, 부록(상세 마케팅 해석 포함)을 포함한다.

---

## Technical Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Causal Inference**: econml, dowhy
- **Optimization**: Optuna (hyperparameter tuning), Ray
- **Visualization**: matplotlib, seaborn

---

## Data Source

**Dunnhumby "The Complete Journey"**
- 102주간 2,500 가구
- 260만 거래
- 캠페인 데이터 (TypeA/B/C), 쿠폰 Redemption
- Demographic 세그먼트

데이터셋 문서: [data/dunnhumby/README.md](../../data/dunnhumby/README.md)

---

## Limitations & Future Work

1. **Positivity Violation**: PS AUC = 0.989는 83%의 CATE 추정치가 외삽임을 의미
2. **Refutation Tests**: Placebo Treatment 및 Subset Stability 테스트 실패
3. **권고사항**: 프로덕션 배포 전 A/B Test 검증 필요 (n=5,748)
4. **단일 캠페인 유형**: 분석이 TypeA 캠페인에 한정됨
