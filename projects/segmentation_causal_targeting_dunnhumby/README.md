# Customer Segmentation & Causal Targeting - Dunnhumby

## Executive Summary

This project implements a **2-Track Framework** for customer analysis using the Dunnhumby "The Complete Journey" retail dataset. By combining descriptive segmentation with causal inference, we address two fundamental marketing questions:

- **Track 1**: "Who are our customers?" → 7 distinct customer segments
- **Track 2**: "Who should we target?" → Optimal 31.3% targeting for 125% ROI improvement

### Key Results

| Track | Key Finding | Business Impact |
|-------|-------------|-----------------|
| Track 1 | 7 segments with 92.44% variance explained | Actionable customer profiles for CRM |
| Track 2 | 31.3% optimal targeting vs 100% baseline | $7,000+ profit improvement |
| Track 2 | VIP Heavy, Bulk Shoppers show negative CATE | Reduce targeting for these segments |

---

## Research Motivation

Traditional marketing segmentation answers "who are our customers" but fails to answer "will this campaign work for them?" This project bridges the gap:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         2-TRACK FRAMEWORK                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRACK 1: CUSTOMER UNDERSTANDING                                            │
│  ════════════════════════════════                                           │
│                                                                             │
│  "Who are our customers?"                                                   │
│                                                                             │
│  • Latent Factor Modeling (NMF)   → Discover behavioral dimensions          │
│  • Clustering (K-Means)           → Derive actionable segments              │
│  • Stability Validation           → Ensure segment reliability              │
│                                                                             │
│  Output: Customer profiles for marketing strategy                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRACK 2: CAUSAL TARGETING                                                  │
│  ═════════════════════════                                                  │
│                                                                             │
│  "Who should we target?"                                                    │
│                                                                             │
│  • Heterogeneous Treatment Effects → Campaign effect by customer            │
│  • Policy Learning                 → Optimal targeting rules                │
│  • ROI Optimization                → Maximize campaign profit               │
│                                                                             │
│  Output: Data-driven targeting policy                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why Both Tracks?**

| Aspect | Track 1 (Descriptive) | Track 2 (Causal) |
|--------|----------------------|------------------|
| Core Question | "Who is this customer?" | "Will this work for them?" |
| Primary Users | Marketing, CRM, Strategy | Data Science, Optimization |
| Explainability | "Premium Fresh Lover segment" | "CATE = +$34 for this customer" |
| Org Requirements | Basic analytics | Causal thinking culture |

---

## Methodology

### Track 1: Customer Segmentation

**Approach**: Latent Factor Modeling + Clustering

1. **Feature Engineering**: 33 customer-level features
   - RFM features (Recency, Frequency, Monetary)
   - Behavioral features (discount usage, variety seeking)
   - Category preferences (grocery, fresh, health & beauty)

2. **Latent Factor Modeling (NMF)**: Discover underlying behavioral dimensions
   - 5 interpretable factors extracted
   - 92.44% cumulative variance explained

3. **Clustering (K-Means)**: Derive customer segments
   - 7 segments via elbow method + silhouette analysis
   - Bootstrap stability validation (ARI = 0.617)

**Track 1 Results**:

| Metric | Value |
|--------|-------|
| Latent Factors | 5 |
| Variance Explained | 92.44% |
| Customer Segments | 7 |
| Bootstrap ARI | 0.617 |
| Davies-Bouldin Index | 0.90 |

**Key Visualizations:**

![NMF Factor Loadings](results/figures/factor_loadings_heatmap.png)
*NMF factor loadings showing 5 behavioral dimensions extracted from 33 customer features*

![Segment Analysis](results/figures/bubble_a_loyal_vs_deal.png)
*Customer segments positioned by loyalty (frequency) and deal sensitivity (discount usage)*

### Track 2: Causal Targeting

**Approach**: Heterogeneous Treatment Effects + Policy Learning

1. **Study Design**: First TypeA Campaign Only
   - 2,430 customers (Treatment: 1,511, Control: 919)
   - Each customer appears exactly once (clean causal identification)
   - Outcome: 4-week post-campaign purchase amount

2. **Positivity Diagnostics**: Assess causal identification quality
   - Propensity Score AUC: 0.989 (severe positivity violation)
   - Only 17% in overlap region [0.1, 0.9]
   - Implications: Results require careful interpretation

3. **CATE Estimation**: Multiple meta-learners
   - S-Learner, T-Learner, X-Learner
   - LinearDML, CausalForestDML
   - Model selection via AUUC (CausalForestDML: 396.3)

4. **Policy Learning**: Optimal targeting rules
   - Breakeven CATE: $42.43 (cost $12.73 / margin 30%)
   - Threshold policy: Target if CATE > Breakeven
   - Risk-adjusted policies for conservative deployment

**Track 2 Results**:

| Metric | Value |
|--------|-------|
| PS AUC | 0.989 (positivity violation) |
| ATE (trimmed) | $21-41/customer |
| Best CATE Model | CausalForestDML (AUUC: 396.3) |
| Optimal Targeting | 31.3% of customers |
| Expected Profit | $2,426 (125% ROI) |

**Segment-Level Recommendations**:

| Segment | Mean CATE | Action |
|---------|-----------|--------|
| Regular + H&B | +$34 | Maintain Targeting |
| Active Loyalists | +$33 | Maintain Targeting |
| VIP Heavy | -$38 | **Reduce Targeting** |
| Bulk Shoppers | -$40 | **Reduce Targeting** |

**Key Visualizations:**

![Uplift Curves](results/figures/uplift_auuc_purchase_amount.png)
*AUUC comparison across CATE models - CausalForestDML achieves highest uplift*

![Segment CATE](results/figures/segment_bubble.png)
*Treatment effect by segment and outcome dimension*

![ROI Optimization](results/figures/roi_curves.png)
*ROI by targeting percentage showing optimal at 31.3%*

---

## Project Structure

```
projects/segmentation_causal_targeting_dunnhumby/
├── notebook/           # Analysis notebooks
├── src/                # Python modules (11 files)
├── docs/               # Technical reports
└── results/
    ├── figures/        # 65 visualization files
    └── *.csv           # 28 result tables
```

---

## Notebooks

### Track 1: Customer Segmentation

| Notebook | Description |
|----------|-------------|
| [00_study_design.ipynb](notebook/00_study_design.ipynb) | Study design and 2-Track framework |
| [01_feature_engineering.ipynb](notebook/01_feature_engineering.ipynb) | 33 customer features construction |
| [02_customer_profiling.ipynb](notebook/02_customer_profiling.ipynb) | NMF latent factors + K-Means segmentation |

### Track 2: Causal Targeting

| Notebook | Description |
|----------|-------------|
| [03a_hte_estimation.ipynb](notebook/03a_hte_estimation.ipynb) | ATE/CATE estimation with multiple methods |
| [03b_hte_validation.ipynb](notebook/03b_hte_validation.ipynb) | Validation, refutation tests, bounds |
| [04_optimal_policy.ipynb](notebook/04_optimal_policy.ipynb) | Policy learning and ROI optimization |

---

## Technical Reports

For detailed methodology, results, and business interpretations:

| Report | Language | Description |
|--------|----------|-------------|
| [Track 1 Report](docs/track1_report.md) | English | Customer Segmentation Analysis |
| [Track 1 Report (KO)](docs/track1_report_ko.md) | 한국어 | 고객 세분화 분석 |
| [Track 2 Report](docs/track2_report.md) | English | Causal Targeting Analysis |
| [Track 2 Report (KO)](docs/track2_report_ko.md) | 한국어 | Causal Targeting 분석 |

Each report includes: Summary, Introduction, Methods, Results, Discussion, and Appendix with detailed marketing interpretations.

---

## Technical Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Causal Inference**: econml, dowhy
- **Optimization**: Optuna (hyperparameter tuning)
- **Visualization**: matplotlib, seaborn

---

## Data Source

**Dunnhumby "The Complete Journey"**
- 2,500 households over 102 weeks
- 2.6 million transactions
- Campaign data (TypeA/B/C), coupon redemptions
- Demographic segments

Dataset documentation: [data/dunnhumby/README.md](../../data/dunnhumby/README.md)

---

## Limitations & Future Work

1. **Positivity Violation**: PS AUC = 0.989 indicates 83% of CATE estimates are extrapolations
2. **Refutation Tests**: Placebo treatment and subset stability tests failed
3. **Recommendation**: A/B test validation (n=5,748) required before production deployment
4. **Single Campaign Type**: Analysis limited to TypeA campaigns