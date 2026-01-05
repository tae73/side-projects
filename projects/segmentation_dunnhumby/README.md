# Customer Segmentation - Dunnhumby

## Overview

Customer segmentation research using the Dunnhumby "The Complete Journey" dataset, implementing a **2-Track Framework** for comprehensive customer analysis.

## Dataset

- Source: `data/dunnhumby/`
- EDA: `data/dunnhumby/eda/exploration_*.ipynb`

## Project Structure

```
projects/segmentation_dunnhumby/
├── experiments/     # Experiment designs and configurations
├── notebook/        # Analysis notebooks
│   ├── 00_study_design.ipynb       # Study design documentation
│   ├── 01_feature_engineering.ipynb # Feature construction
│   └── ...
├── results/         # Output files and findings
└── src/             # Source code
    └── features.py  # Feature engineering functions
```

## 2-Track Framework

### Framework Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         2-TRACK STUDY DESIGN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRACK 1: CUSTOMER UNDERSTANDING (Descriptive)                              │
│  ═══════════════════════════════════════════════                            │
│                                                                             │
│  Step 1.1: Exploratory                                                      │
│    - Factor Analysis (NMF) → Discover latent dimensions                     │
│    - Clustering → Derive base segments                                      │
│                                                                             │
│  Step 1.2: Value × Need Integration                                         │
│    - Value layer: CLV, Engagement                                           │
│    - Need layer: Behavior, Category preference                              │
│    - Integration: Value-Need Matrix or 2D Segmentation                      │
│                                                                             │
│  Output: "To whom (Value) should we offer what (Need)?"                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRACK 2: CAUSAL TARGETING                                                  │
│  ═════════════════════════                                                  │
│                                                                             │
│  Step 2.1: HTE Analysis                                                     │
│    - Campaign effect heterogeneity                                          │
│    - Covariates: Track 1 segments + raw features                            │
│                                                                             │
│  Step 2.2: Optimal Policy                                                   │
│    - Targeting rules                                                        │
│    - Policy value estimation                                                │
│                                                                             │
│  Output: "How do we optimize targeting?"                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Practical Usage Comparison

| Aspect | Track 1 (Descriptive) | Track 2 (Causal) |
|--------|----------------------|------------------|
| **Core Question** | "Who is this customer?" | "Will this work for them?" |
| **Primary Users** | Marketing, CRM, Strategy | Data Science, Optimization |
| **Usage Timing** | Campaign planning, Strategy | Campaign execution, A/B optimization |
| **Difficulty** | Low | High |
| **Explainability** | "Premium Fresh Lover segment" | "CATE = 0.15 for this customer" |
| **Usage Frequency** | High (daily operations) | Medium (specific scenarios) |
| **Org Requirements** | Basic analytics capability | Causal thinking culture required |

### Marketer's Perspective

**Track 1 (Almost certainly utilized):**
- Intuitive: "High-Value + Deal Seeker" → immediately understandable
- Easy to act on: Select segment → Design offer
- Easy to communicate: Explainable to executives and other departments
- Stable: Segments don't change frequently

**Track 2 (Depends on organizational maturity):**
- Requires conceptual understanding: "Treatment effect", "Uplift"
- Requires experimentation infrastructure: A/B testing systems
- Interpretation gap: Numbers need translation to actionable insights
- High value: Enables ROI optimization when properly implemented

## Methods

### Track 1: Customer Understanding

**Step 1.1: Exploratory Analysis**
- Non-negative Matrix Factorization (NMF) for latent dimension discovery
- Clustering (K-Means, GMM) for segment derivation

**Step 1.2: Value × Need Integration**
- Value Layer: RFM + Engagement features (CLV estimation)
- Need Layer: Behavioral + Category features
- Integration: Value-Need Matrix or sequential segmentation

### Track 2: Causal Targeting

**Step 2.1: HTE Analysis**
- Meta-learners: S-Learner, T-Learner, X-Learner, R-Learner
- Causal Forest (GRF) for CATE estimation and rule extraction

**Step 2.2: Optimal Policy**
- Subgroup discovery via Causal Tree
- Policy learning and value estimation

## Causal Graph

```
                Marketing Exposure (display, mailer)
                           │
                           ↓
Campaign Targeting ──────→ Purchase ←────── Customer Characteristics
         │                                     (confounders)
         │                                          │
         └──────→ Coupon Redemption ←───────────────┘
```

## Evaluation Criteria

### Track 1
- **Technical**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- **Interpretability**: Factor loadings, Cluster separation
- **Business**: CLV difference significance, Actionability

### Track 2
- **Causal**: ATE significance, HTE existence (BLP test), CATE distribution
- **Prediction**: AUUC, Qini coefficient, Calibration
- **Policy**: Policy value, Targeting efficiency, ROI

### Additional Criteria
- **Stability**: Segment consistency across periods
- **Robustness**: Bootstrap, Cross-validation, Model comparison
- **Causal Validation**: Refutation tests, Sensitivity analysis (E-value)
- **Business Impact**: Incremental lift, Cost-benefit analysis

## References

- [Dunnhumby Dataset Documentation](../../data/dunnhumby/README.md)
- [Study Design Notebook](notebook/00_study_design.ipynb)