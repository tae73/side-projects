# Customer Segmentation Using Latent Factor Modeling: A Retail Analytics Case Study

## Summary

This study presents a behavioral customer segmentation framework applied to retail transaction data from the Dunnhumby "The Complete Journey" dataset. Using Non-negative Matrix Factorization (NMF) for latent factor modeling combined with K-Means clustering, we identified seven distinct customer segments from 2,500 households over a 102-week observation period.

**Key Findings:**
- Five interpretable latent factors explain 92.44% of customer behavioral variance
- Seven customer segments with high stability (Bootstrap ARI = 0.77 ± 0.11)
- VIP segment (12% of customers) generates average sales of $9,716 per customer
- High-value customer groups (44.8% of total) contribute approximately 70% of revenue
- Clear actionable marketing strategies identified for each segment

The segmentation provides a foundation for personalized marketing strategies and serves as input for subsequent causal targeting analysis (Track 2).

---

## 1. Introduction

### 1.1 Background

Customer segmentation is fundamental to modern retail marketing strategy. By grouping customers based on their behavioral patterns, retailers can develop targeted interventions that maximize return on marketing investment. Traditional demographic-based segmentation often fails to capture the nuanced behavioral differences that drive purchase decisions. This study adopts a behavior-first approach, using transaction-derived features to discover natural customer groupings.

### 1.2 Dataset

We analyze the **Dunnhumby "The Complete Journey"** dataset, a comprehensive retail dataset containing:

| Dimension | Value |
|-----------|-------|
| Households | 2,500 |
| Transactions | 2.6 million |
| Time Period | 102 weeks (2 years) |
| Campaigns | 30 marketing campaigns |
| Products | 92,000+ SKUs |
| Stores | 400+ locations |

The dataset includes transaction records, household demographics (32% coverage), campaign targeting, coupon distribution, and redemption data.

### 1.3 Objectives

1. **Extract latent behavioral factors** that characterize customer shopping patterns
2. **Identify distinct customer segments** with actionable marketing implications
3. **Validate segment stability** through bootstrap resampling
4. **Develop segment-specific marketing recommendations**

### 1.4 Analysis Framework

This analysis is part of a 2-Track research framework:

- **Track 1 (This Report)**: Customer Understanding through descriptive segmentation
- **Track 2 (Separate)**: Causal Targeting using heterogeneous treatment effect estimation

Track 1 segments serve as potential moderators for Track 2 causal analysis, enabling segment-specific campaign optimization.

---

## 2. Method

### 2.1 Feature Engineering

We constructed 33 customer-level features from transaction data, organized into six conceptual groups:

| Group | Count | Description | Examples |
|-------|-------|-------------|----------|
| Recency | 6 | Time since last purchase | days_since_last, active_last_4w |
| Frequency | 6 | Shopping frequency patterns | visits_per_week, purchase_regularity |
| Monetary | 7 | Spending characteristics | total_sales, avg_basket_size, coupon_savings |
| Behavioral | 7 | Shopping behavior | discount_rate, private_label_ratio, n_departments |
| Category | 6 | Category preferences | share_grocery, share_fresh, share_h&b |
| Time | 1 | Tenure coverage | week_coverage |

To address multicollinearity, we reduced the feature set from 33 to **19 features** by removing highly correlated pairs (r ≥ 0.7). For example, `frequency_per_week` and `frequency_per_month` had perfect correlation (r = 1.0), so only one was retained.

**Preprocessing:** MinMaxScaler normalization to [0, 1] range was applied to ensure NMF compatibility (requires non-negative input).

### 2.2 Latent Factor Modeling (NMF)

Non-negative Matrix Factorization (NMF) decomposes the customer-feature matrix into two lower-rank matrices, revealing latent behavioral factors.

**Model Selection:**
- Evaluated n_components from 2 to 8
- Selection criteria: reconstruction error (elbow method) and factor interpretability
- **Selected: 5 components** explaining 92.44% of variance

![NMF Component Selection](../results/figures/nmf_component_selection.png)
*Figure 1: NMF component selection showing reconstruction error and cumulative variance explained.*

**NMF Parameters:**
- Solver: Coordinate Descent
- Initialization: Random
- Max iterations: 1,000
- Random state: Fixed for reproducibility

### 2.3 Clustering

K-Means clustering was applied to NMF factor scores to identify customer segments.

**Clustering Evaluation:**
- Tested k = 2 to 11
- Compared K-Means vs. Gaussian Mixture Model (GMM)
- K-Means significantly outperformed GMM (Silhouette: 0.219 vs. 0.047)

**Optimal k Selection:**
- Davies-Bouldin Index: Minimum at k = 7 (DBI = 1.241)
- Silhouette Score: Stable around k = 6-8
- **Selected: k = 7** balancing cluster quality and business interpretability

![Clustering Metrics](../results/figures/clustering_metrics.png)
*Figure 2: Clustering evaluation metrics across different k values.*

### 2.4 Stability Validation

Bootstrap resampling was performed to assess segment stability:
- 100 bootstrap iterations
- 80% sample fraction per iteration
- Metric: Adjusted Rand Index (ARI) between original and bootstrap assignments

---

## 3. Results

### 3.1 Latent Factor Interpretation

NMF identified five interpretable latent factors representing distinct aspects of customer behavior:

| Factor | Name | Top Features (Loading) | Interpretation |
|--------|------|------------------------|----------------|
| **F1** | Grocery Deal Seeker | share_grocery (6.7), discount_pct (5.1), PL_ratio (3.4) | Budget-conscious grocery shoppers seeking discounts |
| **F2** | Loyal Regular | regularity (4.6), n_dept (2.6), frequency (1.0) | High-engagement one-stop shoppers |
| **F3** | Big Basket | monetary_std (2.5), avg_basket (2.4) | Infrequent bulk purchasers |
| **F4** | Fresh Focused | share_fresh (2.3), n_dept (1.2) | Fresh category specialists |
| **F5** | Health & Beauty | share_h&b (2.0) | Drugstore-type shoppers |

![Factor Loadings Heatmap](../results/figures/factor_loadings_heatmap.png)
*Figure 3: NMF factor loadings heatmap showing feature weights for each latent factor.*

The factors naturally separate into **Value dimensions** (F2, F3 capturing frequency and monetary) and **Need dimensions** (F1, F4, F5 capturing category preferences).

### 3.2 Clustering Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Explained Variance | 92.44% | High factor coverage |
| Silhouette Score | 0.219 | Reasonable for behavioral data |
| Calinski-Harabasz Index | 732.0 | High between-cluster variance |
| Davies-Bouldin Index | 1.241 | Good cluster separation |

The Silhouette Score of 0.219, while moderate, is typical for behavioral clustering where customer characteristics exist on a continuum rather than forming discrete groups.

### 3.3 Stability Validation

Bootstrap resampling (100 iterations, 80% sample) yielded an Adjusted Rand Index of **0.767 ± 0.113** (95% CI: 0.55-0.99), indicating high segment stability. An ARI above 0.70 is generally considered strong agreement, confirming that the seven-segment solution is robust to sampling variation.

### 3.4 Seven Customer Segments

The clustering identified seven distinct customer segments:

| Seg | Name | Size | Avg Sales | Frequency | Recency | Dominant Factor |
|-----|------|------|-----------|-----------|---------|-----------------|
| **0** | Active Loyalists | 509 (20.4%) | $3,878 | 171 visits | 6 days | F2 (Loyal) |
| **1** | VIP Heavy | 299 (12.0%) | $9,716 | 256 visits | 4 days | F2 (Loyal) |
| **2** | Lapsed H&B | 193 (7.7%) | $872 | 37 visits | 75 days | F5 (H&B) |
| **3** | Fresh Lovers | 339 (13.6%) | $1,233 | 76 visits | 12 days | F4 (Fresh) |
| **4** | Light Grocery | 524 (21.0%) | $1,100 | 58 visits | 17 days | F4 (Fresh) |
| **5** | Bulk Shoppers | 318 (12.7%) | $3,206 | 56 visits | 39 days | F3 (Basket) |
| **6** | Regular + H&B | 318 (12.7%) | $3,393 | 141 visits | 9 days | F2 (Loyal) |

![Segment Sizes](../results/figures/segment_sizes.png)
*Figure 4: Customer segment size distribution.*

### 3.5 Segment Profiles

![Segment Profiles Heatmap](../results/figures/segment_profiles_heatmap.png)
*Figure 5: Standardized feature profiles (Z-scores) by segment.*

![Factor Scores by Segment](../results/figures/factor_scores_by_segment.png)
*Figure 6: Mean factor scores for each customer segment.*

**Segment Characterizations:**

**Segment 0: Active Loyalists (20.4%)**
- High purchase regularity and diverse category shopping
- Strong private label preference (highest PL ratio at 0.34)
- Budget-conscious but loyal shoppers

**Segment 1: VIP Heavy (12.0%)**
- Top performers across all RFM metrics
- Highest frequency (256 visits), monetary ($9,716), and lowest recency (4 days)
- True one-stop shoppers purchasing 1,316 unique products on average

**Segment 2: Lapsed H&B (7.7%)**
- Highest recency (75 days) - effectively churned
- H&B category specialists with low overall engagement
- Win-back candidates with uncertain ROI

**Segment 3: Fresh Lovers (13.6%)**
- Fresh category specialists with moderate engagement
- Active shoppers (12-day recency) with focused basket

**Segment 4: Light Grocery (21.0%)**
- Largest segment by count, lowest value per customer
- Light engagement with some fresh preference
- Activation opportunity with habit-building potential

**Segment 5: Bulk Shoppers (12.7%)**
- Highest average basket size ($59 per visit)
- Low frequency but high per-visit spending
- Warehouse/Costco-style shopping pattern

**Segment 6: Regular + H&B (12.7%)**
- Second-tier value segment with VIP conversion potential
- Regular shoppers (141 visits) with H&B focus

### 3.6 Value Tier Distribution

Segments naturally separate into value tiers:

| Tier | Segments | % of Customers | Estimated Revenue Share |
|------|----------|----------------|-------------------------|
| **High** | 0, 1, 6 | 44.8% | ~70% |
| **Medium** | 3, 5 | 26.3% | ~20% |
| **Low/At-Risk** | 2, 4 | 28.7% | ~10% |

### 3.7 Multi-Dimensional Segment Positioning

![Bubble Chart: Loyalty vs Deal-Seeking](../results/figures/bubble_a_loyal_vs_deal.png)
*Figure 7: Segment positioning on Loyalty (F2) vs Deal-Seeking (F1) dimensions.*

![Bubble Chart: Frequency vs Monetary](../results/figures/bubble_d_frequency_vs_monetary.png)
*Figure 8: RFM value positioning showing VIP dominance and segment differentiation.*

![Bubble Chart: Recency vs Monetary](../results/figures/bubble_f_recency_vs_monetary.png)
*Figure 9: Customer lifecycle positioning identifying active high-value and lapsed segments.*

---

## 4. Discussion

### 4.1 Key Insights

**1. Clear Value Hierarchy**
The segmentation reveals a clear Pareto distribution: 44.8% of customers in high-value segments contribute approximately 70% of estimated revenue. VIP Heavy (12%) alone represents the most critical retention target.

**2. Behavioral Differentiation**
Factors successfully separate customers along both **Value** (frequency, monetary) and **Need** (category preference) dimensions. This dual structure enables both value-based prioritization and need-based personalization.

**3. Lifecycle Stages**
Segments map to distinct lifecycle stages:
- Active/Growing: Segments 0, 1, 6 (low recency, high engagement)
- Stable: Segments 3, 4 (moderate recency)
- Declining/Churned: Segment 2 (high recency, low engagement)

**4. Category Specialists**
Fresh Lovers (13.6%) and H&B-focused segments demonstrate category specialization, suggesting opportunity for category-specific marketing approaches.

### 4.2 Marketing Recommendations

| Segment | Priority | Strategy | Key Actions |
|---------|----------|----------|-------------|
| VIP Heavy | High | Retention | Premium benefits, churn prediction alerts, exclusive access |
| Active Loyalists | High | Strengthen | Private label promotions, loyalty points, basket expansion |
| Regular + H&B | Medium | Upgrade | VIP conversion program, cross-category incentives |
| Bulk Shoppers | Medium | Regularize | Subscription offers, scheduled delivery, bundle deals |
| Fresh Lovers | Medium | Engage | Fresh content marketing, daily specials, recipe inspiration |
| Light Grocery | Low | Activate | Habit-building campaigns, progressive rewards, onboarding |
| Lapsed H&B | Low | Win-back | Re-engagement campaigns, H&B-focused offers |

**Recommended Budget Allocation:**
- High Priority (60%): VIP Heavy (25%), Active Loyalists (20%), Regular + H&B (15%)
- Medium Priority (30%): Bulk Shoppers (10%), Fresh Lovers (10%), Light Grocery (10%)
- Low Priority (10%): Lapsed H&B (10%)

### 4.3 Limitations

**1. Moderate Silhouette Score (0.219)**
Behavioral data inherently exhibits continuity rather than discrete boundaries. The score is acceptable for customer segmentation but indicates some overlap between segments.

**2. Limited Demographic Coverage (32%)**
Only 801 of 2,500 households have demographic information, limiting demographic-based stratification and persona development.

**3. Descriptive vs. Causal**
This segmentation is descriptive. Questions like "Which segment responds best to promotions?" require causal analysis (Track 2).

**4. Single Retailer Context**
Results are specific to this retailer's customer base and may not generalize to different retail contexts.

### 4.4 Future Directions

**1. Track 2 Integration**
Segments will serve as heterogeneous treatment effect moderators in Track 2 causal analysis. This enables segment-specific campaign effectiveness estimation.

**2. A/B Testing Validation**
Recommended strategies should be validated through controlled experiments before full-scale deployment.

**3. Dynamic Segmentation**
Periodic re-clustering to capture segment migration and evolving customer behaviors.

**4. Value × Need Framework**
Optional extension using separate Value (RFM) and Need (Category) factor models for cross-sell optimization scenarios.

---

## 5. Conclusion

This study demonstrates an effective approach to behavioral customer segmentation using latent factor modeling and clustering. The NMF + K-Means framework successfully identified seven distinct customer segments with high stability (ARI = 0.77) and clear business interpretability.

Key outcomes include:
- **Five latent factors** capturing Value (loyalty, monetary) and Need (category preference) dimensions
- **Seven actionable segments** ranging from VIP Heavy ($9,716 avg) to Lapsed H&B ($872 avg)
- **Clear priority tiers** with 44.8% high-value customers warranting focused retention efforts
- **Segment-specific strategies** from retention (VIP) to activation (Light Grocery) to win-back (Lapsed)

The segmentation provides a robust foundation for personalized marketing and serves as input for subsequent causal targeting analysis, enabling evidence-based marketing optimization.

---

## Appendix: Technical Details

### A.1 Software Environment
- Python 3.9+
- scikit-learn (NMF, K-Means)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)

### A.2 Reproducibility
- Random seeds fixed for all stochastic processes
- Full code available in project notebooks:
  - `01_feature_engineering.ipynb`
  - `02_customer_profiling.ipynb`

### A.3 Data Artifacts
- Segment assignments: `data/dunnhumby/processed/segment_models.joblib`
- Feature metadata: `data/dunnhumby/processed/feature_metadata.json`

### A.4 Segment Positioning Analysis: Bubble Charts & Marketing Actions

This section provides detailed marketing interpretations for each two-dimensional segment positioning chart.

---

#### A.4.1 Loyalty (F2) vs Deal-Seeking (F1)

![Bubble A: Loyalty vs Deal-Seeking](../results/figures/bubble_a_loyal_vs_deal.png)

| Quadrant | Segments | Profile | Marketing Action |
|----------|----------|---------|------------------|
| **High Loyalty + High Deal** | Active Loyalists | Loyal but price-sensitive | PB promotions, loyalty points with discount triggers |
| **High Loyalty + Low Deal** | VIP Heavy | Premium loyal customers | Exclusive access, premium services, avoid discounts |
| **Low Loyalty + High Deal** | Light Grocery, Fresh Lovers | Cherry-pickers | Convert to loyalty via progressive rewards |
| **Low Loyalty + Low Deal** | Lapsed H&B, Bulk Shoppers | Disengaged or transactional | Win-back or accept low engagement |

---

#### A.4.2 Loyalty (F2) vs Big Basket (F3)

![Bubble B: Loyalty vs Big Basket](../results/figures/bubble_b_loyal_vs_bigbasket.png)

| Quadrant | Segments | Profile | Marketing Action |
|----------|----------|---------|------------------|
| **High Loyalty + High Basket** | VIP Heavy | One-stop power shoppers | Retention focus, personalized recommendations |
| **High Loyalty + Low Basket** | Active Loyalists, Regular+H&B | Frequent small baskets | Basket expansion via cross-sell, bundle offers |
| **Low Loyalty + High Basket** | Bulk Shoppers | Infrequent large purchases | Subscription model, scheduled delivery incentives |
| **Low Loyalty + Low Basket** | Lapsed, Light Grocery | Minimal engagement | Activation campaigns, habit-building |

---

#### A.4.3 Fresh (F4) vs Health & Beauty (F5)

![Bubble C: Fresh vs H&B](../results/figures/bubble_c_fresh_vs_hb.png)

| Quadrant | Segments | Profile | Marketing Action |
|----------|----------|---------|------------------|
| **High Fresh + Low H&B** | Fresh Lovers | Cooking/health conscious | Recipe content, farm-to-store stories, daily fresh deals |
| **Low Fresh + High H&B** | Lapsed H&B, Regular+H&B | Drugstore-type needs | H&B sampling, beauty membership, health subscriptions |
| **Balanced** | VIP Heavy, Active Loyalists | Full-basket shoppers | Cross-category promotions, one-stop convenience |
| **Low Both** | Light Grocery, Bulk | Staples-focused | Category expansion incentives |

---

#### A.4.4 Frequency vs Monetary (RFM Core)

![Bubble D: Frequency vs Monetary](../results/figures/bubble_d_frequency_vs_monetary.png)

| Quadrant | Segments | Profile | Marketing Action |
|----------|----------|---------|------------------|
| **High Freq + High Monetary** | VIP Heavy | Best customers | Protect at all costs, premium treatment |
| **High Freq + Low Monetary** | Active Loyalists | Frequent small spenders | Basket size expansion, upselling |
| **Low Freq + High Monetary** | Bulk Shoppers | Warehouse-style | Increase visit frequency, subscription |
| **Low Freq + Low Monetary** | Lapsed, Light Grocery | At-risk/dormant | Segment for win-back ROI assessment |

---

#### A.4.5 Regularity vs Average Basket Size

![Bubble E: Regularity vs Basket](../results/figures/bubble_e_regularity_vs_basket.png)

| Quadrant | Segments | Profile | Marketing Action |
|----------|----------|---------|------------------|
| **High Regularity + High Basket** | VIP Heavy | Predictable high-value | Maintain rhythm, anticipate needs |
| **High Regularity + Low Basket** | Active Loyalists | Consistent small trips | Top-up to stock-up conversion |
| **Low Regularity + High Basket** | Bulk Shoppers | Sporadic big shops | Regularize via reminders, auto-replenishment |
| **Low Regularity + Low Basket** | Lapsed, Light | Unpredictable low-value | Accept or targeted reactivation |

---

#### A.4.6 Recency vs Monetary (Lifecycle)

![Bubble F: Recency vs Monetary](../results/figures/bubble_f_recency_vs_monetary.png)

| Quadrant | Segments | Profile | Marketing Action |
|----------|----------|---------|------------------|
| **Low Recency + High Monetary** | VIP Heavy, Active Loyalists | Active high-value | Retention, prevent churn signals |
| **Low Recency + Low Monetary** | Fresh Lovers, Light Grocery | Active low-value | Grow value via cross-sell |
| **High Recency + High Monetary** | (Rare) | Recently churned VIP | Urgent win-back with premium offer |
| **High Recency + Low Monetary** | Lapsed H&B | Churned low-value | Low-priority win-back, accept churn |

---

#### A.4.7 Discount Rate vs Private Label Ratio

![Bubble G: Discount vs PL](../results/figures/bubble_g_discount_vs_pl.png)

| Quadrant | Segments | Profile | Marketing Action |
|----------|----------|---------|------------------|
| **High Discount + High PL** | Active Loyalists | Budget maximizers | PB-focused promotions, value messaging |
| **High Discount + Low PL** | Fresh Lovers | Brand loyal deal-seekers | National brand promotions, PB trial incentives |
| **Low Discount + High PL** | Regular+H&B | Quality-seeking PB fans | Premium PB lines, new PB launches |
| **Low Discount + Low PL** | VIP Heavy, Bulk | Price-insensitive | Avoid discounts, focus on convenience/quality |

---

#### A.4.8 Shopping Variety vs Regularity

![Bubble H: Variety vs Regularity](../results/figures/bubble_h_variety_vs_regularity.png)

| Quadrant | Segments | Profile | Marketing Action |
|----------|----------|---------|------------------|
| **High Variety + High Regularity** | VIP Heavy | Ultimate one-stop shopper | Full personalization, category captain |
| **High Variety + Low Regularity** | Bulk Shoppers | Occasional comprehensive trips | Convert to regular cadence |
| **Low Variety + High Regularity** | Fresh Lovers | Category specialists | Category depth, expand to adjacent |
| **Low Variety + Low Regularity** | Lapsed, Light | Narrow, infrequent | Broaden basket first, then frequency |
