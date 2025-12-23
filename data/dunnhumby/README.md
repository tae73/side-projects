# Dunnhumby - The Complete Journey Dataset

## Overview

"The Complete Journey" by dunnhumby is a comprehensive retail transaction dataset capturing the complete shopping journey of 2,500 households over a 2-year period (approximately 102 weeks). 
Unlike category-specific datasets (e.g., beverage-only or frozen food-only data), this includes all purchases made by these households, along with demographic information and direct marketing campaign history.

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Time Period | 2 years (Week 1-102) |
| Households | 2,500 |
| Transactions | 2,595,732 |
| Baskets | 276,484 |
| Campaigns | 30 (TypeA, TypeB, TypeC) |
| Stores | 7 |
| Data Coverage | Transaction details, Demographics, Campaign exposure, Coupon redemption, Product attributes, Promotional activities |

### Exploratory Data Analysis

| Notebook | Description |
|----------|-------------|
| [exploration_1.ipynb](eda/exploration_1.ipynb) | Household Demographics & Transaction Patterns |
| [exploration_2.ipynb](eda/exploration_2.ipynb) | Product Analysis |
| [exploration_3.ipynb](eda/exploration_3.ipynb) | Campaign Analysis |
| [exploration_4.ipynb](eda/exploration_4.ipynb) | Marketing Exposure Analysis |

**Table of Contents:**

1. Household Demographics
   - 1.1 Data Loading & Overview
   - 1.2 Distributions
   - 1.3 Correlation Analysis
2. Transaction Patterns
   - 2.1 Data Loading & Cleaning
   - 2.2 Household-level Analysis
   - 2.3 Store-level Analysis
   - 2.4 Basket-level Analysis
   - 2.5 Time Trends
   - 2.6 Discount Analysis
   - 2.7 Segment Analysis & Demographics
3. Product Analysis
   - 3.1 Data Loading & Overview
   - 3.2 Product Hierarchy & Distribution
   - 3.3 Popular Products & Repeat Purchase
   - 3.4 Product Concentration (Long-tail)
   - 3.5 Customer-Category Relationship
   - 3.6 Price Analysis
   - 3.7 Discount Patterns
4. Campaign Analysis
   - 4.1 Data Loading & Overview
   - 4.2 Campaign Timeline & Schedule
   - 4.3 Campaign Targeting Analysis
   - 4.4 Coupon Structure
   - 4.5 Coupon Redemption Analysis
   - 4.6 Demographic Comparison
5. Marketing Exposure Analysis (causal_data)
   - 5.1 Data Loading & Overview
   - 5.2 Display & Mailer Distributions
   - 5.3 Exposure Rate by Category
   - 5.4 Data Quality Check (Outlier Departments)
   - 5.5 Exposure-Sales Relationship
   - 5.6 Combined Effects (Display × Mailer)

---

## Dataset Strengths

### 1. Real-World Treatment Information

This dataset provides detailed information on actual marketing interventions at multiple levels, making it particularly valuable for causal inference research:

**Campaign-Level Treatments (Household-Targeted)**
- Precise campaign exposure periods with explicit start and end dates
- Three distinct campaign types with different targeting strategies
  - TypeA: Personalized campaigns with targeted coupon selection
  - TypeB & TypeC: Standardized campaigns with uniform coupon distribution
- Complete coupon redemption tracking linked to specific transactions

**Promotional Treatments (Product-Store-Week Level)**
- **Display Placements**: 10 distinct in-store display locations tracked weekly by product and store
  - End caps (front, mid-aisle, rear, side-aisle)
  - In-aisle and store front/rear placements
  - Secondary location displays
- **Mailer Placements**: 12 distinct weekly advertising positions
  - Feature placements (front page, back page, wrap, interior)
  - Coupon placements (interior, wrap)
  - Line items and free product ads

**Treatment Complexity Reflecting Real Business**
- Multiple simultaneous treatments at different granularities
- Independent and combined treatment effects (campaign + display + mailer)
- Time-varying promotional strategies across stores and products
- Enables analysis of synergistic effects between treatment types

### 2. Comprehensive Customer Data

The dataset offers multi-dimensional customer information at transaction-level granularity:

- **Transaction Details**: Line-item records equivalent to point-of-sale receipts
- **Demographic Segmentation**: Seven anonymized classification variables with meaningful ordinality
- **Detailed Pricing Components**:
  - `sales_value`: Revenue received by retailer
  - `retail_disc`: Loyalty program discount
  - `coupon_disc`: Manufacturer coupon discount
  - `coupon_match_disc`: Retailer coupon matching discount

### 3. Rich Contextual Information

Beyond transaction records, the dataset includes promotional and product context that can serve as both control variables and treatment effects:

- **Product Hierarchy**: Three-level classification (Department → Commodity → Sub-commodity)
- **Promotional Activities as Treatments**: Weekly variation in display locations and mailer placements by product and store
  - Enables analysis of promotional effectiveness and optimal placement strategies
  - Product-store-week granularity allows for detailed causal analysis
  - Can be analyzed independently or in combination with coupon campaigns
- **Multi-Store Data**: Enables analysis of promotional variations across retail locations and natural experiments
- **Brand Information**: Distinction between private label and national brand products for brand effect analysis

### 4. Large-Scale Longitudinal Structure

The dataset's panel structure and scale enable sophisticated analytical approaches:

- 2.6 million+ transactions support deep learning and complex statistical models
- 102-week observation period allows for long-term trend analysis
- Panel data structure facilitates causal inference methodologies
- Repeated customer observations enable within-subject analysis

---

## Use Cases & Project Ideas

Given the strengths outlined above, this dataset is particularly well-suited for the following research and analytical applications:

### 1. Customer Analytics & Segmentation

Leveraging the comprehensive customer data and longitudinal structure, researchers can:

- **Latent Factor Modeling**: Extract multi-dimensional customer characteristics including price sensitivity, brand preference, category affinity, and purchase frequency patterns
- **Behavioral Segmentation**: Develop sophisticated segmentation schemes beyond traditional RFM analysis using the complete purchase history
- **Temporal Pattern Analysis**: Examine changes in shopping behavior over time using the 102-week observation period
- **Demographic Profiling**: Analyze relationships between anonymized demographic variables and purchasing patterns

### 2. Causal Inference & Treatment Effect Estimation

The dataset's real-world intervention data makes it particularly valuable for causal inference research, addressing a critical gap in publicly available datasets.

#### Methodological Advantages

This dataset enables realistic causal inference analysis that closely mirrors actual business environments:

- **Multiple Concurrent Treatments**: Thirty distinct campaigns operating simultaneously, plus weekly promotional variations (display and mailer placements)
- **Multi-Level Treatments**: Household-level campaigns and product-store-week level promotions operating independently or in combination
- **Heterogeneous Treatment Timing**: Each campaign has distinct exposure periods, with promotional treatments varying weekly
- **Panel Data Structure**: Repeated observations of identical households enable difference-in-differences and fixed effects approaches
- **Treatment Assignment Variation**: Both personalized (TypeA) and uniform (TypeB/C) campaign structures available for comparative analysis
- **Treatment Interactions**: Ability to analyze synergistic effects between campaigns, display placements, and mailer features

#### Applicable Research Questions

- Estimation of conditional average treatment effects (CATE) across customer segments
- Quantification of incremental revenue attributable to marketing interventions
- Analysis of treatment interference and spillover effects
- **Measurement of promotional effectiveness**: Display location and mailer placement impact on sales
- **Treatment synergy analysis**: Combined effects of coupons, displays, and advertising

### 3. Personalization & Optimal Targeting

Leveraging insights from treatment heterogeneity analysis, researchers can develop sophisticated targeting and personalization strategies:

#### Core Research Questions

- **Identification of treatment heterogeneity and optimal targeting rules**: Characterize how treatment effects vary across customer segments and derive actionable targeting criteria
- **Development and validation of uplift modeling approaches**: Build models that predict individual-level treatment effects for campaign assignment decisions
- **Optimal promotional strategy design**: Determine which products benefit most from which promotional tactics (display locations, mailer placements, coupon types)

#### Targeting Applications

- **CATE-Based Targeting**: Use estimated conditional average treatment effects to identify customers most likely to respond to specific campaigns
- **Uplift-Based Selection**: Target customers with positive treatment effects while avoiding those with negative or zero effects
- **Multi-Treatment Optimization**: Assign customers to their optimal campaign type (TypeA, TypeB, TypeC) based on predicted responses
- **Budget Allocation**: Optimize marketing spend by targeting high-uplift customers and reallocating resources from low-response segments

#### Personalization Strategies

- **Campaign Personalization**: Design individualized campaign offers based on customer characteristics and predicted treatment effects
- **Promotional Tactics Personalization**: Determine optimal promotional mix (coupon + display + mailer) for each product-customer pair
- **Timing Optimization**: Identify optimal campaign timing based on customer purchase cycles and treatment effect dynamics
- **Channel Selection**: Choose between direct mail campaigns, in-store promotions, or their combination based on customer preferences

#### Business Value

- Maximize return on marketing investment through precision targeting
- Reduce campaign costs by avoiding customers unlikely to respond
- Increase customer satisfaction through relevant, personalized offers
- Enable real-time decision systems for promotional strategy

### 4. Recommendation & Prediction Systems

The large-scale transactional data supports various predictive modeling applications:

- **Next Basket Prediction**: Forecast future purchases using sequential transaction patterns
- **Cross-Category Recommendations**: Identify complementary products based on co-purchase behavior
- **Temporal Recommendation Models**: Incorporate time-varying preferences and seasonal patterns
- **Personalized Product Ranking**: Leverage demographic and behavioral features for individualized recommendations

---

## Limitations & Considerations

### 1. Sample Size Constraints

The dataset contains 2,500 households, which presents certain analytical constraints:

- Fine-grained segmentation may result in insufficient sample sizes for robust statistical inference
- Complex models with large parameter spaces may be susceptible to overfitting
- Statistical power for detecting small effect sizes in experimental designs may be limited
- Generalizability to broader populations requires careful consideration

### 2. Variable Anonymization

Demographic variables have been anonymized with generic labels:

- Variables are named `CLASSIFICATION_1` through `CLASSIFICATION_7` without semantic labels
- While values maintain meaningful ordinality, actual demographic dimensions (e.g., income, household size, education) are not specified
- This limits domain-specific interpretability and external validation of findings

### 3. Incomplete Campaign Assignment Data

For TypeA campaigns, the dataset has inherent limitations:

- Each household received 16 coupons from a larger pool, selected based on purchase history
- The specific 16 coupons received by each household cannot be identified
- Only the complete pool of available coupons is provided
- This limits certain causal inference applications that require precise treatment assignment knowledge

---

## Data Structure

### Core Tables

#### 1. **transaction_data** (Transaction Records)
Primary table containing all purchases.

| Variable | Description |
|----------|-------------|
| household_key | Unique household identifier |
| basket_id | Unique shopping session identifier |
| day | Transaction day (1-714) |
| product_id | Unique product identifier |
| quantity | Number of products purchased |
| sales_value | Revenue received by retailer |
| store_id | Store identifier |
| coupon_match_disc | Discount from retailer's coupon matching |
| coupon_disc | Discount from manufacturer coupon |
| retail_disc | Discount from loyalty card |
| trans_time | Time of transaction |
| week_no | Week number (1-102) |

#### 2. **hh_demographic** (Household Demographics)
Demographic information for households (anonymized with meaningful ordinality).

| Variable | Description |
|----------|-------------|
| household_key | Unique household identifier |
| CLASSIFICATION_1 | Demographic segment (Group1-Group6, ordered) |
| CLASSIFICATION_2 | Demographic segment (X, Y, Z) |
| CLASSIFICATION_3 | Demographic segment (Level1-Level12, ordered) |
| CLASSIFICATION_4 | Demographic segment (1-5+, ordered) |
| CLASSIFICATION_5 | Demographic segment (Group1-Group6, ordered) |
| CLASSIFICATION_6 | Demographic segment (Group1-Group5, ordered) |
| CLASSIFICATION_7 | Demographic segment (1, 2, 3, None/Unknown, ordered) |

#### 3. **campaign_table** (Campaign Receipt History)
Lists campaigns received by each household.

| Variable | Description |
|----------|-------------|
| household_key | Unique household identifier |
| campaign | Campaign ID (1-30) |
| description | Campaign type (TypeA, TypeB, TypeC) |

#### 4. **campaign_desc** (Campaign Duration)
Campaign start and end dates.

| Variable | Description |
|----------|-------------|
| campaign | Campaign ID (1-30) |
| description | Campaign type |
| start_day | Campaign start date |
| end_day | Campaign end date |

#### 5. **coupon** (Coupon-Product Mapping)
Maps coupons to redeemable products. Note: One coupon can be valid for multiple products.

| Variable | Description |
|----------|-------------|
| campaign | Campaign ID |
| coupon_upc | Unique coupon identifier (per household per campaign) |
| product_id | Product for which coupon is redeemable |

**Important Notes:**
- **TypeA campaigns**: Pool of possible coupons provided. Each customer received 16 coupons selected based on purchase history (specific selection not identifiable)
- **TypeB & TypeC campaigns**: All customers receive all coupons for that campaign

#### 6. **coupon_redempt** (Coupon Redemption History)
Tracks coupon usage.

| Variable | Description |
|----------|-------------|
| household_key | Unique household identifier |
| day | Redemption date |
| coupon_upc | Unique coupon identifier |
| campaign | Campaign ID |

#### 7. **product** (Product Information)
Product attributes including hierarchy and brand.

| Variable | Description |
|----------|-------------|
| product_id | Unique product identifier |
| department | Product department |
| commodity_desc | Product commodity (mid-level grouping) |
| sub_commodity_desc | Product sub-commodity (lowest-level grouping) |
| manufacturer | Manufacturer code |
| brand | Private or National label |
| curr_size_of_product | Package size (not available for all products) |

#### 8. **causal_data** (Promotional Activities)
Weekly promotional information by product and store. This table tracks marketing interventions independent of coupon campaigns, enabling analysis of display and advertising treatments.

| Variable | Description |
|----------|-------------|
| product_id | Unique product identifier |
| store_id | Store identifier |
| week_no | Week number |
| display | Display location (see codes below) - treatment variable |
| mailer | Mailer placement (see codes below) - treatment variable |

**Note**: This table enables analysis of promotional treatments at the product-store-week level, including their independent effects and interactions with coupon campaigns.

**Display Codes:**
- 0: Not on Display
- 1: Store Front
- 2: Store Rear
- 3: Front End Cap
- 4: Mid-Aisle End Cap
- 5: Rear End Cap
- 6: Side-Aisle End Cap
- 7: In-Aisle
- 9: Secondary Location Display
- A: In-Shelf

**Mailer Codes:**
- 0: Not on ad
- A: Interior page feature
- C: Interior page line item
- D: Front page feature
- F: Back page feature
- H: Wrap front feature
- J: Wrap interior coupon
- L: Wrap back feature
- P: Interior page coupon
- X: Free on interior page
- Z: Free on front page, back page or wrap

---

## Key Insights from User Guide

### Price Calculation Formulas

The `sales_value` field represents the amount the **retailer receives**, not what the customer pays. When a customer uses a manufacturer coupon, the manufacturer reimburses the retailer.

```python
# Loyalty card price (after loyalty discount)
loyalty_price = (sales_value - (retail_disc + coupon_match_disc)) / quantity

# Non-loyalty card price (shelf price)
regular_price = (sales_value - coupon_match_disc) / quantity

# Actual amount customer paid (when using coupon)
customer_paid = sales_value - coupon_disc
```

### Example Calculation

From the user guide (page 3):

| Line | Sales Value | Retail Disc | Coupon Disc | Quantity | Calculation |
|------|-------------|-------------|-------------|----------|-------------|
| 1 | $1.67 | $0 | $0 | 1 | Shelf price = $1.67 |
| 2 | $2.00 | $1.34 | $0 | 2 | Shelf price = ($2 + $1.34)/2 = $1.67<br>Loyalty price = $2/2 = $1.00 |
| 3 | $2.89 | $0 | $0.55 | 2 | Shelf price = ($2.89 + $0.45)/2 = $1.67<br>Customer paid = $2.34<br>Retailer received = $2.89 |

### Case Study: Household 208

The user guide (pages 7-9) provides a detailed walkthrough of household 208's journey:

**Campaign Reception:**
- Received 8 campaigns total
  - 5 TypeA campaigns (personalized)
  - 3 TypeB campaigns (standardized)

**Coupon Activity:**
- Redeemed 7 coupons from 3 campaigns
- Example: Campaign 22 (TypeB)
  - Offered 21 distinct coupons
  - Coupon 51800000050 was valid for 38 different products
  - All products were "refrigerated specialty rolls" from a national brand

**Purchase Behavior:**
- All transactions tracked in `transaction_data`
- Can link coupon redemptions to specific purchases
- Can analyze promotional context (display, mailer) at time of purchase

**Key Insight from Case Study:**
This example demonstrates how to join multiple tables to understand:
- What campaigns a customer received (campaign_table)
- When campaigns were active (campaign_desc)
- What coupons were available (coupon)
- Which coupons were used (coupon_redempt)
- What was purchased (transaction_data)
- What promotions were running (causal_data)

---

[//]: # (## Analytical Workflow)

[//]: # ()
[//]: # (### Recommended Analysis Pipeline)

[//]: # ()
[//]: # (**Phase 1: Data Exploration & Validation**)

[//]: # (- Load all tables and verify structural consistency)

[//]: # (- Examine missing value patterns and data quality)

[//]: # (- Calculate descriptive statistics across key variables)

[//]: # (- Validate referential integrity between tables)

[//]: # ()
[//]: # (**Phase 2: Feature Engineering**)

[//]: # (- Construct customer-level behavioral metrics &#40;e.g., RFM indicators&#41;)

[//]: # (- Derive price sensitivity measures from discount responsiveness)

[//]: # (- Aggregate category-level purchase frequencies and preferences)

[//]: # (- Generate pre-treatment and post-treatment behavioral features)

[//]: # ()
[//]: # (**Phase 3: Causal Analysis**)

[//]: # (- Define treatment and control cohorts based on campaign exposure)

[//]: # (- Establish pre-treatment baseline periods for comparison)

[//]: # (- Implement difference-in-differences or synthetic control methods)

[//]: # (- Estimate heterogeneous treatment effects using meta-learners or causal forests)

[//]: # ()
[//]: # (**Phase 4: Modeling & Evaluation**)

[//]: # (- Develop customer segmentation models with appropriate validation strategies)

[//]: # (- Build uplift models for campaign targeting optimization)

[//]: # (- Construct recommendation systems leveraging collaborative and content-based filtering)

[//]: # (- Evaluate model performance using held-out test sets and appropriate metrics)

[//]: # ()
[//]: # (### Research Questions)

[//]: # ()
[//]: # (The official user guide suggests the following research directions:)

[//]: # ()
[//]: # (1. Characterize customers exhibiting increasing versus decreasing expenditure trends over time)

[//]: # (2. Identify categories driving growth among high-spending customer segments)

[//]: # (3. Analyze category-level disengagement patterns among declining customers)

[//]: # (4. Examine relationships between demographic factors and spending/engagement patterns)

[//]: # (5. Assess evidence for causal impact of direct marketing on customer engagement)

[//]: # (6. Quantify heterogeneity in treatment effects across customer segments &#40;CATE analysis&#41;)

[//]: # (7. **Develop optimal targeting rules based on treatment heterogeneity for maximizing campaign ROI**)

[//]: # (8. **Design personalized promotional strategies combining coupons, displays, and mailer placements**)

---

## Data Summary

| Metric | Value |
|--------|-------|
| Households | 2,500 |
| Transactions | 2,595,732 |
| Baskets | 276,484 |
| Unique Products | ~92,000 |
| Stores | 7 |
| Campaigns | 30 |
| Campaign Types | 3 (TypeA, TypeB, TypeC) |
| Time Span | 102 weeks (≈ 2 years) |
| Coupon Redemptions | ~26,000 |
| Days Covered | 714 days |

---

## References & Resources

- **Source**: [dunnhumby Source Files](https://www.dunnhumby.com/source-files/)
- **Contact**: sourcefiles@dunnhumby.com
- **License**: Academic and research use (attribution required)
- **Documentation**: User guide included with dataset download

---

[//]: # (## Academic & Research Applications)

[//]: # ()
[//]: # (This dataset provides significant value across multiple research domains:)

[//]: # ()
[//]: # (### Causal Inference & Econometrics)

[//]: # ()
[//]: # (- Panel data methodologies including fixed effects and difference-in-differences estimation)

[//]: # (- Heterogeneous treatment effect estimation and characterization)

[//]: # (- Machine learning approaches to causal inference &#40;causal forests, meta-learners, doubly robust estimation&#41;)

[//]: # (- Analysis of treatment timing variations and dynamic treatment effects)

[//]: # (- Investigation of interference and spillover effects in marketing interventions)

[//]: # ()
[//]: # (### Marketing Science & Consumer Behavior)

[//]: # ()
[//]: # (- Quantification of promotional effectiveness and ROI &#40;display placement, mailer advertising, coupon campaigns&#41;)

[//]: # (- Customer lifetime value modeling and prediction)

[//]: # (- Market segmentation strategy development and validation)

[//]: # (- Personalization algorithm development and evaluation)

[//]: # (- Multi-channel attribution modeling)

[//]: # (- In-store merchandising optimization and display effectiveness analysis)

[//]: # (- **Targeting optimization and uplift modeling**: CATE-based customer selection for maximizing campaign ROI)

[//]: # (- **Treatment assignment policies**: Optimal allocation of customers to campaigns based on heterogeneous treatment effects)

[//]: # ()
[//]: # (### Retail Analytics & Operations Research)

[//]: # ()
[//]: # (- Market basket analysis and association rule mining)

[//]: # (- Dynamic pricing and markdown optimization)

[//]: # (- Demand forecasting with promotional effects)

[//]: # (- Assortment optimization and category management)

[//]: # (- Cross-selling and bundling strategy analysis)

[//]: # (---)

## Important Considerations

1. **Initial Exploration**: Begin analysis with a single household (e.g., household 208 documented in the user guide) to understand table relationships and data structure
2. **Price Calculations**: Always apply the documented formulas for price derivation; `sales_value` represents retailer revenue, not customer payment
3. **Treatment Assignment**: Note that TypeA campaigns involve personalized coupon selection (specific assignments unobservable), while TypeB/C campaigns distribute identical coupons to all participants
4. **Temporal Alignment**: Ensure proper alignment of campaign exposure periods with transaction timing for valid causal inference
5. **Statistical Power**: Exercise caution with fine-grained analyses given the limited household sample size of 2,500

---

## Table Relationships

```
hh_demographic ──┐
                 │
                 ├─→ transaction_data ←─→ product
                 │         ↓
campaign_table ──┘         ↓
      ↓              causal_data
campaign_desc            
      ↓
    coupon ←──→ coupon_redempt
```

---

**Disclaimer**: This dataset contains anonymized real customer data provided for academic and research purposes. While the insights derived from this data are valuable for methodological development and theoretical investigation, practitioners should exercise appropriate caution and conduct additional validation before applying findings to operational business decisions. Results should be interpreted within the context of the dataset's specific characteristics and limitations.

[//]: # (---)

[//]: # (*Documentation  Version: Based on dunnhumby Complete Journey User Guide &#40;© 2023&#41;*)