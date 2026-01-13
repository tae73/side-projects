# Positivity Assumption in Causal Inference

## 1. Overview

### Definition
The positivity (overlap) assumption states:
```
0 < P(T=1|X=x) < 1  for all x with positive density
```

Every covariate combination must have a positive probability of receiving both treatment and control.

### Importance
Positivity is required for causal identification because:
- CATE estimation: τ(x) = E[Y|T=1,X=x] - E[Y|T=0,X=x]
- Both conditional expectations require observations in both groups
- Without positivity, estimates become extrapolations, not observations

## 2. Diagnosing Positivity Violations

### Propensity Score Metrics

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| PS AUC | 0.5-0.7 | 0.7-0.9 | > 0.9 |
| Overlap [0.1, 0.9] | > 80% | 50-80% | < 50% |
| SMD (avg) | < 0.1 | 0.1-0.25 | > 0.25 |

### Current Analysis Results
- PS AUC: 0.990 (near-perfect separation)
- Overlap [0.1, 0.9]: ~31%
- Implication: Treatment and control are fundamentally different populations

### What High PS AUC Means
```
PS ≈ 0.99 means:
- Given covariates, we can almost perfectly predict treatment status
- Very few control observations exist where treated obs are common (and vice versa)
- CATE in high-PS regions is purely model extrapolation
```

## 3. Consequences of Violation

### ATE Instability
Different estimators give vastly different results:

| Method | Estimate | Interpretation |
|--------|----------|----------------|
| Naive | +$471 | Selection bias included |
| IPW | +$132 | Extreme weights |
| AIPW | -$176 | Model-dependent |
| OLS | +$65 | Linear extrapolation |
| DML | +$11 | Cross-fitting helps but not immune |

**20x range** in estimates indicates fundamental identification problem.

### CATE Model Disagreement
Models may disagree on treatment effect **direction**:
- CausalForestDML: +$14 (positive)
- LinearDML: -$132 (negative)
- T-Learner: -$199 (negative)

This is not about model quality—it's about identification.

### Anti-Predictive AUUC
Some models have negative AUUC, meaning:
- They rank customers worse than random
- HTE predictions are unreliable
- Cannot trust targeting recommendations

## 4. Solutions

### 4.1 PS Trimming
```python
# Exclude extreme PS values
mask = (ps >= 0.1) & (ps <= 0.9)
data_trimmed = data[mask]
```

**Pros:** Removes extrapolation
**Cons:** Sample loss (may lose 60-70%), local effect only

### 4.2 Overlap Weighting (ATO)
```python
# Weight by propensity overlap
h = ps * (1 - ps)  # Maximum at 0.5
weights = h / ps  # for treated
weights = h / (1 - ps)  # for control
```

**Pros:** Uses all data, down-weights extremes
**Cons:** Estimates Average Treatment on Overlap (ATO), not ATE

### 4.3 Partial Identification Bounds

#### Manski Bounds for ATE
```
ATE ∈ [E[Y|T=1] - Y_max, E[Y|T=1] - Y_min]
```

Without further assumptions, this is the identified set.

#### CATE Bounds
For each x:
```
τ(x) ∈ [μ₁(x) - Y_max, μ₁(x) - Y_min]
```

Bounds are tighter in overlap regions, wider outside.

### 4.4 Sensitivity Analysis
Test stability across:
- Different trimming thresholds
- Different covariate sets
- Different models

## 5. CATE Bounds vs Conformal CATE

| Aspect | Partial ID Bounds | Conformal CATE |
|--------|-------------------|----------------|
| **Measures** | Identification uncertainty | Prediction uncertainty |
| **Source** | Missing counterfactuals | Model/sampling variability |
| **Requires overlap?** | No (bounds widen) | Yes |
| **Positivity violation** | Bounds honest (wide) | Estimates biased |
| **Recommendation** | Primary | Supplementary |

**For our analysis**: Use Partial ID Bounds as primary, Conformal only in overlap region.

## 6. Practical Recommendations

### For Analysis
1. Always report PS AUC and overlap ratio
2. Use trimming AND bounds together
3. Sensitivity analysis is mandatory
4. Explicitly caveat conclusions

### For Business Decisions
1. Focus recommendations on overlap region
2. High-CATE customers outside overlap: uncertain
3. ROI projections should include uncertainty ranges
4. Consider A/B testing to validate CATE predictions

### Reporting Template
```
ATE Estimate: $X (trimmed [0.1, 0.9])
95% CI: [$Y, $Z]
Manski Bounds: [$A, $B]

Caveats:
- PS AUC = 0.99 indicates positivity violation
- Only 31% of sample in overlap region
- Estimates outside overlap are extrapolations
- Results should be validated with randomized testing
```

## 7. References

- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects.
- Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing covariates via propensity score weighting.
- Manski, C. F. (1990). Nonparametric bounds on treatment effects.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research.
