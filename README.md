
# side-projects

A collection of research and experimental projects with a focus on personalization, targeting, and recommendation systems using causal inference and related methodologies.

## Research Interests

### Application Areas

- **Personalization**: Design and implementation of user-tailored experiences
- **Targeting**: Effective targeting strategies and optimization
- **Recommendations**: Recommendation system algorithms and evaluation

### Methodologies

Primarily leveraging the following methodologies:

#### Causal Inference
- **Treatment Heterogeneity**

[//]: # (  - Heterogeneous treatment effects estimation)

[//]: # (  - Conditional average treatment effects &#40;CATE&#41;)

[//]: # (  - Subgroup analysis and optimal treatment regimes)
- **Longitudinal and Panel Data Methods**

[//]: # (  - Time-dependent confounding adjustment)

[//]: # (  - Sequential treatment effect estimation)

[//]: # (  - Dynamic treatment regimes)
- **Unobserved Confounding**

[//]: # (  - Sensitivity analysis frameworks)

[//]: # (  - Instrumental variables and natural experiments)

[//]: # (  - Proxy variables and measurement error)

[//]: # (  - Bounds and partial identification)

[//]: # (- Propensity score methods)

[//]: # (- Difference-in-differences)

[//]: # (- Synthetic control methods)
- ...

#### Adaptive Experiments
- **Bandit Algorithms**

[//]: # (  - Multi-armed and contextual bandits)

[//]: # (  - Thompson sampling and UCB methods)

[//]: # (  - Policy learning and evaluation)
- **Sequential Experimentation**

[//]: # (  - Sequential testing and early stopping)

[//]: # (  - Adaptive sample size determination)

[//]: # (  - Group sequential designs)
- **Adaptive Allocation**

[//]: # (  - Response-adaptive randomization)

[//]: # (  - Optimal treatment assignment)

[//]: # (  - Multi-arm allocation strategies)

[//]: # (#### Graphical Models)

[//]: # (- Bayesian networks)

[//]: # (- Causal graphs &#40;DAGs&#41;)

[//]: # (- Probabilistic graphical models)

[//]: # (- Structural equation modeling)

#### Learning from Incomplete Information
- **Cold-start and few-shot learning**
- **Partial information availability (feature, modality, data ...)**
- **Label noise and uncertainty**
- **Sparse observation settings**
- **Active learning for information acquisition**

## Project Structure

```
.
├── data/                    # Shared datasets organized by source
│   ├── data_1/
│   │   ├── raw/             # Raw data files
│   │   ├── processed/       # Preprocessed data
│   │   ├── eda/             # Common EDA notebooks for this dataset
│   │   └── README.md
│   └── data_2/
│       └── ...
├── models/                  # Reusable model implementations
│   ├── ...
├── projects/                # Individual research projects
│   ├── prj_1/
│   │   ├── experiments/     # Experiment design and analysis
│   │   ├── notebooks/       # Task-specific analysis
│   │   ├── results/         # Experiment results
│   │   └── README.md
│   ├── ...
└── utils/                   # Common utility functions
```

Each dataset under data/ contains baseline EDA, while project-specific notebooks focus on task-relevant analysis.

[//]: # (## Tech Stack)

[//]: # ()
[//]: # (- **Languages**: Python, R)

[//]: # (- **Causal Inference**: DoWhy, EconML, CausalML, CausalImpact)

[//]: # (- **ML/Stats**: scikit-learn, PyTorch, statsmodels, lifelines)

[//]: # (- **Bayesian Methods**: PyMC, Stan, pgmpy)

[//]: # (- **Experimentation**: scipy, statsmodels, bandit libraries)

[//]: # (- **Visualization**: matplotlib, seaborn, plotly, networkx)

[//]: # (## Key Topics)

[//]: # ()
[//]: # (### Causal Inference in Practice)

[//]: # (- Heterogeneous treatment effects and personalization)

[//]: # (- Handling time-varying confounding in longitudinal data)

[//]: # (- Sensitivity analysis for unobserved confounding)

[//]: # (- Causal discovery and structure learning)

[//]: # ()
[//]: # (### Experimentation & Learning)

[//]: # (- Uplift modeling and targeting optimization)

[//]: # (- Online learning and bandit algorithms)

[//]: # (- Adaptive treatment assignment)

[//]: # (- Sequential decision making under uncertainty)

[//]: # ()
[//]: # (### Recommendation Systems)

[//]: # (- Causal recommendation frameworks)

[//]: # (- Debiasing and counterfactual evaluation)

[//]: # (- Context-aware personalization)

[//]: # (- Exploration-exploitation tradeoffs)

## Notes

Each dataset under data/ contains baseline exploratory analysis, while project notebooks focus on task-specific insights. This structure enables efficient reuse when building progressive projects (e.g., segmentation → CATE targeting → recommendations) on shared datasets.


[//]: # (## Contact)

[//]: # ()
[//]: # (Feel free to reach out for questions or collaboration proposals.)

---

**Last Updated**: November 2025