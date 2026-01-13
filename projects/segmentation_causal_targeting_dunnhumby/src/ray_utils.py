"""Ray utilities for parallel causal inference.

This module provides Ray-based parallelization for:
- CATE model training (fit multiple models in parallel)
- Covariate experiments (run different covariate sets in parallel)
- Bootstrap confidence intervals
- Cross-validation

Requires: pip install ray

Usage:
    from src.ray_utils import init_ray, fit_cate_models_parallel

    init_ray()
    results = fit_cate_models_parallel(Y, T, X)
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Conditional Ray import
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not installed. Parallel functions will fall back to sequential execution.")


# =============================================================================
# Ray Initialization
# =============================================================================

def init_ray(
    num_cpus: Optional[int] = None,
    memory: Optional[int] = None,
    log_level: int = logging.WARNING
) -> bool:
    """Initialize Ray cluster.

    Args:
        num_cpus: Number of CPUs to use (default: all available)
        memory: Memory limit in bytes (default: system memory)
        log_level: Logging level for Ray

    Returns:
        True if Ray initialized successfully, False otherwise
    """
    if not RAY_AVAILABLE:
        logging.warning("Ray not available. Using sequential execution.")
        return False

    if ray.is_initialized():
        return True

    try:
        ray.init(
            num_cpus=num_cpus or os.cpu_count(),
            ignore_reinit_error=True,
            logging_level=log_level,
            log_to_driver=False
        )
        logging.info(f"Ray initialized with {ray.available_resources().get('CPU', 'N/A')} CPUs")
        return True
    except Exception as e:
        logging.warning(f"Failed to initialize Ray: {e}")
        return False


def shutdown_ray():
    """Shutdown Ray cluster."""
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


def is_ray_available() -> bool:
    """Check if Ray is available and initialized."""
    return RAY_AVAILABLE and ray.is_initialized()


# =============================================================================
# CATE Model Configurations
# =============================================================================

def get_cate_model_configs(seed: int = 42) -> Dict[str, Dict]:
    """Get default configurations for CATE models.

    All models use econml implementations.
    Parameters aligned with backup notebook (03_hte_analysis_backup.ipynb):
    - model_t: LogisticRegression (better for binary treatment)
    - max_depth: 4 (more flexibility)
    - CausalForestDML: n_estimators=200, cv=5

    Args:
        seed: Random state for reproducibility

    Returns:
        Dict mapping model name to configuration
    """
    # Base regressor for outcome model (max_depth=4 as in backup)
    base_regressor = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, min_samples_leaf=20,
        random_state=seed
    )

    # LogisticRegression for propensity model (as in backup notebook)
    # Better calibrated probabilities for binary treatment
    base_propensity = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)

    return {
        's_learner': {
            'type': 's_learner',
            'overall_model': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=20,
                random_state=seed
            )
        },
        't_learner': {
            'type': 't_learner',
            'models': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=20,
                random_state=seed
            )
        },
        'x_learner': {
            'type': 'x_learner',
            'models': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=20,
                random_state=seed
            ),
            'propensity_model': LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            'cate_models': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=20,
                random_state=seed + 1
            )
        },
        'linear_dml': {
            'type': 'linear_dml',
            'model_y': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=20,
                random_state=seed
            ),
            'model_t': LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            'cv': 5,
            'use_ray': True  # Enable Ray for internal CV
        },
        'causal_forest_dml': {
            'type': 'causal_forest_dml',
            'model_y': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=20,
                random_state=seed
            ),
            'model_t': LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            'n_estimators': 200,  # More trees for better stability
            'min_samples_leaf': 20,
            'cv': 5,
            'use_ray': True  # Enable Ray for internal CV
        },
        'grf': {
            'type': 'grf',
            'n_estimators': 500
        }
    }


# =============================================================================
# Sequential Fallbacks
# =============================================================================

def _fit_s_learner(Y, T, X, config):
    """Fit S-Learner using econml."""
    try:
        from econml.metalearners import SLearner

        model = SLearner(overall_model=config['overall_model'])
        model.fit(Y, T, X=X)
        cate = model.effect(X).flatten()

        return {'model': model, 'cate': cate}
    except ImportError:
        logging.warning("econml not installed. Skipping SLearner.")
        return {'model': None, 'cate': np.zeros(len(Y))}


def _fit_t_learner(Y, T, X, config):
    """Fit T-Learner using econml."""
    try:
        from econml.metalearners import TLearner

        model = TLearner(models=config['models'])
        model.fit(Y, T, X=X)
        cate = model.effect(X).flatten()

        return {'model': model, 'cate': cate}
    except ImportError:
        logging.warning("econml not installed. Skipping TLearner.")
        return {'model': None, 'cate': np.zeros(len(Y))}


def _fit_x_learner(Y, T, X, config):
    """Fit X-Learner using econml."""
    try:
        from econml.metalearners import XLearner

        model = XLearner(
            models=config['models'],
            propensity_model=config.get('propensity_model'),
            cate_models=config.get('cate_models')
        )
        model.fit(Y, T, X=X)
        cate = model.effect(X).flatten()

        return {'model': model, 'cate': cate}
    except ImportError:
        logging.warning("econml not installed. Skipping XLearner.")
        return {'model': None, 'cate': np.zeros(len(Y))}


def _fit_linear_dml(Y, T, X, config):
    """Fit LinearDML from econml with optional Ray parallelization."""
    try:
        from econml.dml import LinearDML

        dml = LinearDML(
            model_y=config['model_y'],
            model_t=config['model_t'],
            discrete_treatment=True,  # Binary treatment
            cv=config.get('cv', 5),
            use_ray=config.get('use_ray', False),  # Enable Ray for internal CV
            random_state=42
        )
        dml.fit(Y, T, X=X)
        cate = dml.effect(X).flatten()

        return {'model': dml, 'cate': cate}
    except ImportError:
        logging.warning("econml not installed. Skipping LinearDML.")
        return {'model': None, 'cate': np.zeros(len(Y))}


def _fit_causal_forest_dml(Y, T, X, config):
    """Fit CausalForestDML from econml with optional Ray parallelization."""
    try:
        from econml.dml import CausalForestDML

        cf = CausalForestDML(
            model_y=config['model_y'],
            model_t=config['model_t'],
            discrete_treatment=True,  # Binary treatment
            n_estimators=config.get('n_estimators', 200),
            min_samples_leaf=config.get('min_samples_leaf', 20),
            cv=config.get('cv', 5),
            use_ray=config.get('use_ray', False),  # Enable Ray for internal CV
            random_state=42
        )
        cf.fit(Y, T, X=X)
        cate = cf.effect(X).flatten()

        return {'model': cf, 'cate': cate}
    except ImportError:
        logging.warning("econml not installed. Skipping CausalForestDML.")
        return {'model': None, 'cate': np.zeros(len(Y))}


def _fit_grf(Y, T, X, config):
    """Fit Generalized Random Forest (econml wrapper)."""
    try:
        from econml.grf import CausalForest

        grf = CausalForest(
            n_estimators=config.get('n_estimators', 500),
            random_state=42
        )
        grf.fit(X, T, Y)
        cate = grf.effect(X)

        return {'model': grf, 'cate': cate}
    except ImportError:
        logging.warning("econml not installed. Skipping GRF.")
        return {'model': None, 'cate': np.zeros(len(Y))}


def _fit_single_model(model_name: str, Y: np.ndarray, T: np.ndarray,
                      X: np.ndarray, config: Dict) -> Tuple[str, Dict]:
    """Fit a single CATE model."""
    model_type = config['type']

    if model_type == 's_learner':
        result = _fit_s_learner(Y, T, X, config)
    elif model_type == 't_learner':
        result = _fit_t_learner(Y, T, X, config)
    elif model_type == 'x_learner':
        result = _fit_x_learner(Y, T, X, config)
    elif model_type == 'linear_dml':
        result = _fit_linear_dml(Y, T, X, config)
    elif model_type == 'causal_forest_dml':
        result = _fit_causal_forest_dml(Y, T, X, config)
    elif model_type == 'grf':
        result = _fit_grf(Y, T, X, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_name, result


def fit_cate_models_sequential(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    models: Optional[List[str]] = None,
    configs: Optional[Dict[str, Dict]] = None
) -> Dict[str, Dict]:
    """Fit CATE models sequentially (fallback when Ray unavailable).

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        models: List of model names to fit (default: all)
        configs: Model configurations (default: get_cate_model_configs())

    Returns:
        Dict mapping model name to {model, cate}
    """
    if configs is None:
        configs = get_cate_model_configs()

    if models is None:
        models = list(configs.keys())

    results = {}
    for model_name in models:
        if model_name not in configs:
            logging.warning(f"Unknown model: {model_name}")
            continue

        try:
            name, result = _fit_single_model(model_name, Y, T, X, configs[model_name])
            results[name] = result
        except Exception as e:
            logging.warning(f"Failed to fit {model_name}: {e}")
            results[model_name] = {'model': None, 'cate': np.zeros(len(Y))}

    return results


# =============================================================================
# Ray Parallel Functions
# =============================================================================

if RAY_AVAILABLE:
    @ray.remote
    def _fit_cate_model_remote(model_name: str, Y: np.ndarray, T: np.ndarray,
                                X: np.ndarray, config: Dict) -> Tuple[str, Dict]:
        """Ray remote function for fitting CATE model."""
        return _fit_single_model(model_name, Y, T, X, config)

    @ray.remote
    def _run_covariate_experiment_remote(
        var_set_name: str,
        var_cols: List[str],
        Y: np.ndarray,
        T: np.ndarray,
        X_full: pd.DataFrame
    ) -> Dict:
        """Ray remote function for covariate experiment."""
        from .treatment_effects import (
            estimate_propensity_score, compute_positivity_diagnostics,
            estimate_ate_dml
        )

        available_cols = [c for c in var_cols if c in X_full.columns]
        if len(available_cols) == 0:
            return None

        X_subset = X_full[available_cols].values

        # Estimate PS
        ps = estimate_propensity_score(X_subset, T)
        ps_diag = compute_positivity_diagnostics(ps, T)

        # Estimate ATE
        ate_result = estimate_ate_dml(Y, T, X_subset)

        return {
            'var_set': var_set_name,
            'n_vars': len(available_cols),
            'ps_auc': ps_diag.ps_auc,
            'overlap_ratio': ps_diag.overlap_ratio,
            'ate': ate_result.estimate,
            'ate_se': ate_result.se,
            'ate_ci_lower': ate_result.ci_lower,
            'ate_ci_upper': ate_result.ci_upper
        }

    @ray.remote
    def _bootstrap_iteration_remote(
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        estimator_fn: Callable
    ) -> float:
        """Single bootstrap iteration."""
        np.random.seed(seed)
        n = len(Y)
        idx = np.random.choice(n, n, replace=True)
        return estimator_fn(Y[idx], T[idx], X[idx])


def fit_cate_models_parallel(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    models: Optional[List[str]] = None,
    configs: Optional[Dict[str, Dict]] = None
) -> Dict[str, Dict]:
    """Fit CATE models in parallel using Ray.

    Falls back to sequential if Ray not available.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        models: List of model names
        configs: Model configurations

    Returns:
        Dict mapping model name to {model, cate}
    """
    if not is_ray_available():
        logging.info("Ray not available. Using sequential execution.")
        return fit_cate_models_sequential(Y, T, X, models, configs)

    if configs is None:
        configs = get_cate_model_configs()

    if models is None:
        models = list(configs.keys())

    # Put data in Ray object store
    Y_ref = ray.put(Y)
    T_ref = ray.put(T)
    X_ref = ray.put(X)

    # Launch parallel tasks
    futures = []
    for model_name in models:
        if model_name not in configs:
            continue
        futures.append(
            _fit_cate_model_remote.remote(model_name, Y_ref, T_ref, X_ref, configs[model_name])
        )

    # Gather results
    results = {}
    for result in ray.get(futures):
        if result is not None:
            name, model_result = result
            results[name] = model_result

    return results


def run_covariate_experiments_parallel(
    Y: np.ndarray,
    T: np.ndarray,
    X_full: pd.DataFrame,
    var_sets: Dict[str, List[str]]
) -> pd.DataFrame:
    """Run covariate experiments in parallel using Ray.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X_full: Full covariate DataFrame
        var_sets: Dict mapping set name to column names

    Returns:
        DataFrame with experiment results
    """
    if not is_ray_available():
        from .treatment_effects import run_covariate_experiment
        return run_covariate_experiment(Y, T, X_full, var_sets)

    # Put data in object store
    Y_ref = ray.put(Y)
    T_ref = ray.put(T)
    X_ref = ray.put(X_full)

    # Launch parallel tasks
    futures = [
        _run_covariate_experiment_remote.remote(name, cols, Y_ref, T_ref, X_ref)
        for name, cols in var_sets.items()
    ]

    # Gather results
    results = [r for r in ray.get(futures) if r is not None]
    return pd.DataFrame(results)


def bootstrap_ate_parallel(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    estimator_fn: Callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Dict[str, float]:
    """Parallel bootstrap for ATE confidence interval.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        estimator_fn: Function that takes (Y, T, X) and returns ATE estimate
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level for CI

    Returns:
        Dict with mean, ci_lower, ci_upper, se
    """
    if not is_ray_available():
        # Sequential fallback
        ate_samples = []
        for i in range(n_bootstrap):
            np.random.seed(i)
            idx = np.random.choice(len(Y), len(Y), replace=True)
            ate_samples.append(estimator_fn(Y[idx], T[idx], X[idx]))

        return {
            'mean': np.mean(ate_samples),
            'se': np.std(ate_samples),
            'ci_lower': np.percentile(ate_samples, alpha / 2 * 100),
            'ci_upper': np.percentile(ate_samples, (1 - alpha / 2) * 100)
        }

    # Put data in object store
    Y_ref = ray.put(Y)
    T_ref = ray.put(T)
    X_ref = ray.put(X)
    fn_ref = ray.put(estimator_fn)

    # Launch parallel tasks
    futures = [
        _bootstrap_iteration_remote.remote(Y_ref, T_ref, X_ref, i, fn_ref)
        for i in range(n_bootstrap)
    ]

    ate_samples = ray.get(futures)

    return {
        'mean': np.mean(ate_samples),
        'se': np.std(ate_samples),
        'ci_lower': np.percentile(ate_samples, alpha / 2 * 100),
        'ci_upper': np.percentile(ate_samples, (1 - alpha / 2) * 100)
    }


# =============================================================================
# econml Ray Integration
# =============================================================================

def create_ray_enabled_dml(
    model_y: Optional[object] = None,
    model_t: Optional[object] = None,
    cv: int = 5,
    n_estimators: int = 100,
    use_ray: bool = True
) -> object:
    """Create a CausalForestDML with Ray backend for tree fitting.

    Note: econml's CausalForestDML uses joblib for parallelization.
    Setting n_jobs=-1 uses all cores, which is similar to Ray.

    Args:
        model_y: Outcome model (default: GradientBoostingRegressor)
        model_t: Treatment model (default: GradientBoostingClassifier)
        cv: Cross-validation folds
        n_estimators: Number of trees
        use_ray: If True, use Ray for parallelization (via joblib)

    Returns:
        Configured CausalForestDML model
    """
    try:
        from econml.dml import CausalForestDML

        if model_y is None:
            model_y = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, min_samples_leaf=20,
                learning_rate=0.05, random_state=42
            )

        if model_t is None:
            model_t = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, min_samples_leaf=20,
                learning_rate=0.05, random_state=42
            )

        # Use all cores for tree fitting
        cf = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,  # Binary treatment
            n_estimators=n_estimators,
            cv=cv,
            n_jobs=-1 if use_ray else 1,
            random_state=42
        )

        return cf
    except ImportError:
        logging.warning("econml not installed.")
        return None


# =============================================================================
# Hyperparameter Tuning with RScorer
# =============================================================================

def get_param_grid(size: str = 'medium') -> Dict[str, List[Dict]]:
    """Get parameter grids for CATE model tuning.

    Args:
        size: Grid size - 'small' (4 configs), 'medium' (12 configs), 'large' (24+ configs)

    Returns:
        Dict mapping model name to list of parameter configs
    """
    if size == 'small':
        return {
            's_learner': [
                {'max_depth': 3, 'n_estimators': 100},
                {'max_depth': 4, 'n_estimators': 100},
                {'max_depth': 5, 'n_estimators': 100},
                {'max_depth': 4, 'n_estimators': 200},
            ],
            't_learner': [
                {'max_depth': 3, 'n_estimators': 100},
                {'max_depth': 4, 'n_estimators': 100},
                {'max_depth': 5, 'n_estimators': 100},
                {'max_depth': 4, 'n_estimators': 200},
            ],
            'x_learner': [
                {'max_depth': 3, 'n_estimators': 100},
                {'max_depth': 4, 'n_estimators': 100},
                {'max_depth': 5, 'n_estimators': 100},
                {'max_depth': 4, 'n_estimators': 200},
            ],
            'linear_dml': [
                {'max_depth': 3, 'C': 0.1},
                {'max_depth': 4, 'C': 1.0},
                {'max_depth': 5, 'C': 1.0},
                {'max_depth': 4, 'C': 10.0},
            ],
            'causal_forest_dml': [
                {'max_depth': 4, 'n_estimators': 100, 'cf_n_estimators': 100},
                {'max_depth': 4, 'n_estimators': 100, 'cf_n_estimators': 200},
                {'max_depth': 5, 'n_estimators': 100, 'cf_n_estimators': 200},
                {'max_depth': 4, 'n_estimators': 200, 'cf_n_estimators': 300},
            ],
        }

    elif size == 'large':
        # Full grid search
        max_depths = [2, 3, 4, 5, 6]
        n_estimators_list = [50, 100, 150, 200, 300]
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        cf_n_estimators_list = [100, 200, 300, 500]
        min_samples_leaf_list = [10, 20, 30, 50]

        return {
            's_learner': [
                {'max_depth': d, 'n_estimators': n, 'min_samples_leaf': m}
                for d in max_depths for n in n_estimators_list for m in min_samples_leaf_list
            ],
            't_learner': [
                {'max_depth': d, 'n_estimators': n, 'min_samples_leaf': m}
                for d in max_depths for n in n_estimators_list for m in min_samples_leaf_list
            ],
            'x_learner': [
                {'max_depth': d, 'n_estimators': n, 'C': c}
                for d in max_depths for n in n_estimators_list for c in C_values
            ],
            'linear_dml': [
                {'max_depth': d, 'n_estimators': n, 'C': c}
                for d in max_depths for n in n_estimators_list for c in C_values
            ],
            'causal_forest_dml': [
                {'max_depth': d, 'n_estimators': n, 'cf_n_estimators': cf, 'min_samples_leaf': m}
                for d in max_depths for n in [100, 200] for cf in cf_n_estimators_list for m in [10, 20, 30]
            ],
        }

    else:  # medium (default)
        max_depths = [3, 4, 5, 6]
        n_estimators_list = [100, 150, 200]
        C_values = [0.1, 1.0, 10.0]

        return {
            's_learner': [
                {'max_depth': d, 'n_estimators': n}
                for d in max_depths for n in n_estimators_list
            ],
            't_learner': [
                {'max_depth': d, 'n_estimators': n}
                for d in max_depths for n in n_estimators_list
            ],
            'x_learner': [
                {'max_depth': d, 'n_estimators': n, 'C': c}
                for d in max_depths for n in [100, 200] for c in C_values
            ],
            'linear_dml': [
                {'max_depth': d, 'n_estimators': n, 'C': c}
                for d in max_depths for n in [100, 200] for c in C_values
            ],
            'causal_forest_dml': [
                {'max_depth': d, 'n_estimators': n, 'cf_n_estimators': cf}
                for d in max_depths for n in [100, 200] for cf in [100, 200, 300]
            ],
        }


def _create_model_from_params(model_type: str, params: Dict, seed: int = 42):
    """Create a CATE model from parameters.

    Args:
        model_type: One of 's_learner', 't_learner', 'x_learner', 'linear_dml', 'causal_forest_dml'
        params: Parameter dictionary
        seed: Random state

    Returns:
        Configured CATE model
    """
    try:
        from econml.metalearners import SLearner, TLearner, XLearner
        from econml.dml import LinearDML, CausalForestDML

        max_depth = params.get('max_depth', 4)
        n_estimators = params.get('n_estimators', 100)
        min_samples_leaf = params.get('min_samples_leaf', 20)

        base_regressor = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=seed
        )

        if model_type == 's_learner':
            return SLearner(overall_model=base_regressor)

        elif model_type == 't_learner':
            return TLearner(models=base_regressor)

        elif model_type == 'x_learner':
            propensity_model = LogisticRegression(
                C=params.get('C', 1.0), max_iter=1000, random_state=seed
            )
            return XLearner(
                models=base_regressor,
                propensity_model=propensity_model,
                cate_models=GradientBoostingRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf, random_state=seed + 1
                )
            )

        elif model_type == 'linear_dml':
            return LinearDML(
                model_y=base_regressor,
                model_t=LogisticRegression(
                    C=params.get('C', 1.0), max_iter=1000, random_state=seed
                ),
                discrete_treatment=True,
                cv=5,
                random_state=seed
            )

        elif model_type == 'causal_forest_dml':
            cf_n_estimators = params.get('cf_n_estimators', 200)
            cf_min_samples_leaf = params.get('min_samples_leaf', 20)
            return CausalForestDML(
                model_y=base_regressor,
                model_t=LogisticRegression(
                    C=params.get('C', 1.0), max_iter=1000, random_state=seed
                ),
                discrete_treatment=True,
                n_estimators=cf_n_estimators,
                min_samples_leaf=cf_min_samples_leaf,
                cv=5,
                random_state=seed
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    except ImportError as e:
        logging.warning(f"econml not installed: {e}")
        return None


def tune_cate_model_rscorer(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    model_type: str,
    param_grid: Optional[List[Dict]] = None,
    cv: int = 3,
    seed: int = 42
) -> Dict[str, Any]:
    """Tune CATE model using RScorer cross-validation.

    RScorer uses R-loss (Robinson's residual-on-residual loss) which is
    appropriate for CATE model selection since true CATE is unknown.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        model_type: Type of CATE model
        param_grid: List of parameter configurations to try
        cv: Number of cross-validation folds
        seed: Random state

    Returns:
        Dict with best_model, best_params, best_score, all_scores
    """
    try:
        from econml.score import RScorer
        from sklearn.model_selection import KFold
    except ImportError:
        logging.warning("econml or sklearn not available for tuning")
        return {'best_model': None, 'best_params': None, 'best_score': None}

    if param_grid is None:
        param_grid = get_param_grid().get(model_type, [{}])

    # Create RScorer
    # RScorer needs nuisance models for Y and T
    scorer = RScorer(
        model_y=GradientBoostingRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=seed
        ),
        model_t=LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
        discrete_treatment=True,
        cv=cv,
        random_state=seed
    )

    # Fit scorer (learns nuisance functions)
    scorer.fit(Y, T, X=X)

    # Evaluate each parameter configuration
    all_scores = []
    best_score = -np.inf
    best_params = None
    best_model = None

    for params in param_grid:
        try:
            model = _create_model_from_params(model_type, params, seed)
            if model is None:
                continue

            # Fit model
            model.fit(Y, T, X=X)

            # Score using RScorer
            # RScorer.score() returns R-squared based on R-loss
            score = scorer.score(model)

            all_scores.append({
                'params': params,
                'score': score
            })

            if score > best_score:
                best_score = score
                best_params = params
                best_model = model

            logging.info(f"{model_type} params={params}: R-score={score:.4f}")

        except Exception as e:
            logging.warning(f"Failed to evaluate {model_type} with {params}: {e}")
            all_scores.append({'params': params, 'score': np.nan, 'error': str(e)})

    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_score': best_score,
        'all_scores': all_scores
    }


def tune_cate_models_parallel(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    model_types: Optional[List[str]] = None,
    param_grids: Optional[Dict[str, List[Dict]]] = None,
    cv: int = 3,
    seed: int = 42
) -> Dict[str, Dict]:
    """Tune multiple CATE models in parallel using RScorer.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        model_types: List of model types to tune (default: all)
        param_grids: Dict of parameter grids per model type
        cv: Cross-validation folds for RScorer
        seed: Random state

    Returns:
        Dict mapping model type to tuning results
    """
    if model_types is None:
        model_types = ['s_learner', 't_learner', 'x_learner', 'linear_dml', 'causal_forest_dml']

    if param_grids is None:
        param_grids = get_param_grid()

    if not is_ray_available():
        # Sequential fallback
        results = {}
        for model_type in model_types:
            logging.info(f"Tuning {model_type}...")
            results[model_type] = tune_cate_model_rscorer(
                Y, T, X, model_type,
                param_grid=param_grids.get(model_type),
                cv=cv, seed=seed
            )
        return results

    # Parallel tuning with Ray
    @ray.remote
    def _tune_remote(model_type, Y, T, X, param_grid, cv, seed):
        return model_type, tune_cate_model_rscorer(Y, T, X, model_type, param_grid, cv, seed)

    # Put data in object store
    Y_ref = ray.put(Y)
    T_ref = ray.put(T)
    X_ref = ray.put(X)

    # Launch parallel tuning
    futures = [
        _tune_remote.remote(
            model_type, Y_ref, T_ref, X_ref,
            param_grids.get(model_type), cv, seed
        )
        for model_type in model_types
    ]

    # Collect results
    results = {}
    for result in ray.get(futures):
        model_type, tuning_result = result
        results[model_type] = tuning_result

    return results


def get_best_tuned_configs(tuning_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Convert tuning results to model configs for fit_cate_models_parallel.

    Args:
        tuning_results: Output from tune_cate_models_parallel

    Returns:
        Dict of configs compatible with get_cate_model_configs format
    """
    configs = {}

    for model_type, result in tuning_results.items():
        if result['best_model'] is None:
            continue

        best_params = result['best_params'] or {}
        max_depth = best_params.get('max_depth', 4)
        n_estimators = best_params.get('n_estimators', 100)
        min_samples_leaf = best_params.get('min_samples_leaf', 20)
        C = best_params.get('C', 1.0)

        base_regressor = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=42
        )

        if model_type == 's_learner':
            configs[model_type] = {
                'type': 's_learner',
                'overall_model': base_regressor
            }

        elif model_type == 't_learner':
            configs[model_type] = {
                'type': 't_learner',
                'models': base_regressor
            }

        elif model_type == 'x_learner':
            configs[model_type] = {
                'type': 'x_learner',
                'models': base_regressor,
                'propensity_model': LogisticRegression(C=C, max_iter=1000, random_state=42),
                'cate_models': GradientBoostingRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf, random_state=43
                )
            }

        elif model_type == 'linear_dml':
            configs[model_type] = {
                'type': 'linear_dml',
                'model_y': base_regressor,
                'model_t': LogisticRegression(C=C, max_iter=1000, random_state=42),
                'cv': 5,
                'use_ray': True
            }

        elif model_type == 'causal_forest_dml':
            cf_n_estimators = best_params.get('cf_n_estimators', 200)
            configs[model_type] = {
                'type': 'causal_forest_dml',
                'model_y': base_regressor,
                'model_t': LogisticRegression(C=C, max_iter=1000, random_state=42),
                'n_estimators': cf_n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'cv': 5,
                'use_ray': True
            }

    return configs


# =============================================================================
# Optuna-based Hyperparameter Tuning
# =============================================================================

# Conditional imports for Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not installed. Use: pip install optuna")


def get_param_space_optuna(model_type: str) -> Dict[str, Any]:
    """Get expanded Optuna search space for each model type.

    Returns dict with tunable parameter ranges for Optuna.

    Args:
        model_type: One of 's_learner', 't_learner', 'x_learner', 'linear_dml', 'causal_forest_dml'

    Returns:
        Dict mapping parameter names to (low, high, type) tuples or list of choices
    """
    # Expanded GBM parameters
    gbm_space = {
        'max_depth': (2, 15, 'int'),              # expanded: 10→15
        'n_estimators': (50, 800, 'int'),         # expanded: 500→800
        'min_samples_leaf': (3, 150, 'int'),      # expanded: 5-100→3-150
        'min_samples_split': (2, 50, 'int'),      # NEW
        'learning_rate': (0.005, 0.5, 'log'),
        'subsample': (0.5, 1.0, 'float'),
        'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
        'ccp_alpha': (0.0, 0.1, 'float'),         # NEW - complexity pruning
    }

    if model_type == 's_learner':
        return gbm_space

    elif model_type == 't_learner':
        return gbm_space

    elif model_type == 'x_learner':
        return {
            **gbm_space,
            'propensity_C': (0.001, 100.0, 'log'),
        }

    elif model_type == 'linear_dml':
        return {
            'model_y_max_depth': (2, 10, 'int'),          # expanded: 8→10
            'model_y_n_estimators': (50, 500, 'int'),     # expanded: 300→500
            'model_y_learning_rate': (0.01, 0.3, 'log'),
            'model_y_min_samples_leaf': (5, 50, 'int'),   # NEW
            'model_y_subsample': (0.5, 1.0, 'float'),     # NEW
            'model_t_C': (0.001, 100.0, 'log'),
            'cv': [3, 5],
        }

    elif model_type == 'causal_forest_dml':
        return {
            # First stage nuisance models
            'model_y_max_depth': (2, 10, 'int'),          # expanded
            'model_y_n_estimators': (50, 500, 'int'),     # expanded
            'model_y_learning_rate': (0.01, 0.3, 'log'),
            'model_t_C': (0.001, 100.0, 'log'),
            # Causal forest params
            'n_estimators': (100, 2000, 'int'),           # expanded: 1000→2000
            'max_depth': (4, 30, 'int'),                  # expanded: 20→30
            'min_samples_leaf': (3, 150, 'int'),          # expanded
            'min_impurity_decrease': (1e-7, 1e-2, 'log'), # expanded lower bound
            'max_samples': (0.3, 1.0, 'float'),
            'min_balancedness_tol': (0.3, 0.5, 'float'),  # NEW
            'honest': [True, False],                       # NEW
            'subforest_size': (2, 8, 'int'),              # NEW
            'cv': [3, 5],
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _sample_params_optuna(trial: 'optuna.Trial', param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample parameters from Optuna trial based on param_space definition."""
    params = {}
    for name, spec in param_space.items():
        if isinstance(spec, list):
            # Choice parameter
            params[name] = trial.suggest_categorical(name, spec)
        elif isinstance(spec, tuple) and len(spec) == 3:
            low, high, dtype = spec
            if dtype == 'int':
                params[name] = trial.suggest_int(name, low, high)
            elif dtype == 'float':
                params[name] = trial.suggest_float(name, low, high)
            elif dtype == 'log':
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                raise ValueError(f"Unknown dtype: {dtype}")
        else:
            raise ValueError(f"Invalid param spec for {name}: {spec}")
    return params


def _create_model_from_params_optuna(
    model_type: str,
    params: Dict[str, Any],
    seed: int = 42
) -> Any:
    """Create CATE model from Optuna-sampled parameters.

    Args:
        model_type: Model type
        params: Sampled parameters from Optuna
        seed: Random seed

    Returns:
        Configured CATE model (econml estimator)
    """
    from econml.metalearners import SLearner, TLearner, XLearner
    from econml.dml import LinearDML, CausalForestDML

    if model_type in ['s_learner', 't_learner', 'x_learner']:
        # Extract GBM params (expanded)
        gbm_params = {
            'max_depth': params.get('max_depth', 4),
            'n_estimators': params.get('n_estimators', 100),
            'min_samples_leaf': params.get('min_samples_leaf', 20),
            'min_samples_split': params.get('min_samples_split', 2),  # NEW
            'learning_rate': params.get('learning_rate', 0.1),
            'subsample': params.get('subsample', 1.0),
            'ccp_alpha': params.get('ccp_alpha', 0.0),  # NEW
            'random_state': seed,
        }

        # Handle max_features
        max_features = params.get('max_features', 'sqrt')
        if isinstance(max_features, str):
            gbm_params['max_features'] = max_features
        else:
            gbm_params['max_features'] = max_features

        base_regressor = GradientBoostingRegressor(**gbm_params)

        if model_type == 's_learner':
            return SLearner(overall_model=base_regressor)

        elif model_type == 't_learner':
            return TLearner(models=base_regressor)

        elif model_type == 'x_learner':
            propensity_C = params.get('propensity_C', 1.0)
            return XLearner(
                models=base_regressor,
                propensity_model=LogisticRegression(C=propensity_C, max_iter=1000, random_state=seed),
                cate_models=GradientBoostingRegressor(**{**gbm_params, 'random_state': seed + 1})
            )

    elif model_type == 'linear_dml':
        model_y = GradientBoostingRegressor(
            max_depth=params.get('model_y_max_depth', 4),
            n_estimators=params.get('model_y_n_estimators', 100),
            learning_rate=params.get('model_y_learning_rate', 0.1),
            min_samples_leaf=params.get('model_y_min_samples_leaf', 20),  # NEW
            subsample=params.get('model_y_subsample', 1.0),  # NEW
            random_state=seed
        )
        model_t = LogisticRegression(
            C=params.get('model_t_C', 1.0),
            max_iter=1000,
            random_state=seed
        )
        return LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            cv=params.get('cv', 3),
            random_state=seed
        )

    elif model_type == 'causal_forest_dml':
        model_y = GradientBoostingRegressor(
            max_depth=params.get('model_y_max_depth', 4),
            n_estimators=params.get('model_y_n_estimators', 100),
            learning_rate=params.get('model_y_learning_rate', 0.1),
            random_state=seed
        )
        model_t = LogisticRegression(
            C=params.get('model_t_C', 1.0),
            max_iter=1000,
            random_state=seed
        )
        return CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', None),
            min_samples_leaf=params.get('min_samples_leaf', 10),
            min_impurity_decrease=params.get('min_impurity_decrease', 0.0),
            max_samples=params.get('max_samples', 0.5),
            min_balancedness_tol=params.get('min_balancedness_tol', 0.45),  # NEW
            honest=params.get('honest', True),  # NEW
            subforest_size=params.get('subforest_size', 4),  # NEW
            cv=params.get('cv', 3),
            random_state=seed
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def tune_cate_model_optuna(
    model_type: str,
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 3,
    param_space: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    verbose: bool = False,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Tune single CATE model using Optuna (TPE) with RScorer evaluation.

    Uses econml's RScorer (R-loss) for CATE model evaluation.

    Args:
        model_type: One of 's_learner', 't_learner', 'x_learner', 'linear_dml', 'causal_forest_dml'
        Y: Outcome variable
        T: Treatment indicator
        X: Covariates
        n_trials: Number of Optuna trials
        cv_folds: Number of CV folds for RScorer
        param_space: Custom parameter space (default: get_param_space_optuna)
        seed: Random seed
        verbose: Whether to show Optuna progress
        timeout: Optional timeout in seconds

    Returns:
        dict with 'best_params', 'best_score', 'best_model', 'study'
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available. Install with: pip install optuna")

    from econml.score import RScorer

    if param_space is None:
        param_space = get_param_space_optuna(model_type)

    # Create RScorer for evaluation
    scorer = RScorer(
        model_y=GradientBoostingRegressor(max_depth=3, n_estimators=50, random_state=seed),
        model_t=LogisticRegression(max_iter=1000, random_state=seed),
        discrete_treatment=True,
        cv=cv_folds,
        random_state=seed
    )

    # Fit nuisance models once
    scorer.fit(Y, T, X=X)

    def objective(trial):
        # Sample parameters
        params = _sample_params_optuna(trial, param_space)

        try:
            # Create model
            model = _create_model_from_params_optuna(model_type, params, seed)

            # Fit and score
            model.fit(Y, T, X=X)
            score = scorer.score(model)

            return score

        except Exception as e:
            if verbose:
                logging.warning(f"Trial {trial.number} failed: {e}")
            return float('-inf')

    # Create study with TPE sampler
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f'{model_type}_tuning'
    )

    # Optimize
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )

    # Get best params and create best model
    best_params = study.best_params
    best_model = _create_model_from_params_optuna(model_type, best_params, seed)
    best_model.fit(Y, T, X=X)

    return {
        'best_params': best_params,
        'best_score': study.best_value,
        'best_model': best_model,
        'study': study,
    }


def tune_cate_models_optuna(
    model_types: List[str],
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 3,
    seed: int = 42,
    verbose: bool = True,
    n_jobs: int = -1,
) -> Dict[str, Dict[str, Any]]:
    """Tune multiple CATE models using Optuna.

    Runs tuning for each model type. Can be parallelized with Ray.

    Args:
        model_types: List of model types to tune
        Y: Outcome variable
        T: Treatment indicator
        X: Covariates
        n_trials: Number of Optuna trials per model
        cv_folds: Number of CV folds for RScorer
        seed: Random seed
        verbose: Whether to show progress
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Dict mapping model_type to tuning results
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available. Install with: pip install optuna")

    results = {}

    if RAY_AVAILABLE and ray.is_initialized() and n_jobs != 1:
        # Parallel tuning with Ray
        if verbose:
            print(f"Tuning {len(model_types)} models with Ray parallel ({n_trials} trials each)...")

        @ray.remote
        def _tune_remote(model_type, Y, T, X, n_trials, cv_folds, seed):
            result = tune_cate_model_optuna(
                model_type=model_type,
                Y=Y, T=T, X=X,
                n_trials=n_trials,
                cv_folds=cv_folds,
                seed=seed,
                verbose=False
            )
            # Remove non-serializable study object
            result_clean = {k: v for k, v in result.items() if k != 'study'}
            return model_type, result_clean

        # Put data in Ray object store
        Y_ref = ray.put(Y)
        T_ref = ray.put(T)
        X_ref = ray.put(X)

        # Launch parallel tuning
        futures = [
            _tune_remote.remote(model_type, Y_ref, T_ref, X_ref, n_trials, cv_folds, seed + i)
            for i, model_type in enumerate(model_types)
        ]

        # Collect results
        for result in ray.get(futures):
            model_type, tuning_result = result
            results[model_type] = tuning_result
            if verbose:
                print(f"  {model_type}: best_score={tuning_result['best_score']:.4f}")

    else:
        # Sequential tuning
        if verbose:
            print(f"Tuning {len(model_types)} models sequentially ({n_trials} trials each)...")

        for i, model_type in enumerate(model_types):
            if verbose:
                print(f"  Tuning {model_type}...")

            result = tune_cate_model_optuna(
                model_type=model_type,
                Y=Y, T=T, X=X,
                n_trials=n_trials,
                cv_folds=cv_folds,
                seed=seed + i,
                verbose=verbose
            )

            # Remove non-serializable study object for consistency
            results[model_type] = {k: v for k, v in result.items() if k != 'study'}

            if verbose:
                print(f"    best_score={result['best_score']:.4f}")

    return results


def get_best_tuned_configs_optuna(tuning_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Convert Optuna tuning results to model configs for fit_cate_models_parallel.

    Args:
        tuning_results: Output from tune_cate_models_optuna

    Returns:
        Dict of configs compatible with get_cate_model_configs format
    """
    configs = {}

    for model_type, result in tuning_results.items():
        if result.get('best_model') is None:
            continue

        best_params = result.get('best_params', {})

        if model_type == 's_learner':
            configs[model_type] = {
                'type': 's_learner',
                'overall_model': GradientBoostingRegressor(
                    max_depth=best_params.get('max_depth', 4),
                    n_estimators=best_params.get('n_estimators', 100),
                    min_samples_leaf=best_params.get('min_samples_leaf', 20),
                    learning_rate=best_params.get('learning_rate', 0.1),
                    subsample=best_params.get('subsample', 1.0),
                    max_features=best_params.get('max_features', 'sqrt'),
                    random_state=42
                )
            }

        elif model_type == 't_learner':
            configs[model_type] = {
                'type': 't_learner',
                'models': GradientBoostingRegressor(
                    max_depth=best_params.get('max_depth', 4),
                    n_estimators=best_params.get('n_estimators', 100),
                    min_samples_leaf=best_params.get('min_samples_leaf', 20),
                    learning_rate=best_params.get('learning_rate', 0.1),
                    subsample=best_params.get('subsample', 1.0),
                    max_features=best_params.get('max_features', 'sqrt'),
                    random_state=42
                )
            }

        elif model_type == 'x_learner':
            base_regressor = GradientBoostingRegressor(
                max_depth=best_params.get('max_depth', 4),
                n_estimators=best_params.get('n_estimators', 100),
                min_samples_leaf=best_params.get('min_samples_leaf', 20),
                learning_rate=best_params.get('learning_rate', 0.1),
                subsample=best_params.get('subsample', 1.0),
                max_features=best_params.get('max_features', 'sqrt'),
                random_state=42
            )
            configs[model_type] = {
                'type': 'x_learner',
                'models': base_regressor,
                'propensity_model': LogisticRegression(
                    C=best_params.get('propensity_C', 1.0),
                    max_iter=1000, random_state=42
                ),
                'cate_models': GradientBoostingRegressor(
                    max_depth=best_params.get('max_depth', 4),
                    n_estimators=best_params.get('n_estimators', 100),
                    min_samples_leaf=best_params.get('min_samples_leaf', 20),
                    learning_rate=best_params.get('learning_rate', 0.1),
                    random_state=43
                )
            }

        elif model_type == 'linear_dml':
            configs[model_type] = {
                'type': 'linear_dml',
                'model_y': GradientBoostingRegressor(
                    max_depth=best_params.get('model_y_max_depth', 4),
                    n_estimators=best_params.get('model_y_n_estimators', 100),
                    learning_rate=best_params.get('model_y_learning_rate', 0.1),
                    random_state=42
                ),
                'model_t': LogisticRegression(
                    C=best_params.get('model_t_C', 1.0),
                    max_iter=1000, random_state=42
                ),
                'cv': best_params.get('cv', 5),
                'use_ray': True
            }

        elif model_type == 'causal_forest_dml':
            configs[model_type] = {
                'type': 'causal_forest_dml',
                'model_y': GradientBoostingRegressor(
                    max_depth=best_params.get('model_y_max_depth', 4),
                    n_estimators=best_params.get('model_y_n_estimators', 100),
                    learning_rate=best_params.get('model_y_learning_rate', 0.1),
                    random_state=42
                ),
                'model_t': LogisticRegression(
                    C=best_params.get('model_t_C', 1.0),
                    max_iter=1000, random_state=42
                ),
                'n_estimators': best_params.get('n_estimators', 200),
                'max_depth': best_params.get('max_depth', None),
                'min_samples_leaf': best_params.get('min_samples_leaf', 10),
                'min_impurity_decrease': best_params.get('min_impurity_decrease', 0.0),
                'max_samples': best_params.get('max_samples', 0.5),
                'cv': best_params.get('cv', 5),
                'use_ray': True
            }

    return configs


# =============================================================================
# Context Manager
# =============================================================================

class RayContext:
    """Context manager for Ray initialization/shutdown.

    Usage:
        with RayContext(num_cpus=4):
            results = fit_cate_models_parallel(Y, T, X)
    """

    def __init__(self, num_cpus: Optional[int] = None, **kwargs):
        self.num_cpus = num_cpus
        self.kwargs = kwargs
        self.initialized = False

    def __enter__(self):
        self.initialized = init_ray(num_cpus=self.num_cpus, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.initialized:
            shutdown_ray()
        return False


# =============================================================================
# Policy Tree Tuning with Optuna
# =============================================================================

def get_policy_param_space_optuna(model_type: str) -> Dict[str, Any]:
    """Get Optuna search space for policy tree models.

    Args:
        model_type: One of 'policy_tree', 'dr_policy_tree', 'cate_tree'

    Returns:
        Dict mapping parameter names to (low, high, type) tuples or list of choices
    """
    if model_type == 'policy_tree':
        # econml PolicyTree parameters
        return {
            'max_depth': (2, 8, 'int'),
            'min_samples_leaf': (5, 150, 'int'),
            'min_balancedness_tol': (0.3, 0.5, 'float'),
        }

    elif model_type == 'dr_policy_tree':
        # econml DRPolicyTree parameters
        return {
            'max_depth': (2, 8, 'int'),
            'min_samples_leaf': (5, 100, 'int'),
            'min_impurity_decrease': (0.0, 0.05, 'float'),
        }

    elif model_type == 'cate_tree':
        # sklearn DecisionTreeRegressor for CATE approximation
        return {
            'max_depth': (2, 10, 'int'),
            'min_samples_leaf': (5, 150, 'int'),
            'min_samples_split': (10, 100, 'int'),
            'min_impurity_decrease': (0.0, 0.1, 'float'),
            'ccp_alpha': (0.0, 0.05, 'float'),
            'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0, None],
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _sample_policy_params_optuna(trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample parameters from Optuna trial for policy trees."""
    params = {}

    for name, spec in param_space.items():
        if isinstance(spec, list):
            # Categorical
            params[name] = trial.suggest_categorical(name, spec)
        elif isinstance(spec, tuple) and len(spec) == 3:
            low, high, param_type = spec
            if param_type == 'int':
                params[name] = trial.suggest_int(name, low, high)
            elif param_type == 'float':
                params[name] = trial.suggest_float(name, low, high)
            elif param_type == 'log':
                params[name] = trial.suggest_float(name, low, high, log=True)
        else:
            params[name] = spec  # Fixed value

    return params


def tune_policy_tree_optuna(
    model_type: str,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    ps: np.ndarray,
    cate: Optional[np.ndarray] = None,
    cost_per_contact: float = 0.0,
    margin_rate: float = 1.0,
    n_trials: int = 50,
    param_space: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    verbose: bool = False,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Tune policy tree using Optuna with policy value optimization.

    Args:
        model_type: 'policy_tree', 'dr_policy_tree', or 'cate_tree'
        X: Feature matrix
        Y: Outcome variable
        T: Treatment indicator
        ps: Propensity scores
        cate: CATE predictions (required for 'cate_tree')
        cost_per_contact: Cost per treatment
        margin_rate: Profit margin rate
        n_trials: Number of Optuna trials
        param_space: Custom parameter space (optional)
        seed: Random seed
        verbose: Show progress
        timeout: Timeout in seconds

    Returns:
        Dict with 'best_params', 'best_value', 'best_model', 'study', 'all_results'
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available. Install with: pip install optuna")

    if param_space is None:
        param_space = get_policy_param_space_optuna(model_type)

    if model_type == 'cate_tree' and cate is None:
        raise ValueError("cate is required for 'cate_tree' model_type")

    # Import tree models
    from sklearn.tree import DecisionTreeRegressor

    try:
        from econml.policy import PolicyTree, DRPolicyTree
        ECONML_POLICY_AVAILABLE = True
    except ImportError:
        ECONML_POLICY_AVAILABLE = False

    # Prepare profit-based rewards for PolicyTree
    if model_type == 'policy_tree' and cate is not None:
        profit_if_target = cate * margin_rate - cost_per_contact
        Y_rewards = np.column_stack([np.zeros_like(cate), profit_if_target])
    else:
        Y_rewards = None

    all_results = []

    def objective(trial):
        params = _sample_policy_params_optuna(trial, param_space)

        try:
            if model_type == 'policy_tree':
                if not ECONML_POLICY_AVAILABLE:
                    return float('-inf')

                tree = PolicyTree(
                    max_depth=params.get('max_depth', 3),
                    min_samples_leaf=params.get('min_samples_leaf', 20),
                    min_balancedness_tol=params.get('min_balancedness_tol', 0.45),
                    random_state=seed
                )
                tree.fit(X, Y_rewards)
                policy = tree.predict(X)

                # Profit-based value
                value = (policy * (cate * margin_rate - cost_per_contact)).sum()

            elif model_type == 'dr_policy_tree':
                if not ECONML_POLICY_AVAILABLE:
                    return float('-inf')

                tree = DRPolicyTree(
                    max_depth=params.get('max_depth', 3),
                    min_samples_leaf=params.get('min_samples_leaf', 20),
                    min_impurity_decrease=params.get('min_impurity_decrease', 0.0),
                    honest=True,
                    cv=3,
                    random_state=seed
                )
                tree.fit(Y, T, X=X)
                policy = tree.predict(X)

                # Avoid trivial solutions
                target_rate = policy.mean()
                if target_rate < 0.01 or target_rate > 0.99:
                    return float('-inf')

                # IPW policy value
                w = np.where(T == policy, 1 / np.where(policy == 1, ps, 1 - ps), 0)
                value = (w * Y).sum() / (w.sum() + 1e-6)

            elif model_type == 'cate_tree':
                tree = DecisionTreeRegressor(
                    max_depth=params.get('max_depth', 4),
                    min_samples_leaf=params.get('min_samples_leaf', 20),
                    min_samples_split=params.get('min_samples_split', 40),
                    min_impurity_decrease=params.get('min_impurity_decrease', 0.0),
                    ccp_alpha=params.get('ccp_alpha', 0.0),
                    max_features=params.get('max_features', None),
                    random_state=seed
                )
                tree.fit(X, cate)
                tree_cate = tree.predict(X)
                policy = (tree_cate > 0).astype(int)

                # IPW policy value
                w = np.where(T == policy, 1 / np.where(policy == 1, ps, 1 - ps), 0)
                value = (w * Y).sum() / (w.sum() + 1e-6)

            else:
                return float('-inf')

            # Store result
            n_targeted = int(policy.sum())
            all_results.append({
                **params,
                'value': value,
                'n_targeted': n_targeted,
                'pct_targeted': n_targeted / len(policy)
            })

            return value

        except Exception as e:
            if verbose:
                logging.warning(f"Trial {trial.number} failed: {e}")
            return float('-inf')

    # Create study
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f'{model_type}_tuning'
    )

    # Optimize
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )

    # Get best result and refit
    best_params = study.best_params
    best_value = study.best_value

    # Refit best model
    if model_type == 'policy_tree':
        best_model = PolicyTree(
            max_depth=best_params.get('max_depth', 3),
            min_samples_leaf=best_params.get('min_samples_leaf', 20),
            min_balancedness_tol=best_params.get('min_balancedness_tol', 0.45),
            random_state=seed
        )
        best_model.fit(X, Y_rewards)

    elif model_type == 'dr_policy_tree':
        best_model = DRPolicyTree(
            max_depth=best_params.get('max_depth', 3),
            min_samples_leaf=best_params.get('min_samples_leaf', 20),
            min_impurity_decrease=best_params.get('min_impurity_decrease', 0.0),
            honest=True,
            cv=3,
            random_state=seed
        )
        best_model.fit(Y, T, X=X)

    elif model_type == 'cate_tree':
        best_model = DecisionTreeRegressor(
            max_depth=best_params.get('max_depth', 4),
            min_samples_leaf=best_params.get('min_samples_leaf', 20),
            min_samples_split=best_params.get('min_samples_split', 40),
            min_impurity_decrease=best_params.get('min_impurity_decrease', 0.0),
            ccp_alpha=best_params.get('ccp_alpha', 0.0),
            max_features=best_params.get('max_features', None),
            random_state=seed
        )
        best_model.fit(X, cate)

    else:
        best_model = None

    return {
        'best_params': best_params,
        'best_value': best_value,
        'best_model': best_model,
        'study': study,
        'all_results': pd.DataFrame(all_results).sort_values('value', ascending=False)
    }


def tune_policy_trees_optuna(
    model_types: List[str],
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    ps: np.ndarray,
    cate: Optional[np.ndarray] = None,
    cost_per_contact: float = 0.0,
    margin_rate: float = 1.0,
    n_trials: int = 50,
    seed: int = 42,
    verbose: bool = True,
    n_jobs: int = -1,
) -> Dict[str, Dict[str, Any]]:
    """Tune multiple policy trees in parallel using Ray + Optuna.

    Args:
        model_types: List of model types ('policy_tree', 'dr_policy_tree', 'cate_tree')
        X, Y, T, ps, cate: Data arrays
        cost_per_contact: Cost per treatment
        margin_rate: Profit margin
        n_trials: Optuna trials per model
        seed: Random seed
        verbose: Show progress
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Dict mapping model_type to tuning results
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available. Install with: pip install optuna")

    results = {}

    if RAY_AVAILABLE and ray.is_initialized() and n_jobs != 1:
        if verbose:
            print(f"Tuning {len(model_types)} policy trees in parallel ({n_trials} trials each)...")

        @ray.remote
        def _tune_remote(model_type, X_ref, Y_ref, T_ref, ps_ref, cate_ref,
                         cost, margin, n_trials, seed):
            X = ray.get(X_ref)
            Y = ray.get(Y_ref)
            T = ray.get(T_ref)
            ps = ray.get(ps_ref)
            cate = ray.get(cate_ref) if cate_ref is not None else None

            result = tune_policy_tree_optuna(
                model_type=model_type,
                X=X, Y=Y, T=T, ps=ps, cate=cate,
                cost_per_contact=cost,
                margin_rate=margin,
                n_trials=n_trials,
                seed=seed,
                verbose=False
            )
            # Remove non-serializable objects
            result_clean = {k: v for k, v in result.items() if k != 'study'}
            return model_type, result_clean

        # Put data in Ray object store
        X_ref = ray.put(X)
        Y_ref = ray.put(Y)
        T_ref = ray.put(T)
        ps_ref = ray.put(ps)
        cate_ref = ray.put(cate) if cate is not None else None

        # Launch parallel tuning
        futures = [
            _tune_remote.remote(
                model_type, X_ref, Y_ref, T_ref, ps_ref, cate_ref,
                cost_per_contact, margin_rate, n_trials, seed + i
            )
            for i, model_type in enumerate(model_types)
        ]

        # Collect results
        for result in ray.get(futures):
            model_type, tuning_result = result
            results[model_type] = tuning_result
            if verbose:
                print(f"  {model_type}: best_value={tuning_result['best_value']:.2f}")

    else:
        # Sequential tuning
        if verbose:
            print(f"Tuning {len(model_types)} policy trees sequentially ({n_trials} trials each)...")

        for i, model_type in enumerate(model_types):
            if verbose:
                print(f"  Tuning {model_type}...")

            result = tune_policy_tree_optuna(
                model_type=model_type,
                X=X, Y=Y, T=T, ps=ps, cate=cate,
                cost_per_contact=cost_per_contact,
                margin_rate=margin_rate,
                n_trials=n_trials,
                seed=seed + i,
                verbose=verbose
            )

            # Remove non-serializable study object
            results[model_type] = {k: v for k, v in result.items() if k != 'study'}

            if verbose:
                print(f"    best_value={result['best_value']:.2f}")

    return results
