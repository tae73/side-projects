"""Segmentation project source modules."""

from .features import (
    build_all_features,
    build_rfm_features,
    build_behavioral_features,
    build_category_features,
    build_time_features,
    FEATURE_COLS,
    ALL_FEATURE_COLS,
    MACRO_CATEGORY,
)

from .metrics import (
    UpliftMetrics,
    compute_auuc,
    compute_qini_coef,
    compute_uplift_at_k,
    compute_all_uplift_metrics,
    compare_models_uplift,
)

from .business import (
    ROIConfig,
    ROISummary,
    compute_roi_curve,
    compute_breakeven_analysis,
    extract_campaign_cost,
    compute_targeting_summary,
    compute_segment_targeting_priority,
)

from .utils import (
    safe_qcut,
    format_currency,
    format_percentage,
    create_ps_region_labels,
    is_in_overlap,
    summary_stats,
)

from .treatment_effects import (
    # ATE Estimators
    estimate_ate_naive,
    estimate_ate_ipw,
    estimate_ate_aipw,
    estimate_ate_ols,
    estimate_ate_dml,
    estimate_all_ate,
    estimate_ate_ato,
    # Propensity Score
    estimate_propensity_score,
    estimate_propensity_score_cv,
    # Positivity Diagnostics
    compute_positivity_diagnostics,
    compute_covariate_balance,
    # Trimming & Weighting
    apply_ps_trimming,
    compute_ato_weights,
    # Bounds
    compute_ate_manski_bounds,
    compute_cate_bounds,
    compute_cate_bounds_with_overlap,
    compute_monotone_cate_bounds,
    # Sensitivity
    positivity_sensitivity,
    compute_e_value,
    # Confounder Analysis
    analyze_ps_feature_importance,
    run_covariate_experiment,
    # BLP Test
    blp_test,
    # Types
    ATEResult,
    CATEBounds,
    PositivityDiagnostics,
)

from .plots import (
    PlotConfig,
    setup_style,
    # PS Plots
    plot_ps_distribution,
    plot_ps_overlap,
    plot_ps_train_test,
    # Balance Plots
    plot_love_plot,
    plot_balance_comparison,
    # Uplift Plots
    plot_uplift_curve,
    plot_uplift_with_auuc,
    plot_qini_curve,
    plot_qini_with_coef,
    plot_cate_comparison,
    plot_cate_by_segment,
    plot_cate_boxplot_by_segment,
    # Bounds Plots
    plot_cate_bounds,
    plot_bounds_by_ps,
    # Sensitivity Plots
    plot_trimming_sensitivity,
    plot_model_comparison_heatmap,
    # ATE Plots
    plot_ate_comparison,
    plot_ate_forest,
    # Experiment Plots
    plot_covariate_experiment,
    plot_positivity_summary,
    # ROI Plots
    plot_roi_curves,
)

from .ray_utils import (
    init_ray,
    shutdown_ray,
    is_ray_available,
    RayContext,
    fit_cate_models_parallel,
    fit_cate_models_sequential,
    run_covariate_experiments_parallel,
    bootstrap_ate_parallel,
    create_ray_enabled_dml,
    get_cate_model_configs,
    # Hyperparameter Tuning (Grid Search)
    get_param_grid,
    tune_cate_model_rscorer,
    tune_cate_models_parallel,
    get_best_tuned_configs,
    # Hyperparameter Tuning (Optuna)
    get_param_space_optuna,
    tune_cate_model_optuna,
    tune_cate_models_optuna,
    get_best_tuned_configs_optuna,
)