"""statistical_tests.py

Rigorous statistical evaluation of packaging optimization.
Includes: t-tests, Mann-Whitney-U tests, effect sizes, sensitivity analysis.
"""

from typing import Tuple, Dict
import numpy as np
from scipy import stats
import pandas as pd


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0.0


def run_statistical_tests(baseline_values: np.ndarray, optimized_values: np.ndarray) -> Dict:
    """
    Run paired statistical tests: t-test and Mann-Whitney-U.
    Compute effect size (Cohen's d) and 95% CI.
    
    Returns:
    {
        'mean_baseline': float,
        'mean_optimized': float,
        'mean_diff': float,
        'ci_lower': float,
        'ci_upper': float,
        'ttest_pvalue': float,
        'ttest_statistic': float,
        'mannwhitneyu_pvalue': float,
        'mannwhitneyu_statistic': float,
        'cohens_d': float,
        'improvement_pct': float,
    }
    """
    mean_base = np.mean(baseline_values)
    mean_opt = np.mean(optimized_values)
    diff = mean_base - mean_opt
    
    # t-test
    t_stat, t_pval = stats.ttest_ind(baseline_values, optimized_values)
    
    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_pval = stats.mannwhitneyu(baseline_values, optimized_values, alternative='two-sided')
    
    # Effect size
    d = cohens_d(baseline_values, optimized_values)
    
    # 95% Confidence interval for the difference (bootstrap)
    diffs = baseline_values - optimized_values
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    improvement_pct = (diff / mean_base * 100.0) if mean_base > 0 else 0.0
    
    return {
        'mean_baseline': float(mean_base),
        'mean_optimized': float(mean_opt),
        'mean_diff': float(diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ttest_statistic': float(t_stat),
        'ttest_pvalue': float(t_pval),
        'mannwhitneyu_statistic': float(u_stat),
        'mannwhitneyu_pvalue': float(u_pval),
        'cohens_d': float(d),
        'improvement_pct': float(improvement_pct),
    }


def sensitivity_analysis(
    test_df: pd.DataFrame,
    param_name: str,
    param_values: list,
    objective_func,
    baseline_func
) -> pd.DataFrame:
    """
    Run sensitivity analysis: vary a parameter and measure impact on objective.
    
    Args:
        test_df: test dataframe with product dimensions
        param_name: name of parameter to vary (e.g., 'safety_margin')
        param_values: list of values to test
        objective_func: function(test_df, param_value) -> metric (e.g., mean waste reduction)
        baseline_func: function(test_df, param_value) -> baseline metric
    
    Returns: DataFrame with columns [param_value, objective_value, baseline_value, reduction_pct]
    """
    results = []
    for val in param_values:
        obj_val = objective_func(test_df, val)
        base_val = baseline_func(test_df, val)
        reduction = (base_val - obj_val) / base_val * 100.0 if base_val > 0 else 0.0
        results.append({
            'param_value': float(val),
            'optimized_metric': float(obj_val),
            'baseline_metric': float(base_val),
            'reduction_pct': float(reduction)
        })
    return pd.DataFrame(results)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def create_results_table(metrics_dict: Dict) -> pd.DataFrame:
    """
    Create a clean results table from multiple metric evaluations.
    
    Args:
        metrics_dict: {'metric_name': statistical_test_result, ...}
    
    Returns: formatted DataFrame
    """
    rows = []
    for metric_name, result in metrics_dict.items():
        rows.append({
            'Metric': metric_name,
            'Baseline (mean)': f"{result['mean_baseline']:.4f}",
            'Optimized (mean)': f"{result['mean_optimized']:.4f}",
            'Improvement (%)': f"{result['improvement_pct']:.2f}%",
            'p-value (t-test)': f"{result['ttest_pvalue']:.4f}",
            'Effect Size (d)': f"{result['cohens_d']:.4f} ({interpret_effect_size(result['cohens_d'])})",
            '95% CI': f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]",
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Test
    baseline = np.random.normal(100, 10, 50)
    optimized = np.random.normal(85, 10, 50)
    result = run_statistical_tests(baseline, optimized)
    print("Statistical test result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
