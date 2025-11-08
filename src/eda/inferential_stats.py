"""
Inferential statistics: hypothesis testing, confidence intervals.
Optimized with parallel processing for multiple tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, kruskal
from typing import Dict, List, Tuple, Optional
import logging
from joblib import Parallel, delayed
from utils.io_utils import read_yaml
logger = logging.getLogger(__name__)

config =  read_yaml('config/EDA_config.yaml')

def calculate_confidence_interval(
    data: pd.Series,
    confidence: float = config['statistics'].get('confidence_level', 0.95)
) -> Dict[str, float]:
    """
    Calculate confidence interval for a numeric variable.
    
    Args:
        data: Numeric data series
        confidence: Confidence level
        
    Returns:
        Dictionary with CI statistics
    """
    data_clean = data.dropna()
    n = len(data_clean)
    
    if n < 2:
        return None
    
    mean = float(np.mean(data_clean))
    se = float(stats.sem(data_clean))
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return {
        'mean': round(mean, 4),
        'lower_bound': round(mean - margin, 4),
        'upper_bound': round(mean + margin, 4),
        'margin_of_error': round(margin, 4),
        'sample_size': n,
        'confidence_level': confidence
    }


def compute_confidence_intervals(
    df: pd.DataFrame,
    numeric_columns: List[str],
    confidence: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Compute confidence intervals for all numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns
        confidence: Confidence level
        
    Returns:
        Dictionary mapping column names to CI statistics
    """
    logger.info(f"Computing {confidence*100}% confidence intervals...")
    
    ci_results = {}
    for col in numeric_columns:
        ci = calculate_confidence_interval(df[col], confidence)
        if ci is not None:
            ci_results[col] = ci
            logger.info(f"{col}: Mean={ci['mean']:.2f}, CI=[{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]")
    
    return ci_results


def compare_two_groups_ttest(
    df: pd.DataFrame,
    numeric_col: str,
    grouping_col: str,
    group1_val: str,
    group2_val: str,
    alpha: float = config['statistics'].get('significance_level', 0.05)
) -> Dict[str, any]:
    """
    Compare two groups using t-test or Mann-Whitney U test.
    
    Args:
        df: Input DataFrame
        numeric_col: Numeric column to compare
        grouping_col: Column defining groups
        group1_val: Value for group 1
        group2_val: Value for group 2
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Extract groups
    group1 = df[df[grouping_col] == group1_val][numeric_col].dropna()
    group2 = df[df[grouping_col] == group2_val][numeric_col].dropna()
    
    # Check minimum sample size
    if len(group1) < 3 or len(group2) < 3:
        logger.warning(f"{numeric_col}: Insufficient sample size")
        return None
    
    # Check normality (sample if large dataset)
    sample_size = min(5000, len(group1), len(group2))
    _, p_norm1 = stats.shapiro(group1.sample(min(sample_size, len(group1)), random_state=42))
    _, p_norm2 = stats.shapiro(group2.sample(min(sample_size, len(group2)), random_state=42))
    
    # Choose test based on normality
    if p_norm1 < alpha or p_norm2 < alpha:
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        test_type = "Mann-Whitney U"
    else:
        statistic, p_value = ttest_ind(group1, group2, equal_var=False)
        test_type = "Welch's t-test"
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    cohens_d = abs(group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Interpret effect size
    if cohens_d < 0.2:
        effect_interpretation = "negligible"
    elif cohens_d < 0.5:
        effect_interpretation = "small"
    elif cohens_d < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    return {
        'test_type': test_type,
        'group1_mean': round(float(group1.mean()), 4),
        'group2_mean': round(float(group2.mean()), 4),
        'group1_std': round(float(group1.std()), 4),
        'group2_std': round(float(group2.std()), 4),
        'group1_n': len(group1),
        'group2_n': len(group2),
        'statistic': round(float(statistic), 4),
        'p_value': round(float(p_value), 6),
        'significant': p_value < alpha,
        'cohens_d': round(float(cohens_d), 4),
        'effect_interpretation': effect_interpretation
    }


def run_ttest_parallel(
    df: pd.DataFrame,
    numeric_columns: List[str],
    target_column: str = config['data'].get('target_column'),
    positive_value: str = config['data'].get('target_positive_value'),
    negative_value: str = config['data'].get('target_negative_value'),
    n_jobs: int = -1
) -> Dict[str, Dict[str, any]]:
    """
    Run t-tests for multiple columns in parallel.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to test
        target_column: Target column
        positive_value: Positive class value
        negative_value: Negative class value
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary mapping column names to test results
    """
    logger.info(f"Running t-tests for {len(numeric_columns)} columns (parallel)...")
    
    # Parallel execution
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compare_two_groups_ttest)(
            df, col, target_column, positive_value, negative_value
        )
        for col in numeric_columns
    )
    
    # Filter out None results
    ttest_results = {
        col: result 
        for col, result in zip(numeric_columns, results) 
        if result is not None
    }
    
    # Log summary
    significant_count = sum(1 for r in ttest_results.values() if r['significant'])
    logger.info(f"Completed: {significant_count}/{len(ttest_results)} columns show significant differences")
    
    return ttest_results


def chi_square_test(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Perform chi-square test of independence.
    
    Args:
        df: Input DataFrame
        cat_col: Categorical column
        target_col: Target column
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    contingency_table = pd.crosstab(df[cat_col], df[target_col])
    
    # Check minimum expected frequencies
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    if (expected < 5).any():
        logger.warning(f"{cat_col}: Expected frequencies < 5, results may be unreliable")
    
    # Cramér's V effect size
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    # Interpret effect size
    if cramers_v < 0.1:
        effect_interpretation = "negligible"
    elif cramers_v < 0.3:
        effect_interpretation = "small"
    elif cramers_v < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    return {
        'chi2': round(float(chi2), 4),
        'p_value': round(float(p_value), 6),
        'dof': int(dof),
        'cramers_v': round(float(cramers_v), 4),
        'effect_interpretation': effect_interpretation,
        'significant': p_value < alpha,
        'contingency_table': contingency_table.to_dict()
    }


def run_chi_square_tests(
    df: pd.DataFrame,
    categorical_columns: List[str],
    target_column: str,
    alpha: float = 0.05
) -> Dict[str, Dict[str, any]]:
    """
    Run chi-square tests for multiple categorical columns.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical columns
        target_column: Target column
        alpha: Significance level
        
    Returns:
        Dictionary mapping column names to test results
    """
    logger.info(f"Running chi-square tests for {len(categorical_columns)} columns...")
    
    chi_square_results = {}
    
    for col in categorical_columns:
        try:
            result = chi_square_test(df, col, target_column, alpha)
            chi_square_results[col] = result
            
            sig_marker = "✓" if result['significant'] else "✗"
            logger.info(f"{col}: χ²={result['chi2']:.2f}, p={result['p_value']:.4f}, V={result['cramers_v']:.3f} {sig_marker}")
        except Exception as e:
            logger.error(f"Error in chi-square test for {col}: {e}")
    
    # Log summary
    significant_count = sum(1 for r in chi_square_results.values() if r['significant'])
    logger.info(f"Completed: {significant_count}/{len(chi_square_results)} columns show significant association")
    
    return chi_square_results


def multi_group_comparison(
    df: pd.DataFrame,
    numeric_col: str,
    grouping_col: str,
    alpha: float = 0.05
) -> Optional[Dict[str, any]]:
    """
    Compare numeric variable across multiple groups (ANOVA/Kruskal-Wallis).
    
    Args:
        df: Input DataFrame
        numeric_col: Numeric column to compare
        grouping_col: Grouping column
        alpha: Significance level
        
    Returns:
        Dictionary with test results or None if insufficient data
    """
    groups = [group[numeric_col].dropna() for name, group in df.groupby(grouping_col)]
    
    # Check minimum sample size
    if len(groups) < 2 or any(len(g) < 3 for g in groups):
        logger.warning(f"{numeric_col} vs {grouping_col}: Insufficient sample size")
        return None
    
    # Check normality for each group
    normality_pvals = [
        stats.shapiro(g.sample(min(5000, len(g)), random_state=42))[1] 
        for g in groups
    ]
    
    # Choose test
    if all(p >= alpha for p in normality_pvals):
        statistic, p_value = stats.f_oneway(*groups)
        test_type = "One-way ANOVA"
    else:
        statistic, p_value = kruskal(*groups)
        test_type = "Kruskal-Wallis H-test"
    
    return {
        'test_type': test_type,
        'statistic': round(float(statistic), 4),
        'p_value': round(float(p_value), 6),
        'significant': p_value < alpha,
        'group_means': {
            str(name): round(float(group[numeric_col].mean()), 4)
            for name, group in df.groupby(grouping_col)
        }
    }