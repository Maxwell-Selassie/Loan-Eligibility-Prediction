# ============================================================================
# FILE: src/eda/data_quality.py
# ============================================================================
"""
Data quality checks: missing values, duplicates, outliers.
Optimized for performance and comprehensive reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_missing_values(
    df: pd.DataFrame,
    warning_threshold: float = 0.05,
    critical_threshold: float = 0.30
) -> Dict[str, any]:
    """
    Analyze missing values with thresholds.
    
    Args:
        df: Input DataFrame
        warning_threshold: Threshold for warning level
        critical_threshold: Threshold for critical level
        
    Returns:
        Dictionary with missing value analysis
    """
    logger.info("Analyzing missing values...")
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        logger.info("No missing values found")
        return {"has_missing": False, "missing_df": pd.DataFrame()}
    
    total_rows = len(df)
    missing_pct = (missing / total_rows).round(4)
    
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percentage': missing_pct,
        'severity': ['CRITICAL' if pct >= critical_threshold 
                    else 'WARNING' if pct >= warning_threshold 
                    else 'INFO' 
                    for pct in missing_pct]
    })
    
    # Log summary
    critical_cols = missing_df[missing_df['severity'] == 'CRITICAL']
    warning_cols = missing_df[missing_df['severity'] == 'WARNING']
    
    logger.info(f"Missing values found in {len(missing_df)} columns")
    if len(critical_cols) > 0:
        logger.warning(f"CRITICAL: {len(critical_cols)} columns exceed {critical_threshold*100}% missing")
    if len(warning_cols) > 0:
        logger.warning(f"WARNING: {len(warning_cols)} columns exceed {warning_threshold*100}% missing")
    
    return {
        "has_missing": True,
        "missing_df": missing_df,
        "critical_columns": critical_cols.index.tolist(),
        "warning_columns": warning_cols.index.tolist()
    }


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Check for duplicate rows.
    
    Args:
        df: Input DataFrame
        subset: Columns to check for duplicates
        
    Returns:
        Dictionary with duplicate analysis
    """
    logger.info("Checking for duplicate rows...")
    
    duplicates = df.duplicated(subset=subset, keep='first')
    n_duplicates = duplicates.sum()
    
    if n_duplicates == 0:
        logger.info("No duplicate rows found")
        return {"has_duplicates": False, "count": 0, "percentage": 0.0}
    
    duplicate_pct = (n_duplicates / len(df)) * 100
    logger.warning(f"Found {n_duplicates:,} duplicate rows ({duplicate_pct:.2f}%)")
    
    return {
        "has_duplicates": True,
        "count": n_duplicates,
        "percentage": duplicate_pct,
        "duplicate_rows": df[duplicates]
    }


def detect_outliers_iqr(
    df: pd.DataFrame,
    numeric_columns: List[str],
    multiplier: float = 1.5,
    id_column: Optional[str] = None
) -> Dict[str, any]:
    """
    Detect outliers using IQR method (vectorized for speed).
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to check
        multiplier: IQR multiplier (default 1.5)
        id_column: ID column to exclude
        
    Returns:
        Dictionary with outlier analysis per column
    """
    logger.info(f"Detecting outliers using IQR method (multiplier={multiplier})...")
    
    # Filter out ID column
    cols_to_check = [col for col in numeric_columns if col != id_column]
    
    outlier_summary = {}
    
    # Vectorized computation
    for col in cols_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Vectorized outlier detection
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = outlier_mask.sum()
        outlier_pct = (n_outliers / len(df)) * 100
        
        outlier_summary[col] = {
            'count': int(n_outliers),
            'percentage': round(outlier_pct, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'Q1': round(Q1, 2),
            'Q3': round(Q3, 2),
            'IQR': round(IQR, 2)
        }
        
        logger.info(f"{col}: {n_outliers} outliers ({outlier_pct:.2f}%) | Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return outlier_summary


# ============================================================================
# FILE: src/eda/descriptive_stats.py
# ============================================================================
"""
Descriptive statistics computation with optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Compute basic statistics efficiently.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with basic statistics
    """
    logger.info("Computing basic descriptive statistics...")
    
    stats = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'columns': df.columns.tolist()
    }
    
    logger.info(f"Dataset: {stats['n_rows']:,} rows × {stats['n_columns']} columns")
    logger.info(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
    
    return stats


def analyze_numeric_columns(
    df: pd.DataFrame,
    id_column: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Analyze numeric columns with summary statistics.
    
    Args:
        df: Input DataFrame
        id_column: ID column to exclude from analysis
        
    Returns:
        Tuple of (summary DataFrame, list of numeric columns)
    """
    logger.info("Analyzing numeric columns...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID column
    if id_column and id_column in numeric_cols:
        numeric_cols.remove(id_column)
    
    if not numeric_cols:
        logger.warning("No numeric columns found")
        return pd.DataFrame(), []
    
    # Compute statistics (vectorized)
    summary = df[numeric_cols].describe().T
    summary['min'] = df[numeric_cols].min()
    summary['max'] = df[numeric_cols].max()
    summary['range'] = summary['max'] - summary['min']
    summary['cv'] = (summary['std'] / summary['mean']).round(4)  # Coefficient of variation
    
    logger.info(f"Analyzed {len(numeric_cols)} numeric columns")
    
    return summary, numeric_cols


def analyze_categorical_columns(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Analyze categorical columns.
    
    Args:
        df: Input DataFrame
        target_column: Target column (will be included in analysis)
        
    Returns:
        Tuple of (summary DataFrame, list of categorical columns)
    """
    logger.info("Analyzing categorical columns...")
    
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not cat_cols:
        logger.warning("No categorical columns found")
        return pd.DataFrame(), []
    
    summary_data = []
    for col in cat_cols:
        summary_data.append({
            'column': col,
            'unique_count': df[col].nunique(),
            'most_frequent': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
            'top_5_values': df[col].value_counts().head(5).to_dict()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    logger.info(f"Analyzed {len(cat_cols)} categorical columns")
    
    return summary_df, cat_cols


# ============================================================================
# FILE: src/eda/inferential_stats.py
# ============================================================================
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

logger = logging.getLogger(__name__)


def calculate_confidence_interval(
    data: pd.Series,
    confidence: float = 0.95
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
    alpha: float = 0.05
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
    target_column: str,
    positive_value: str,
    negative_value: str,
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


# ============================================================================
# FILE: src/eda/visualizations.py
# ============================================================================
"""
Visualization module with optimized plotting and memory management.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def setup_plot_style(style: str = "seaborn-v0_8-darkgrid", context: str = "notebook"):
    """Setup matplotlib/seaborn style."""
    try:
        plt.style.use(style)
    except:
        plt.style.use('default')
    sns.set_context(context)


def plot_numeric_distributions(
    df: pd.DataFrame,
    numeric_columns: List[str],
    output_dir: Path,
    figsize: Tuple[int, int] = (18, 10),
    dpi: int = 300,
    kde: bool = True
) -> None:
    """
    Plot distributions for numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns
        output_dir: Output directory for plots
        figsize: Figure size
        dpi: DPI for saved figures
        kde: Whether to show KDE
    """
    logger.info(f"Plotting distributions for {len(numeric_columns)} numeric columns...")
    
    n_cols = len(numeric_columns)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_columns):
        try:
            sns.histplot(
                data=df, 
                x=col, 
                kde=kde, 
                ax=axes[idx],
                color='purple',
                alpha=0.7
            )
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting {col}: {e}")
    
    # Hide empty subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = output_dir / "numeric_distributions.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved: {output_file}")


def plot_boxplots(
    df: pd.DataFrame,
    numeric_columns: List[str],
    output_dir: Path,
    figsize: Tuple[int, int] = (18, 10),
    dpi: int = 300
) -> None:
    """
    Plot boxplots for outlier detection.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns
        output_dir: Output directory
        figsize: Figure size
        dpi: DPI for saved figures
    """
    logger.info(f"Plotting boxplots for {len(numeric_columns)} columns...")
    
    n_cols = len(numeric_columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_columns):
        try:
            sns.boxplot(
                data=df,
                y=col,
                ax=axes[idx],
                color='gold',
                linewidth=2
            )
            axes[idx].set_title(f'Boxplot - {col}', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting {col}: {e}")
    
    # Hide empty subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = output_dir / "boxplots_outliers.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved: {output_file}")


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_columns: List[str],
    output_dir: Path,
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 300
) -> None:
    """
    Plot count plots for categorical columns.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical columns
        output_dir: Output directory
        figsize: Figure size
        dpi: DPI
    """
    logger.info(f"Plotting categorical distributions for {len(categorical_columns)} columns...")
    
    n_cols = len(categorical_columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(categorical_columns):
        try:
            ax = sns.countplot(
                data=df,
                x=col,
                ax=axes[idx],
                color='green',
                saturation=0.8
            )
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, label_type='edge')
            
            ax.set_title(f'{col}', fontweight='bold')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        except Exception as e:
            logger.error(f"Error plotting {col}: {e}")
    
    # Hide empty subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = output_dir / "categorical_distributions.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved: {output_file}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_columns: List[str],
    output_dir: Path,
    method: str = 'spearman',
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
) -> None:
    """
    Plot correlation heatmap.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns
        output_dir: Output directory
        method: Correlation method
        figsize: Figure size
        dpi: DPI
    """
    logger.info(f"Plotting correlation heatmap ({method} method)...")
    
    if len(numeric_columns) < 2:
        logger.warning("Need at least 2 numeric columns for correlation")
        return
    
    try:
        corr = df[numeric_columns].corr(method=method)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            linewidths=0.5,
            square=True,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title(f'Correlation Heatmap ({method.capitalize()})', fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_file = output_dir / f"correlation_heatmap_{method}.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved: {output_file}")
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")


def plot_target_distribution(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300
) -> None:
    """
    Plot target variable distribution.
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        output_dir: Output directory
        figsize: Figure size
        dpi: DPI
    """
    logger.info(f"Plotting target distribution for '{target_column}'...")
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        counts = df[target_column].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        
        ax = sns.countplot(
            data=df,
            x=target_column,
            palette=colors,
            saturation=0.8
        )
        
        # Add percentages
        total = len(df)
        for i, (label, count) in enumerate(counts.items()):
            percentage = (count / total) * 100
            ax.text(i, count, f'{count}\n({percentage:.1f}%)', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Target Variable Distribution: {target_column}', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(target_column, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_dir / "target_distribution.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved: {output_file}")
    except Exception as e:
        logger.error(f"Error plotting target distribution: {e}")


# ============================================================================
# FILE: src/eda_pipeline.py
# ============================================================================
"""
Main EDA Pipeline Orchestrator
Production-grade with comprehensive error handling, logging, and performance optimization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    read_csv, write_csv, read_yaml, write_json, save_joblib,
    ensure_directory, get_timestamp, Timer, setup_logger
)
from eda.data_quality import (
    check_missing_values, check_duplicates, detect_outliers_iqr
)
from eda.descriptive_stats import (
    compute_basic_stats, analyze_numeric_columns, analyze_categorical_columns
)
from eda.inferential_stats import (
    compute_confidence_intervals, run_ttest_parallel,
    run_chi_square_tests, multi_group_comparison
)
from eda.visualizations import (
    setup_plot_style, plot_numeric_distributions, plot_boxplots,
    plot_categorical_distributions, plot_correlation_heatmap,
    plot_target_distribution
)


class EDAExecutionError(Exception):
    """Custom exception for EDA pipeline errors."""
    pass


class EDAPipeline:
    """
    Production-grade EDA Pipeline for Loan Eligibility Prediction.
    
    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        timestamp: Pipeline execution timestamp
        results: Dictionary storing all analysis results
    """
    
    def __init__(self, config_path: str = "../config/eda_config.yaml"):
        """
        Initialize EDA Pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.timestamp = get_timestamp()
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.results: Dict[str, Any] = {}
        self.df: Optional[pd.DataFrame] = None
        
        self.logger.info("="*80)
        self.logger.info(f"EDA PIPELINE INITIALIZED - {self.timestamp}")
        self.logger.info("="*80)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config = read_yaml(config_path)
            return config
        except FileNotFoundError:
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> Any:
        """Setup logging system."""
        log_config = self.config.get('logging', {})
        log_dir = Path(log_config.get('log_dir', '../logs/'))
        
        ensure_directory(log_dir)
        
        logger = setup_logger(
            name='eda_pipeline',
            log_dir=log_dir,
            log_level=log_config.get('log_level', 'INFO'),
            max_bytes=log_config.get('max_bytes', 10485760),
            backup_count=log_config.get('backup_count', 7)
        )
        
        return logger
    
    def _ensure_output_directories(self) -> None:
        """Ensure all output directories exist."""
        self.logger.info("Creating output directories...")
        
        output_config = self.config.get('output', {})
        plots_dir = Path(output_config.get('plots_dir', '../plots/'))
        
        data_config = self.config.get('data', {})
        artifacts_dir = Path(data_config.get('artifacts_path', '../data/artifacts/'))
        processed_dir = Path(data_config.get('processed_path', '../data/processed/'))
        
        ensure_directory(plots_dir)
        ensure_directory(artifacts_dir)
        ensure_directory(processed_dir)
        
        # Store paths for later use
        self.plots_dir = plots_dir
        self.artifacts_dir = artifacts_dir
        self.processed_dir = processed_dir
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data with optimizations.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            EDAExecutionError: If data loading fails
        """
        with Timer("Data loading", self.logger):
            try:
                data_config = self.config.get('data', {})
                raw_path = data_config.get('raw_path')
                
                self.logger.info(f"Loading data from: {raw_path}")
                
                df = read_csv(
                    filepath=raw_path,
                    optimize_dtypes=data_config.get('optimize_dtypes', True),
                    categorical_columns=data_config.get('categorical_columns', [])
                )
                
                self.df = df
                self.logger.info(f"Data loaded: {len(df):,} rows × {len(df.columns)} columns")
                
                return df
                
            except Exception as e:
                self.logger.error(f"Data loading failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Failed to load data: {e}")
    
    def run_data_quality_checks(self) -> Dict[str, Any]:
        """
        Execute data quality checks.
        
        Returns:
            Dictionary with quality check results
        """
        self.logger.info("="*80)
        self.logger.info("DATA QUALITY CHECKS")
        self.logger.info("="*80)
        
        quality_results = {}
        
        with Timer("Data quality checks", self.logger):
            try:
                dq_config = self.config.get('data_quality', {})
                
                # Missing values
                missing_result = check_missing_values(
                    self.df,
                    warning_threshold=dq_config.get('missing_threshold_warning', 0.05),
                    critical_threshold=dq_config.get('missing_threshold_critical', 0.30)
                )
                quality_results['missing_values'] = missing_result
                
                # Duplicates
                if dq_config.get('check_duplicates', True):
                    duplicate_result = check_duplicates(self.df)
                    quality_results['duplicates'] = duplicate_result
                
                # Outliers
                if dq_config.get('detect_outliers', True):
                    numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                    id_col = self.config['data'].get('id_column')
                    
                    outlier_result = detect_outliers_iqr(
                        self.df,
                        numeric_cols,
                        multiplier=self.config['statistics'].get('outlier_iqr_multiplier', 1.5),
                        id_column=id_col
                    )
                    quality_results['outliers'] = outlier_result
                
                self.results['data_quality'] = quality_results
                return quality_results
                
            except Exception as e:
                self.logger.error(f"Data quality checks failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Data quality checks failed: {e}")
    
    def run_descriptive_analysis(self) -> Dict[str, Any]:
        """
        Execute descriptive statistical analysis.
        
        Returns:
            Dictionary with descriptive statistics
        """
        self.logger.info("="*80)
        self.logger.info("DESCRIPTIVE STATISTICS")
        self.logger.info("="*80)
        
        descriptive_results = {}
        
        with Timer("Descriptive analysis", self.logger):
            try:
                # Basic stats
                basic_stats = compute_basic_stats(self.df)
                descriptive_results['basic'] = basic_stats
                
                # Numeric columns
                id_col = self.config['data'].get('id_column')
                numeric_summary, numeric_cols = analyze_numeric_columns(self.df, id_col)
                descriptive_results['numeric_summary'] = numeric_summary.to_dict()
                descriptive_results['numeric_columns'] = numeric_cols
                
                # Categorical columns
                target_col = self.config['data'].get('target_column')
                cat_summary, cat_cols = analyze_categorical_columns(self.df, target_col)
                descriptive_results['categorical_summary'] = cat_summary.to_dict()
                descriptive_results['categorical_columns'] = cat_cols
                
                # Target variable distribution
                if target_col and target_col in self.df.columns:
                    target_dist = self.df[target_col].value_counts().to_dict()
                    descriptive_results['target_distribution'] = target_dist
                    
                    self.logger.info(f"Target variable '{target_col}' distribution:")
                    for value, count in target_dist.items():
                        pct = (count / len(self.df)) * 100
                        self.logger.info(f"  {value}: {count:,} ({pct:.2f}%)")
                
                self.results['descriptive_stats'] = descriptive_results
                return descriptive_results
                
            except Exception as e:
                self.logger.error(f"Descriptive analysis failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Descriptive analysis failed: {e}")
    
    def run_inferential_analysis(self) -> Dict[str, Any]:
        """
        Execute inferential statistical analysis.
        
        Returns:
            Dictionary with inferential statistics
        """
        self.logger.info("="*80)
        self.logger.info("INFERENTIAL STATISTICS")
        self.logger.info("="*80)
        
        inferential_results = {}
        
        with Timer("Inferential analysis", self.logger):
            try:
                stats_config = self.config.get('statistics', {})
                data_config = self.config.get('data', {})
                
                target_col = data_config.get('target_column')
                id_col = data_config.get('id_column')
                positive_val = data_config.get('target_positive_value', 'Y')
                negative_val = data_config.get('target_negative_value', 'N')
                
                # Get column lists
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                if id_col in numeric_cols:
                    numeric_cols.remove(id_col)
                
                cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
                if target_col in cat_cols:
                    cat_cols.remove(target_col)
                
                # Confidence intervals
                self.logger.info("-" * 60)
                self.logger.info("Computing Confidence Intervals")
                self.logger.info("-" * 60)
                
                ci_results = compute_confidence_intervals(
                    self.df,
                    numeric_cols,
                    confidence=stats_config.get('confidence_level', 0.95)
                )
                inferential_results['confidence_intervals'] = ci_results
                
                # T-tests (parallel execution)
                if target_col and target_col in self.df.columns:
                    self.logger.info("-" * 60)
                    self.logger.info("T-Tests: Numeric Features vs Target")
                    self.logger.info("-" * 60)
                    
                    perf_config = self.config.get('performance', {})
                    n_jobs = perf_config.get('n_jobs', -1) if perf_config.get('parallel_statistical_tests', True) else 1
                    
                    ttest_results = run_ttest_parallel(
                        self.df,
                        numeric_cols,
                        target_col,
                        positive_val,
                        negative_val,
                        n_jobs=n_jobs
                    )
                    inferential_results['ttests'] = ttest_results
                    
                    # Log significant results
                    significant_features = [
                        col for col, result in ttest_results.items() 
                        if result['significant']
                    ]
                    self.logger.info(f"Significant features ({len(significant_features)}): {significant_features}")
                    
                    # Chi-square tests
                    if cat_cols:
                        self.logger.info("-" * 60)
                        self.logger.info("Chi-Square Tests: Categorical Features vs Target")
                        self.logger.info("-" * 60)
                        
                        chi_square_results = run_chi_square_tests(
                            self.df,
                            cat_cols,
                            target_col,
                            alpha=stats_config.get('significance_level', 0.05)
                        )
                        inferential_results['chi_square_tests'] = chi_square_results
                
                self.results['inferential_stats'] = inferential_results
                return inferential_results
                
            except Exception as e:
                self.logger.error(f"Inferential analysis failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Inferential analysis failed: {e}")
    
    def generate_visualizations(self) -> None:
        """Generate all visualizations."""
        self.logger.info("="*80)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("="*80)
        
        with Timer("Visualization generation", self.logger):
            try:
                viz_config = self.config.get('visualization', {})
                output_config = self.config.get('output', {})
                
                # Setup plot style
                setup_plot_style(
                    style=viz_config.get('style', 'seaborn-v0_8-darkgrid'),
                    context=viz_config.get('context', 'notebook')
                )
                
                dpi = output_config.get('plot_dpi', 300)
                
                # Get column lists
                id_col = self.config['data'].get('id_column')
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                if id_col in numeric_cols:
                    numeric_cols.remove(id_col)
                
                cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
                target_col = self.config['data'].get('target_column')
                
                # Numeric distributions
                if numeric_cols:
                    plot_numeric_distributions(
                        self.df,
                        numeric_cols,
                        self.plots_dir,
                        figsize=tuple(viz_config.get('figure_size_univariate', [18, 10])),
                        dpi=dpi,
                        kde=viz_config.get('histogram', {}).get('kde', True)
                    )
                
                # Boxplots
                if numeric_cols:
                    plot_boxplots(
                        self.df,
                        numeric_cols,
                        self.plots_dir,
                        figsize=tuple(viz_config.get('figure_size_univariate', [18, 10])),
                        dpi=dpi
                    )
                
                # Categorical distributions
                if cat_cols:
                    plot_categorical_distributions(
                        self.df,
                        cat_cols,
                        self.plots_dir,
                        figsize=tuple(viz_config.get('figure_size_multivariate', [15, 10])),
                        dpi=dpi
                    )
                
                # Correlation heatmap
                if len(numeric_cols) >= 2:
                    plot_correlation_heatmap(
                        self.df,
                        numeric_cols,
                        self.plots_dir,
                        method='spearman',
                        figsize=tuple(viz_config.get('figure_size_multivariate', [15, 10])),
                        dpi=dpi
                    )
                
                # Target distribution
                if target_col and target_col in self.df.columns:
                    plot_target_distribution(
                        self.df,
                        target_col,
                        self.plots_dir,
                        figsize=(8, 6),
                        dpi=dpi
                    )
                
                self.logger.info(f"All visualizations saved to: {self.plots_dir}")
                
            except Exception as e:
                self.logger.error(f"Visualization generation failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Visualization generation failed: {e}")
    
    def save_artifacts(self) -> None:
        """Save all analysis artifacts."""
        self.logger.info("="*80)
        self.logger.info("SAVING ARTIFACTS")
        self.logger.info("="*80)
        
        with Timer("Artifact saving", self.logger):
            try:
                output_config = self.config.get('output', {})
                
                if not output_config.get('save_artifacts', True):
                    self.logger.info("Artifact saving disabled in config")
                    return
                
                artifact_format = output_config.get('artifact_format', 'joblib')
                
                # Add metadata
                self.results['metadata'] = {
                    'timestamp': self.timestamp,
                    'n_rows': len(self.df),
                    'n_columns': len(self.df.columns),
                    'columns': self.df.columns.tolist()
                }
                
                # Save based on format
                if artifact_format == 'joblib':
                    output_file = self.artifacts_dir / f"eda_results_{self.timestamp}.joblib"
                    save_joblib(self.results, output_file, compress=3)
                    self.logger.info(f"Saved artifacts (joblib): {output_file}")
                
                elif artifact_format == 'json':
                    output_file = self.artifacts_dir / f"eda_results_{self.timestamp}.json"
                    write_json(self.results, output_file, indent=2)
                    self.logger.info(f"Saved artifacts (JSON): {output_file}")
                
                # Also save summary report
                summary_file = self.artifacts_dir / f"eda_summary_{self.timestamp}.txt"
                self._generate_summary_report(summary_file)
                
            except Exception as e:
                self.logger.error(f"Artifact saving failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Artifact saving failed: {e}")
    
    def _generate_summary_report(self, output_file: Path) -> None:
        """Generate human-readable summary report."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"EDA SUMMARY REPORT - {self.timestamp}\n")
                f.write("="*80 + "\n\n")
                
                # Dataset overview
                f.write("DATASET OVERVIEW\n")
                f.write("-"*80 + "\n")
                basic = self.results.get('descriptive_stats', {}).get('basic', {})
                f.write(f"Rows: {basic.get('n_rows', 0):,}\n")
                f.write(f"Columns: {basic.get('n_columns', 0)}\n")
                f.write(f"Memory: {basic.get('memory_usage_mb', 0):.2f} MB\n\n")
                
                # Data quality
                f.write("DATA QUALITY\n")
                f.write("-"*80 + "\n")
                quality = self.results.get('data_quality', {})
                
                missing = quality.get('missing_values', {})
                if missing.get('has_missing'):
                    f.write(f"Missing values: {len(missing.get('missing_df', []))} columns affected\n")
                else:
                    f.write("Missing values: None\n")
                
                duplicates = quality.get('duplicates', {})
                if duplicates.get('has_duplicates'):
                    f.write(f"Duplicates: {duplicates.get('count', 0):,} rows ({duplicates.get('percentage', 0):.2f}%)\n")
                else:
                    f.write("Duplicates: None\n")
                
                f.write("\n")
                
                # Statistical tests
                f.write("STATISTICAL TESTS - SIGNIFICANT RESULTS\n")
                f.write("-"*80 + "\n")
                
                inferential = self.results.get('inferential_stats', {})
                
                # T-tests
                ttests = inferential.get('ttests', {})
                significant_ttests = {k: v for k, v in ttests.items() if v.get('significant')}
                f.write(f"\nT-Tests: {len(significant_ttests)}/{len(ttests)} features significant\n")
                for col, result in significant_ttests.items():
                    f.write(f"  - {col}: p={result['p_value']:.4f}, d={result['cohens_d']:.3f} ({result['effect_interpretation']})\n")
                
                # Chi-square
                chi_tests = inferential.get('chi_square_tests', {})
                significant_chi = {k: v for k, v in chi_tests.items() if v.get('significant')}
                f.write(f"\nChi-Square Tests: {len(significant_chi)}/{len(chi_tests)} features significant\n")
                for col, result in significant_chi.items():
                    f.write(f"  - {col}: p={result['p_value']:.4f}, V={result['cramers_v']:.3f} ({result['effect_interpretation']})\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")
            
            self.logger.info(f"Summary report saved: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute complete EDA pipeline.
        
        Returns:
            Dictionary with all analysis results
            
        Raises:
            EDAExecutionError: If pipeline execution fails
        """
        try:
            with Timer("Complete EDA Pipeline", self.logger):
                # Setup
                self._ensure_output_directories()
                
                # Load data
                self.load_data()
                
                # Analysis stages
                self.run_data_quality_checks()
                self.run_descriptive_analysis()
                self.run_inferential_analysis()
                
                # Visualizations
                self.generate_visualizations()
                
                # Save results
                self.save_artifacts()
                
                self.logger.info("="*80)
                self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("="*80)
                
                return self.results
                
        except EDAExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise EDAExecutionError(f"Pipeline failed: {e}")


def main():
    """Main entry point for EDA pipeline."""
    try:
        # Initialize pipeline
        pipeline = EDAPipeline(config_path="../config/eda_config.yaml")
        
        # Execute
        results = pipeline.execute()
        
        return 0
        
    except EDAExecutionError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())