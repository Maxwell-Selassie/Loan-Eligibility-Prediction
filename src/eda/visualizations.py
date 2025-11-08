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