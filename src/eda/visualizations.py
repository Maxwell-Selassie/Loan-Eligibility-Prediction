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
import mlflow
from utils import LoggerMixin

class Visualizations(LoggerMixin):

    def __init__(self, config):
        self.config = config
        self.logger = self.setup_class_logger('visualizations',config, 'logging')
        self.output_dir = config['output'].get('plots_dir', 'plots/')
        self.dpi = config['output'].get('plot_dpi',300)
        self.figure_size_univariate = config['visualization'].get('figure_size_univariate',(15,10))
        self.figure_size_multivariate = config['visualization'].get('figure_size_multivariate',(18,10))

    def plot_numeric_distributions(self,
        df: pd.DataFrame,
    ) -> None:
        """
        Plot distributions for numeric columns.
        
        Args:
            df: Input DataFrame
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.logger.info(f"Plotting distributions for {len(numeric_columns)} numeric columns...")
        
        n_cols = len(numeric_columns)
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=self.figure_size_multivariate)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_columns):
            try:
                sns.histplot(
                    data=df, 
                    x=col, 
                    kde=self.kde, 
                    ax=axes[idx],
                    color='purple',
                    alpha=0.7
                )
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                self.logger.error(f"Error plotting {col}: {e}")
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_file = Path(self.output_dir / "numeric_distributions.png")
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        mlflow.log_artifact(output_file)
        
        self.logger.info(f"Saved: {output_file}")


    def plot_boxplots(self,
        df: pd.DataFrame
    ) -> None:
        """
        Plot boxplots for outlier detection.
        
        Args:
            df: Input DataFrame
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.logger.info(f"Plotting boxplots for {len(numeric_columns)} columns...")
        
        n_cols = len(numeric_columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=self.figure_size_multivariate)
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
                self.logger.error(f"Error plotting {col}: {e}")
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_file = self.output_dir / "boxplots_outliers.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        mlflow.log_artifact(output_file)
        
        self.logger.info(f"Saved: {output_file}")


    def plot_categorical_distributions(self,
        df: pd.DataFrame,
    ) -> None:
        """
        Plot count plots for categorical columns.
        
        Args:
            df: Input DataFrame
        """
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.logger.info(f"Plotting categorical distributions for {len(categorical_columns)} columns...")
        
        n_cols = len(categorical_columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=self.figure_size_univariate)
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
                self.logger.error(f"Error plotting {col}: {e}")
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_file = self.output_dir / "categorical_distributions.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        mlflow.log_artifact(output_file)
        
        self.logger.info(f"Saved: {output_file}")


    def plot_correlation_heatmap(self,
        df: pd.DataFrame
    ) -> None:
        """
        Plot correlation heatmap.
        
        Args:
            df: Input DataFrame
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        method = self.config['visualization']['heatmap'].get('method','spearman')
        self.logger.info(f"Plotting correlation heatmap ({method} method)...")
        
        if len(numeric_columns) < 2:
            self.logger.warning("Need at least 2 numeric columns for correlation")
            return
        
        try:
            corr = df[numeric_columns].corr(method=method)
            
            fig, ax = plt.subplots(figsize=self.figure_size_univariate)
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
            output_file = self.output_dir / f"correlation_heatmap_{method}.png"
            plt.savefig(output_file, dpi=self.pi, bbox_inches='tight')
            plt.close(fig)

            mlflow.log_artifact(output_file)
            
            self.logger.info(f"Saved: {output_file}")
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {e}")


    def plot_target_distribution(self,
        df: pd.DataFrame,
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
        target_column = self.config['data'].get('target_column')
        self.logger.info(f"Plotting target distribution for '{target_column}'...")
        
        try:
            fig, ax = plt.subplots(figsize=(8,6))
            
            colors = ['#2ecc71', '#e74c3c']
            
            ax = sns.countplot(
                data=df,
                x=target_column,
                palette=colors,
                saturation=0.8
            )
            
            # Add percentages
            for container in ax.containers:
                ax.bar_label(container, label_type='edge')
            
            ax.set_title(f'Target Variable Distribution: {target_column}', 
                        fontweight='bold', fontsize=14, pad=20)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_xlabel(target_column, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            output_file = self.output_dir / "target_distribution.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

            mlflow.log_artifact(output_file)
            
            self.logger.info(f"Saved: {output_file}")
        except Exception as e:
            self.logger.error(f"Error plotting target distribution: {e}")


    def run_visualizations(self, df):
        '''Runs all visualizations and log plots to MLflow'''
        self.plot_numeric_distributions(df)
        self.plot_boxplots(df)
        self.plot_categorical_distributions(df)
        self.plot_correlation_heatmap(df)
        self.plot_target_distribution(df)