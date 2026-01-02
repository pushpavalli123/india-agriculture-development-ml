"""
Visualization utilities for India Agriculture & Development Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


class IndiaDataVisualizer:
    """Class for creating visualizations of India agriculture and development data"""
    
    def __init__(self, style=None, figsize=(12, 8)):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style (None for default)
            figsize: Default figure size
        """
        if style:
            try:
                plt.style.use(style)
            except OSError:
                # Fallback to default style if specified style not available
                pass
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        self.figsize = figsize
        
    def plot_state_comparison(self, df, x_col, y_col, title=None, save_path=None):
        """
        Create scatter plot comparing states on two metrics
        
        Args:
            df: DataFrame with state data
            x_col: Column for x-axis
            y_col: Column for y-axis
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        scatter = ax.scatter(df[x_col], df[y_col], s=100, alpha=0.6)
        
        # Add state labels
        if 'State' in df.columns:
            for idx, row in df.iterrows():
                ax.annotate(row['State'], (row[x_col], row[y_col]), 
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.set_title(title or f'{x_col} vs {y_col} by State')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df, title='Correlation Heatmap', save_path=None):
        """
        Create correlation heatmap
        
        Args:
            df: DataFrame with numeric columns
            title: Plot title
            save_path: Path to save the plot
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(title, fontsize=16, pad=20)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_top_states(self, df, column, top_n=10, title=None, save_path=None):
        """
        Plot top N states by a metric
        
        Args:
            df: DataFrame with state data
            column: Column to rank by
            top_n: Number of top states to show
            title: Plot title
            save_path: Path to save the plot
        """
        top_states = df.nlargest(top_n, column)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.barh(range(len(top_states)), top_states[column])
        ax.set_yticks(range(len(top_states)))
        ax.set_yticklabels(top_states['State'] if 'State' in top_states.columns else top_states.index)
        ax.set_xlabel(column.replace('_', ' ').title())
        ax.set_title(title or f'Top {top_n} States by {column.replace("_", " ").title()}')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(top_states.iterrows()):
            ax.text(row[column], i, f' {row[column]:,.0f}', 
                   va='center', fontsize=9)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_distribution(self, df, columns, title=None, save_path=None):
        """
        Plot distribution of metrics
        
        Args:
            df: DataFrame
            columns: List of columns to plot
            title: Plot title
            save_path: Path to save the plot
        """
        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            ax = axes[i]
            df[col].hist(bins=20, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=16, y=1.02)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_map(self, df, value_column, title=None, save_path=None):
        """
        Create interactive choropleth map (requires plotly)
        
        Args:
            df: DataFrame with state data
            value_column: Column to visualize
            title: Plot title
            save_path: Path to save HTML file
        """
        # Note: This is a simplified version. For actual India map,
        # you would need GeoJSON data for Indian states
        
        fig = px.bar(
            df.sort_values(value_column, ascending=True),
            x=value_column,
            y='State' if 'State' in df.columns else df.index,
            orientation='h',
            title=title or f'{value_column.replace("_", " ").title()} by State',
            labels={value_column: value_column.replace('_', ' ').title()}
        )
        
        fig.update_layout(
            height=800,
            showlegend=False
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        
        return fig
    
    def plot_agriculture_vs_development(self, df, save_path=None):
        """
        Create comprehensive agriculture vs development visualization
        
        Args:
            df: Merged DataFrame
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Agricultural GDP vs Total GDP
        if 'Agricultural_GDP_crores' in df.columns and 'Total_GDP_crores' in df.columns:
            axes[0, 0].scatter(df['Total_GDP_crores'], df['Agricultural_GDP_crores'], 
                             s=100, alpha=0.6)
            axes[0, 0].set_xlabel('Total GDP (Crores)')
            axes[0, 0].set_ylabel('Agricultural GDP (Crores)')
            axes[0, 0].set_title('Agricultural GDP vs Total GDP')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. HDI vs Agricultural Productivity
        if 'HDI' in df.columns and 'Rice_Yield_per_ha' in df.columns:
            axes[0, 1].scatter(df['HDI'], df['Rice_Yield_per_ha'], s=100, alpha=0.6)
            axes[0, 1].set_xlabel('Human Development Index')
            axes[0, 1].set_ylabel('Rice Yield (tons/ha)')
            axes[0, 1].set_title('HDI vs Agricultural Productivity')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Literacy vs Irrigation Coverage
        if 'Literacy_Rate' in df.columns and 'Irrigation_Coverage' in df.columns:
            axes[1, 0].scatter(df['Literacy_Rate'], df['Irrigation_Coverage'], 
                             s=100, alpha=0.6)
            axes[1, 0].set_xlabel('Literacy Rate (%)')
            axes[1, 0].set_ylabel('Irrigation Coverage')
            axes[1, 0].set_title('Literacy vs Irrigation Coverage')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top states by combined score
        if 'HDI' in df.columns and 'Agricultural_GDP_crores' in df.columns:
            # Normalize and combine scores
            df_norm = df.copy()
            df_norm['HDI_norm'] = (df_norm['HDI'] - df_norm['HDI'].min()) / (df_norm['HDI'].max() - df_norm['HDI'].min())
            df_norm['Agri_GDP_norm'] = (df_norm['Agricultural_GDP_crores'] - df_norm['Agricultural_GDP_crores'].min()) / (df_norm['Agricultural_GDP_crores'].max() - df_norm['Agricultural_GDP_crores'].min())
            df_norm['Combined_Score'] = (df_norm['HDI_norm'] + df_norm['Agri_GDP_norm']) / 2
            
            top_states = df_norm.nlargest(10, 'Combined_Score')
            axes[1, 1].barh(range(len(top_states)), top_states['Combined_Score'])
            axes[1, 1].set_yticks(range(len(top_states)))
            axes[1, 1].set_yticklabels(top_states['State'] if 'State' in top_states.columns else top_states.index)
            axes[1, 1].set_xlabel('Combined Score (HDI + Agri GDP)')
            axes[1, 1].set_title('Top 10 States by Combined Development Score')
            axes[1, 1].invert_yaxis()
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Agriculture & Development Analysis', fontsize=16, y=0.995)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def create_dashboard(self, df, save_path='results/dashboard.html'):
        """
        Create an interactive dashboard
        
        Args:
            df: Merged DataFrame
            save_path: Path to save HTML dashboard
        """
        # This is a simplified dashboard. For a full dashboard,
        # you would use plotly dash or streamlit
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Agricultural GDP Distribution', 'HDI Distribution',
                          'Top States by Agricultural GDP', 'Top States by HDI'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        if 'Agricultural_GDP_crores' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['Agricultural_GDP_crores'], name='Agri GDP'),
                row=1, col=1
            )
        
        if 'HDI' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['HDI'], name='HDI'),
                row=1, col=2
            )
        
        if 'Agricultural_GDP_crores' in df.columns and 'State' in df.columns:
            top_agri = df.nlargest(10, 'Agricultural_GDP_crores')
            fig.add_trace(
                go.Bar(x=top_agri['State'], y=top_agri['Agricultural_GDP_crores'],
                      name='Top Agri GDP'),
                row=2, col=1
            )
        
        if 'HDI' in df.columns and 'State' in df.columns:
            top_hdi = df.nlargest(10, 'HDI')
            fig.add_trace(
                go.Bar(x=top_hdi['State'], y=top_hdi['HDI'], name='Top HDI'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="India Agriculture & Development Dashboard")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Dashboard saved to {save_path}")
        
        return fig


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append('..')
    from data_loader import IndiaDataLoader
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    loader = IndiaDataLoader()
    preprocessor = DataPreprocessor()
    
    agri_data = loader.load_agriculture_data()
    dev_data = loader.load_development_data()
    merged_data = loader.merge_data(agri_data, dev_data)
    processed_data = preprocessor.create_features(merged_data)
    
    # Create visualizations
    viz = IndiaDataVisualizer()
    
    # Correlation heatmap
    viz.plot_correlation_heatmap(processed_data, save_path='results/correlation_heatmap.png')
    
    # Agriculture vs Development
    viz.plot_agriculture_vs_development(processed_data, save_path='results/agri_dev_analysis.png')
    
    # Top states
    if 'Agricultural_GDP_crores' in processed_data.columns:
        viz.plot_top_states(processed_data, 'Agricultural_GDP_crores', 
                          save_path='results/top_agri_states.png')
    
    # Dashboard
    viz.create_dashboard(processed_data)

