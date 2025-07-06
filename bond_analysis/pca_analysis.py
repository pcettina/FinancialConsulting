import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class BondPCAnalysis:
    def __init__(self, data_path='bond_trading_data.csv'):
        """
        Initialize PCA analysis for bond trading data
        
        Parameters:
        data_path (str): Path to the bond trading data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.pca = None
        self.pca_results = None
        self.feature_columns = None
        
    def load_data(self):
        """Load and prepare the bond trading data"""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded data with shape: {self.df.shape}")
        return self.df
    
    def prepare_features(self, include_derived=True):
        """
        Prepare features for PCA analysis
        
        Parameters:
        include_derived (bool): Whether to include derived features
        """
        # Select numerical features for PCA
        base_features = ['duration', 'yield', 'base_coupon', 'price', 'ytm', 'ytw']
        
        if include_derived:
            derived_features = ['price_to_par', 'coupon_yield_spread', 'ytm_ytw_spread',
                              'moodys_numeric', 'sp_numeric', 'fitch_numeric']
            self.feature_columns = base_features + derived_features
        else:
            self.feature_columns = base_features
        
        # Create feature matrix
        X = self.df[self.feature_columns].copy()
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        print(f"Prepared {len(self.feature_columns)} features for PCA analysis")
        print(f"Features: {self.feature_columns}")
        
        return X
    
    def perform_pca(self, n_components=None, random_state=42):
        """
        Perform PCA analysis
        
        Parameters:
        n_components (int): Number of components to keep (None for all)
        random_state (int): Random seed for reproducibility
        """
        # Prepare features
        X = self.prepare_features()
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.pca_results = self.pca.fit_transform(X_scaled)
        
        # Create results DataFrame
        pca_df = pd.DataFrame(
            self.pca_results,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add original features and target
        for col in self.feature_columns:
            pca_df[col] = X[col]
        pca_df['trade_rating'] = self.df['trade_rating']
        pca_df['cusip'] = self.df['cusip']
        
        print(f"PCA completed with {n_components} components")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.cumsum(self.pca.explained_variance_ratio_)}")
        
        return pca_df
    
    def plot_explained_variance(self, save_path=None):
        """Plot explained variance and cumulative explained variance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance ratio
        ax1.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                self.pca.explained_variance_ratio_, 'bo-')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance Ratio by Principal Component')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumulative_var = np.cumsum(self.pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
        ax2.set_xlabel('Number of Principal Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal line at 0.95
        ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% Variance')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary
        n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
        print(f"Number of components needed for 95% variance: {n_components_95}")
        print(f"Number of components needed for 90% variance: {np.argmax(cumulative_var >= 0.90) + 1}")
    
    def plot_feature_importance(self, n_components=5, save_path=None):
        """Plot feature importance (loadings) for top principal components"""
        loadings = self.pca.components_[:n_components]
        
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            # Sort features by absolute loading value
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'loading': loadings[i]
            })
            feature_importance['abs_loading'] = abs(feature_importance['loading'])
            feature_importance = feature_importance.sort_values('abs_loading', ascending=False)
            
            # Create horizontal bar plot
            colors = ['red' if x < 0 else 'blue' for x in feature_importance['loading']]
            axes[i].barh(feature_importance['feature'], feature_importance['loading'], color=colors)
            axes[i].set_title(f'PC{i+1} Feature Loadings (Explained Variance: {self.pca.explained_variance_ratio_[i]:.3f})')
            axes[i].set_xlabel('Loading Value')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pca_scatter(self, pc1=1, pc2=2, color_by='trade_rating', save_path=None):
        """Create scatter plot of PCA results"""
        fig = px.scatter(
            x=self.pca_results[:, pc1-1],
            y=self.pca_results[:, pc2-1],
            color=self.df[color_by],
            title=f'PCA Scatter Plot: PC{pc1} vs PC{pc2}',
            labels={'x': f'PC{pc1}', 'y': f'PC{pc2}', 'color': color_by},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_3d_pca(self, pc1=1, pc2=2, pc3=3, color_by='trade_rating', save_path=None):
        """Create 3D scatter plot of PCA results"""
        fig = px.scatter_3d(
            x=self.pca_results[:, pc1-1],
            y=self.pca_results[:, pc2-1],
            z=self.pca_results[:, pc3-1],
            color=self.df[color_by],
            title=f'3D PCA Scatter Plot: PC{pc1}, PC{pc2}, PC{pc3}',
            labels={'x': f'PC{pc1}', 'y': f'PC{pc2}', 'z': f'PC{pc3}', 'color': color_by},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            width=800,
            height=600,
            scene=dict(
                xaxis_title=f'PC{pc1}',
                yaxis_title=f'PC{pc2}',
                zaxis_title=f'PC{pc3}'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def correlation_analysis(self):
        """Analyze correlations between original features and principal components"""
        # Create correlation matrix
        pca_df = pd.DataFrame(self.pca_results, columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
        
        # Add original features
        for col in self.feature_columns:
            pca_df[col] = self.df[col]
        
        # Calculate correlations
        correlations = pca_df.corr()
        
        # Focus on correlations between PCs and original features
        pc_cols = [f'PC{i+1}' for i in range(min(5, self.pca.n_components_))]
        feature_correlations = correlations.loc[pc_cols, self.feature_columns]
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(feature_correlations, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.3f')
        plt.title('Correlation between Principal Components and Original Features')
        plt.tight_layout()
        plt.show()
        
        return feature_correlations
    
    def get_top_features_by_pc(self, n_top=3):
        """Get top features contributing to each principal component"""
        loadings = self.pca.components_
        feature_importance = {}
        
        for i in range(min(5, len(loadings))):
            # Get absolute loadings for this PC
            pc_loadings = abs(loadings[i])
            
            # Get indices of top features
            top_indices = np.argsort(pc_loadings)[-n_top:][::-1]
            
            # Get feature names and their loadings
            top_features = [(self.feature_columns[idx], loadings[i][idx]) for idx in top_indices]
            feature_importance[f'PC{i+1}'] = top_features
        
        return feature_importance
    
    def generate_report(self, save_path='pca_report.txt'):
        """Generate a comprehensive PCA analysis report"""
        report = []
        report.append("=" * 60)
        report.append("BOND TRADING PCA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset information
        report.append("DATASET INFORMATION:")
        report.append(f"Total samples: {len(self.df)}")
        report.append(f"Features analyzed: {len(self.feature_columns)}")
        report.append(f"Features: {', '.join(self.feature_columns)}")
        report.append("")
        
        # PCA results
        report.append("PCA RESULTS:")
        report.append(f"Total components: {self.pca.n_components_}")
        report.append("")
        
        # Explained variance
        report.append("EXPLAINED VARIANCE:")
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            report.append(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        report.append("")
        
        # Cumulative variance
        cumulative_var = np.cumsum(self.pca.explained_variance_ratio_)
        report.append("CUMULATIVE EXPLAINED VARIANCE:")
        for i, var in enumerate(cumulative_var):
            report.append(f"PC1-PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        report.append("")
        
        # Top features by PC
        report.append("TOP FEATURES BY PRINCIPAL COMPONENT:")
        feature_importance = self.get_top_features_by_pc()
        for pc, features in feature_importance.items():
            report.append(f"{pc}:")
            for feature, loading in features:
                report.append(f"  {feature}: {loading:.4f}")
            report.append("")
        
        # Recommendations
        n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
        n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
        
        report.append("RECOMMENDATIONS:")
        report.append(f"For 95% variance retention: Use {n_components_95} components")
        report.append(f"For 90% variance retention: Use {n_components_90} components")
        report.append("")
        
        # Write report to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"PCA report saved to {save_path}")
        return '\n'.join(report)

if __name__ == "__main__":
    # Example usage
    pca_analyzer = BondPCAnalysis()
    pca_analyzer.load_data()
    pca_results = pca_analyzer.perform_pca()
    
    # Generate visualizations
    pca_analyzer.plot_explained_variance()
    pca_analyzer.plot_feature_importance()
    pca_analyzer.plot_pca_scatter()
    pca_analyzer.correlation_analysis()
    
    # Generate report
    report = pca_analyzer.generate_report()
    print(report) 