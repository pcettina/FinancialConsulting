#!/usr/bin/env python3
"""
Custom Gaussian Process Factor Analysis (GPFA) Implementation
============================================================

A custom implementation of GPFA for analyzing stock price trajectories.
This implementation uses:
- PCA for initial dimensionality reduction
- Gaussian Process regression for smooth latent trajectories
- Visualization of neural-like trajectories in financial data

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class CustomGPFA:
    def __init__(self, n_factors=3, n_pca_components=10, random_state=42):
        """
        Initialize Custom GPFA
        
        Parameters:
        n_factors (int): Number of latent factors to extract
        n_pca_components (int): Number of PCA components for initial reduction
        random_state (int): Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        
        # Initialize components
        self.pca = PCA(n_components=n_pca_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.gp_models = []
        self.latent_trajectories = None
        self.time_points = None
        
    def prepare_data(self, stock_data):
        """
        Prepare stock data for GPFA analysis with z-scoring normalization
        
        Parameters:
        stock_data (pd.DataFrame): DataFrame with columns ['datetime', 'ticker', 'price']
        
        Returns:
        np.array: Matrix of shape (n_stocks, n_timepoints)
        """
        print("Preparing data for GPFA analysis...")
        
        # Convert datetime to pandas datetime
        stock_data['datetime'] = pd.to_datetime(stock_data['datetime'])
        
        # Pivot data to get stocks as rows and time as columns
        price_matrix = stock_data.pivot(index='ticker', columns='datetime', values='price')
        
        # Sort by datetime
        price_matrix = price_matrix.sort_index(axis=1)
        
        # Handle missing values (forward fill, then backward fill)
        price_matrix = price_matrix.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        
        # Calculate returns instead of absolute prices for better analysis
        returns_matrix = price_matrix.pct_change(axis=1).fillna(0)
        
        # Z-score each stock's returns (normalize across time for each stock)
        print("Z-scoring each stock's returns for normalization...")
        zscored_matrix = returns_matrix.copy()
        
        for i in range(returns_matrix.shape[0]):
            stock_returns = returns_matrix.iloc[i, :]
            mean_return = stock_returns.mean()
            std_return = stock_returns.std()
            
            if std_return > 0:  # Avoid division by zero
                zscored_matrix.iloc[i, :] = (stock_returns - mean_return) / std_return
            else:
                zscored_matrix.iloc[i, :] = 0  # If no variation, set to 0
        
        print(f"Data shape: {zscored_matrix.shape} (stocks × timepoints)")
        print(f"Time range: {zscored_matrix.columns[0]} to {zscored_matrix.columns[-1]}")
        print(f"Z-scored data - Mean: {zscored_matrix.values.mean():.4f}, Std: {zscored_matrix.values.std():.4f}")
        
        return zscored_matrix.values, zscored_matrix.columns
    
    def fit(self, stock_data):
        """
        Fit GPFA model to stock data
        
        Parameters:
        stock_data (pd.DataFrame): Stock price data
        """
        print("Fitting GPFA model...")
        
        # Prepare data
        self.data_matrix, self.time_points = self.prepare_data(stock_data)
        self.n_stocks, self.n_timepoints = self.data_matrix.shape
        
        # Step 1: Standardize data
        print("Step 1: Standardizing data...")
        self.data_scaled = self.scaler.fit_transform(self.data_matrix.T).T
        
        # Step 2: Initial dimensionality reduction with PCA
        print("Step 2: Performing initial PCA...")
        self.pca_results = self.pca.fit_transform(self.data_scaled.T)
        print(f"PCA explained variance: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        # Step 3: Fit Gaussian Process models for each factor
        print("Step 3: Fitting Gaussian Process models...")
        self._fit_gaussian_processes()
        
        # Step 4: Extract smooth latent trajectories
        print("Step 4: Extracting latent trajectories...")
        self._extract_latent_trajectories()
        
        print("GPFA fitting completed!")
        
    def _fit_gaussian_processes(self):
        """Fit Gaussian Process models for each latent factor"""
        # Create time index for GP fitting
        time_index = np.arange(self.n_timepoints).reshape(-1, 1)
        
        # Define kernel for smooth trajectories
        kernel = ConstantKernel(1.0) * RBF(length_scale=50.0) + WhiteKernel(noise_level=0.1)
        
        self.gp_models = []
        for i in range(self.n_factors):
            print(f"  Fitting GP for factor {i+1}/{self.n_factors}...")
            
            # Use the first n_factors PCA components
            if i < self.pca_results.shape[1]:
                y = self.pca_results[:, i]
            else:
                # If we need more factors than PCA components, use random projections
                y = np.random.randn(self.n_timepoints)
            
            # Fit Gaussian Process
            gp = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
            gp.fit(time_index, y)
            
            self.gp_models.append(gp)
    
    def _extract_latent_trajectories(self):
        """Extract smooth latent trajectories using fitted GP models"""
        time_index = np.arange(self.n_timepoints).reshape(-1, 1)
        
        self.latent_trajectories = np.zeros((self.n_factors, self.n_timepoints))
        
        for i, gp in enumerate(self.gp_models):
            # Predict smooth trajectory
            trajectory, _ = gp.predict(time_index, return_std=True)
            self.latent_trajectories[i, :] = trajectory
    
    def get_latent_trajectories(self):
        """Get the extracted latent trajectories"""
        return self.latent_trajectories
    
    def get_factor_loadings(self):
        """Get factor loadings (how each stock contributes to each factor)"""
        # Use PCA loadings as initial factor loadings
        return self.pca.components_[:self.n_factors, :]
    
    def reconstruct_data(self):
        """Reconstruct data using latent trajectories and factor loadings"""
        loadings = self.get_factor_loadings()
        reconstructed = loadings.T @ self.latent_trajectories
        
        # Transform back to original scale
        reconstructed = self.scaler.inverse_transform(reconstructed.T).T
        
        return reconstructed
    
    def calculate_reconstruction_error(self):
        """Calculate reconstruction error"""
        reconstructed = self.reconstruct_data()
        mse = np.mean((self.data_matrix - reconstructed) ** 2)
        return mse
    
    def plot_latent_trajectories(self, save_path=None):
        """Plot the extracted latent trajectories"""
        print("Creating latent trajectory plots...")
        
        fig, axes = plt.subplots(self.n_factors, 1, figsize=(12, 3*self.n_factors))
        if self.n_factors == 1:
            axes = [axes]
        
        for i in range(self.n_factors):
            axes[i].plot(self.time_points, self.latent_trajectories[i, :], 
                        linewidth=2, color=f'C{i}', label=f'Factor {i+1}')
            axes[i].set_title(f'Latent Factor {i+1} Trajectory')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Factor Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_3d_trajectory(self, factors=[0, 1, 2], save_path=None, sample_rate=50):
        """
        Plot 3D trajectory of selected factors with subsampling for better visualization
        
        Parameters:
        factors (list): Which factors to plot [x, y, z]
        save_path (str): Path to save the HTML file
        sample_rate (int): Plot every Nth point (e.g., 50 means plot every 50th point)
        """
        print(f"Creating 3D trajectory plot (sampling every {sample_rate}th point)...")
        
        # Subsample the trajectory points
        indices = np.arange(0, len(self.time_points), sample_rate)
        if indices[-1] != len(self.time_points) - 1:  # Add the last point if not included
            indices = np.append(indices, len(self.time_points) - 1)
        
        x_sampled = self.latent_trajectories[factors[0], indices]
        y_sampled = self.latent_trajectories[factors[1], indices]
        z_sampled = self.latent_trajectories[factors[2], indices]
        time_sampled = self.time_points[indices]
        
        fig = go.Figure()
        
        # Create 3D scatter plot with sampled points
        fig.add_trace(go.Scatter3d(
            x=x_sampled,
            y=y_sampled,
            z=z_sampled,
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=6, color='red'),
            name=f'Latent Trajectory (every {sample_rate}th point)',
            hovertemplate='Time: %{text}<br>' +
                         f'Factor {factors[0]+1}: %{{x:.3f}}<br>' +
                         f'Factor {factors[1]+1}: %{{y:.3f}}<br>' +
                         f'Factor {factors[2]+1}: %{{z:.3f}}<br>' +
                         '<extra></extra>',
            text=[str(t) for t in time_sampled]
        ))
        
        # Add time annotations (fewer, more spaced out)
        annotation_spacing = max(1, len(indices) // 8)  # Show ~8 annotations
        for i in range(0, len(indices), annotation_spacing):
            if i < len(indices):
                fig.add_trace(go.Scatter3d(
                    x=[x_sampled[i]],
                    y=[y_sampled[i]],
                    z=[z_sampled[i]],
                    mode='markers+text',
                    marker=dict(size=10, color='green', symbol='diamond'),
                    text=[f't{indices[i]}'],
                    textposition='middle center',
                    showlegend=False,
                    hovertemplate=f'Time index: {indices[i]}<br>Time: {time_sampled[i]}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'3D Latent Trajectory (Factors {factors[0]+1}, {factors[1]+1}, {factors[2]+1})',
            scene=dict(
                xaxis_title=f'Factor {factors[0]+1}',
                yaxis_title=f'Factor {factors[1]+1}',
                zaxis_title=f'Factor {factors[2]+1}'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_3d_trajectory_by_day(self, factors=[0, 1, 2], save_path=None):
        """Plot 3D trajectory separated by trading days"""
        print("Creating 3D trajectory plot separated by day...")
        
        # Convert time points to pandas datetime for easier day extraction
        time_dt = pd.to_datetime(self.time_points)
        
        # Get unique days
        unique_days = time_dt.date.unique()
        colors = px.colors.qualitative.Set3  # Use a color palette
        
        fig = go.Figure()
        
        for day_idx, day in enumerate(unique_days):
            # Get indices for this day
            day_mask = time_dt.date == day
            day_indices = np.where(day_mask)[0]
            
            if len(day_indices) > 1:  # Only plot if we have multiple points for the day
                # Extract trajectory for this day
                x_day = self.latent_trajectories[factors[0], day_indices]
                y_day = self.latent_trajectories[factors[1], day_indices]
                z_day = self.latent_trajectories[factors[2], day_indices]
                
                # Create line trace for this day
                fig.add_trace(go.Scatter3d(
                    x=x_day,
                    y=y_day,
                    z=z_day,
                    mode='lines+markers',
                    line=dict(color=colors[day_idx % len(colors)], width=4),
                    marker=dict(size=6, color=colors[day_idx % len(colors)]),
                    name=f'Day {day}',
                    hovertemplate=f'Day: {day}<br>' +
                                 f'Factor {factors[0]+1}: %{{x:.3f}}<br>' +
                                 f'Factor {factors[1]+1}: %{{y:.3f}}<br>' +
                                 f'Factor {factors[2]+1}: %{{z:.3f}}<br>' +
                                 '<extra></extra>'
                ))
                
                # Add start and end markers for each day
                fig.add_trace(go.Scatter3d(
                    x=[x_day[0]],
                    y=[y_day[0]],
                    z=[z_day[0]],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='diamond'),
                    name=f'Start {day}',
                    showlegend=False,
                    hovertemplate=f'Start of {day}<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=[x_day[-1]],
                    y=[y_day[-1]],
                    z=[z_day[-1]],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='diamond'),
                    name=f'End {day}',
                    showlegend=False,
                    hovertemplate=f'End of {day}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'3D Latent Trajectory by Trading Day<br>(Factors {factors[0]+1}, {factors[1]+1}, {factors[2]+1})',
            scene=dict(
                xaxis_title=f'Factor {factors[0]+1}',
                yaxis_title=f'Factor {factors[1]+1}',
                zaxis_title=f'Factor {factors[2]+1}',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=700,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_factor_loadings(self, save_path=None):
        """Plot factor loadings heatmap"""
        print("Creating factor loadings heatmap...")
        
        loadings = self.get_factor_loadings()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings, 
                   cmap='RdBu_r', 
                   center=0, 
                   square=True, 
                   cbar_kws={'label': 'Loading Value'},
                   xticklabels=False,
                   yticklabels=[f'Factor {i+1}' for i in range(self.n_factors)])
        
        plt.title('Factor Loadings (Stock Contributions to Each Factor)')
        plt.xlabel('Stocks')
        plt.ylabel('Latent Factors')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_reconstruction_quality(self, save_path=None):
        """Plot original vs reconstructed data for a few stocks"""
        print("Creating reconstruction quality plot...")
        
        reconstructed = self.reconstruct_data()
        
        # Select a few stocks to plot
        n_stocks_to_plot = min(6, self.n_stocks)
        stock_indices = np.linspace(0, self.n_stocks-1, n_stocks_to_plot, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, stock_idx in enumerate(stock_indices):
            axes[i].plot(self.time_points, self.data_matrix[stock_idx, :], 
                        'b-', alpha=0.7, label='Original', linewidth=1)
            axes[i].plot(self.time_points, reconstructed[stock_idx, :], 
                        'r--', alpha=0.8, label='Reconstructed', linewidth=1)
            axes[i].set_title(f'Stock {stock_idx+1}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Returns')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, save_path='gpfa_report.txt'):
        """Generate comprehensive GPFA analysis report"""
        print("Generating GPFA analysis report...")
        
        report = []
        report.append("=" * 60)
        report.append("CUSTOM GPFA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset information
        report.append("DATASET INFORMATION:")
        report.append(f"Number of stocks: {self.n_stocks}")
        report.append(f"Number of timepoints: {self.n_timepoints}")
        report.append(f"Time range: {self.time_points[0]} to {self.time_points[-1]}")
        report.append("")
        
        # Model parameters
        report.append("MODEL PARAMETERS:")
        report.append(f"Number of latent factors: {self.n_factors}")
        report.append(f"Number of PCA components: {self.n_pca_components}")
        report.append(f"PCA explained variance: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        report.append("")
        
        # Reconstruction quality
        mse = self.calculate_reconstruction_error()
        report.append("RECONSTRUCTION QUALITY:")
        report.append(f"Mean Squared Error: {mse:.6f}")
        report.append(f"Root Mean Squared Error: {np.sqrt(mse):.6f}")
        report.append("")
        
        # Factor analysis
        report.append("FACTOR ANALYSIS:")
        loadings = self.get_factor_loadings()
        for i in range(self.n_factors):
            factor_variance = np.var(self.latent_trajectories[i, :])
            report.append(f"Factor {i+1}: Variance = {factor_variance:.4f}")
        report.append("")
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {save_path}")
        return '\n'.join(report)

def main():
    """Main GPFA analysis pipeline"""
    
    print("🚀 CUSTOM GPFA ANALYSIS FOR STOCK DATA")
    print("=" * 50)
    
    # Load stock data
    print("Loading stock data...")
    stock_data = pd.read_csv('stock_prices.csv')
    print(f"Loaded {len(stock_data)} data points")
    
    # Initialize and fit GPFA
    gpfa = CustomGPFA(n_factors=3, n_pca_components=10, random_state=42)
    gpfa.fit(stock_data)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    gpfa.plot_3d_trajectory(save_path='3d_trajectory.html', sample_rate=100)
    # gpfa.plot_latent_trajectories('latent_trajectories.png')
    # gpfa.plot_3d_trajectory_by_day(save_path='3d_trajectory_by_day.html')
    # gpfa.plot_factor_loadings('factor_loadings.png')
    # gpfa.plot_reconstruction_quality('reconstruction_quality.png')
    
    # Generate report
    report = gpfa.generate_report()
    print("\n" + "=" * 50)
    print("GPFA ANALYSIS COMPLETE!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - 3d_trajectory.html (3D interactive trajectory)")
    print("  - gpfa_report.txt (comprehensive analysis report)")
    
    return gpfa

if __name__ == "__main__":
    gpfa = main() 