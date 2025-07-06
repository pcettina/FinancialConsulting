# Real-time GPFA Stock Price Predictor
"""
Real-time GPFA Stock Price Prediction System
Implementation following the step-by-step guide

Phase 1: Foundation - Data Infrastructure and Basic Real-time Feed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import schedule
import warnings
warnings.filterwarnings('ignore')

# Financial data libraries
import yfinance as yf
import ta

# ML libraries
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Real-time processing
import requests
import json
from typing import Dict, List, Tuple, Optional
import logging

# Visualization
from realtime_visualization import RealTimeVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeDataFeed:
    """
    Real-time data feed for stock prices using yfinance
    """
    
    def __init__(self, symbols: List[str], update_interval: int = 60):
        """
        Initialize real-time data feed
        
        Args:
            symbols: List of stock symbols to track
            update_interval: Update interval in seconds
        """
        self.symbols = symbols
        self.update_interval = update_interval
        self.data_buffer = {symbol: [] for symbol in symbols}
        self.last_update = None
        
        logger.info(f"Initialized data feed for {len(symbols)} symbols: {symbols}")
    
    def fetch_real_time_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch real-time data for all symbols using yfinance
        
        Returns:
            Dictionary of DataFrames for each symbol
        """
        data = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get 1-day data with 1-minute intervals
                hist_data = ticker.history(period='1d', interval='1m')
                
                if not hist_data.empty:
                    data[symbol] = hist_data
                    logger.debug(f"Fetched {len(hist_data)} data points for {symbol}")
                else:
                    logger.warning(f"No data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        self.last_update = datetime.now()
        return data
    
    def update_buffer(self, new_data: Dict[str, pd.DataFrame]):
        """
        Update the data buffer with new data
        
        Args:
            new_data: New data from fetch_real_time_data
        """
        for symbol, df in new_data.items():
            if symbol in self.data_buffer:
                # Append new data to buffer
                self.data_buffer[symbol].append(df)
                
                # Keep only last 1000 data points to prevent memory issues
                if len(self.data_buffer[symbol]) > 1000:
                    self.data_buffer[symbol] = self.data_buffer[symbol][-1000:]
        
        logger.debug(f"Updated data buffer for {len(new_data)} symbols")
    
    def get_latest_data(self) -> pd.DataFrame:
        """
        Get the latest data for all symbols as a combined DataFrame
        
        Returns:
            Combined DataFrame with latest prices for all symbols
        """
        latest_data = []
        
        for symbol in self.symbols:
            if symbol in self.data_buffer and self.data_buffer[symbol]:
                # Get the most recent data
                latest_df = self.data_buffer[symbol][-1]
                if not latest_df.empty:
                    # Get the last row (most recent data)
                    latest_row = latest_df.iloc[-1:].copy()
                    latest_row['Symbol'] = symbol
                    latest_data.append(latest_row)
        
        if latest_data:
            combined_df = pd.concat(latest_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

class RealTimeGPFA:
    """
    Real-time GPFA model for stock price prediction
    Extends the custom GPFA implementation for real-time updates
    """
    
    def __init__(self, n_factors: int = 5, update_frequency: str = '1min'):
        """
        Initialize real-time GPFA model
        
        Args:
            n_factors: Number of latent factors
            update_frequency: Update frequency ('1min', '5min', etc.)
        """
        self.n_factors = n_factors
        self.update_frequency = update_frequency
        self.pca = PCA(n_components=n_factors)
        self.gp_models = []
        self.factor_loadings = None
        self.latent_trajectories = None
        self.last_update = None
        
        # Initialize Gaussian Process models for each factor
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        for _ in range(n_factors):
            self.gp_models.append(GaussianProcessRegressor(kernel=kernel, random_state=42))
        
        logger.info(f"Initialized RealTimeGPFA with {n_factors} factors")
    
    def fit_model(self, data: pd.DataFrame) -> None:
        """
        Fit the GPFA model to historical data
        
        Args:
            data: Historical price data (symbols as columns, time as index)
        """
        try:
            # Prepare data for PCA
            price_data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Apply PCA to extract factors
            self.factor_loadings = self.pca.fit_transform(price_data.T)
            latent_data = self.pca.transform(price_data.T).T
            
            # Fit Gaussian Process models for each factor
            time_points = np.arange(len(latent_data[0])).reshape(-1, 1)
            
            for i, gp_model in enumerate(self.gp_models):
                gp_model.fit(time_points, latent_data[i])
            
            self.latent_trajectories = latent_data
            self.last_update = datetime.now()
            
            logger.info(f"Fitted GPFA model with {self.n_factors} factors")
            
        except Exception as e:
            logger.error(f"Error fitting GPFA model: {e}")
    
    def update_model(self, new_data: pd.DataFrame) -> None:
        """
        Incrementally update the GPFA model with new data
        
        Args:
            new_data: New price data to incorporate
        """
        try:
            # For now, we'll refit the entire model
            # In a production system, you'd implement incremental updates
            self.fit_model(new_data)
            
            logger.info("Updated GPFA model with new data")
            
        except Exception as e:
            logger.error(f"Error updating GPFA model: {e}")
    
    def predict_factors(self, time_horizon: int) -> np.ndarray:
        """
        Predict latent factors for a given time horizon
        
        Args:
            time_horizon: Number of time steps to predict ahead
            
        Returns:
            Predicted latent factors
        """
        if not self.gp_models:
            raise ValueError("Model not fitted yet")
        
        try:
            # Get current time point
            current_time = len(self.latent_trajectories[0]) if self.latent_trajectories is not None else 0
            future_times = np.arange(current_time, current_time + time_horizon).reshape(-1, 1)
            
            # Predict each factor
            predictions = []
            for gp_model in self.gp_models:
                pred, _ = gp_model.predict(future_times, return_std=True)
                predictions.append(pred)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting factors: {e}")
            return None
    
    def reconstruct_prices(self, latent_factors: np.ndarray) -> np.ndarray:
        """
        Reconstruct price predictions from latent factors
        
        Args:
            latent_factors: Predicted latent factors
            
        Returns:
            Reconstructed price predictions
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted yet")
        
        try:
            # Transform latent factors back to price space
            reconstructed = self.pca.inverse_transform(latent_factors.T)
            return reconstructed.T
            
        except Exception as e:
            logger.error(f"Error reconstructing prices: {e}")
            return None

class PredictionEnsemble:
    """
    Ensemble of prediction models for different time horizons with uncertainty quantification
    """
    
    def __init__(self, horizons: List[str] = ['1min', '5min', '15min', '1hour', '1day']):
        """
        Initialize prediction ensemble
        
        Args:
            horizons: List of prediction horizons
        """
        self.horizons = horizons
        self.models = {}
        self.feature_importance = {}
        self.uncertainty_models = {}
        
        # Initialize models for each horizon with optimized hyperparameters
        for horizon in horizons:
            self.models[horizon] = {
                'rf': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'xgb': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    enable_categorical=False,
                    missing=0
                ),
                'lgb': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True
                )
            }
            
            # Initialize uncertainty quantification models
            self.uncertainty_models[horizon] = {
                'rf': RandomForestRegressor(
                    n_estimators=50, 
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'xgb': xgb.XGBRegressor(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    enable_categorical=False,
                    missing=0
                ),
                'lgb': lgb.LGBMRegressor(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True
                )
            }
        
        logger.info(f"Initialized prediction ensemble with uncertainty quantification for {len(horizons)} horizons")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction models with robust data cleaning
        
        Args:
            data: Raw price data
            
        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame()
        
        # Ensure we have enough data for technical indicators
        min_data_required = 30
        
        for column in data.columns:
            if column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                # Basic price-based features with safety checks
                try:
                    # Returns with clipping to prevent extreme values
                    returns = data[column].pct_change()
                    returns = returns.clip(-0.5, 0.5)  # Clip to ±50% to prevent extreme values
                    features[f'{column}_return'] = returns
                    
                    # Moving averages with minimum window
                    if len(data) >= 5:
                        features[f'{column}_ma_5'] = data[column].rolling(5, min_periods=3).mean()
                    else:
                        features[f'{column}_ma_5'] = data[column].mean()
                    
                    if len(data) >= 20:
                        features[f'{column}_ma_20'] = data[column].rolling(20, min_periods=10).mean()
                    else:
                        features[f'{column}_ma_20'] = data[column].mean()
                    
                    # Standard deviation with safety
                    if len(data) >= 5:
                        std_values = data[column].rolling(5, min_periods=3).std()
                        # Clip std to prevent extreme values
                        std_values = std_values.clip(0, data[column].mean() * 0.5)
                        features[f'{column}_std_5'] = std_values
                    else:
                        features[f'{column}_std_5'] = 0
                    
                    # Technical indicators only for Close price with sufficient data
                    if column == 'Close' and len(data) >= min_data_required:
                        try:
                            # RSI with safety checks
                            rsi = ta.momentum.RSIIndicator(data[column], window=14).rsi()
                            rsi = rsi.clip(0, 100)  # RSI should be between 0-100
                            features['rsi'] = rsi
                            
                            # RSI-based features
                            features['rsi_overbought'] = (rsi > 70).astype(int)
                            features['rsi_oversold'] = (rsi < 30).astype(int)
                        except Exception as e:
                            logger.warning(f"RSI calculation failed: {e}")
                            features['rsi'] = 50  # Neutral RSI
                            features['rsi_overbought'] = 0
                            features['rsi_oversold'] = 0
                        
                        try:
                            # MACD with safety checks
                            macd = ta.trend.MACD(data[column])
                            macd_line = macd.macd()
                            macd_signal = macd.macd_signal()
                            macd_histogram = macd.macd_diff()
                            
                            # Clip MACD to prevent extreme values
                            macd_line = macd_line.clip(-data[column].mean() * 0.1, data[column].mean() * 0.1)
                            macd_signal = macd_signal.clip(-data[column].mean() * 0.1, data[column].mean() * 0.1)
                            macd_histogram = macd_histogram.clip(-data[column].mean() * 0.05, data[column].mean() * 0.05)
                            
                            features['macd'] = macd_line
                            features['macd_signal'] = macd_signal
                            features['macd_histogram'] = macd_histogram
                            features['macd_cross'] = (macd_line > macd_signal).astype(int)
                        except Exception as e:
                            logger.warning(f"MACD calculation failed: {e}")
                            features['macd'] = 0
                            features['macd_signal'] = 0
                            features['macd_histogram'] = 0
                            features['macd_cross'] = 0
                        
                        try:
                            # Bollinger Bands with safety checks
                            bb = ta.volatility.BollingerBands(data[column])
                            bb_upper = bb.bollinger_hband()
                            bb_lower = bb.bollinger_lband()
                            bb_middle = bb.bollinger_mavg()
                            
                            # Ensure bands are reasonable
                            bb_upper = bb_upper.clip(data[column] * 0.8, data[column] * 1.5)
                            bb_lower = bb_lower.clip(data[column] * 0.5, data[column] * 1.2)
                            
                            features['bb_upper'] = bb_upper
                            features['bb_lower'] = bb_lower
                            features['bb_middle'] = bb_middle
                            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
                            features['bb_position'] = (data[column] - bb_lower) / (bb_upper - bb_lower)
                        except Exception as e:
                            logger.warning(f"Bollinger Bands calculation failed: {e}")
                            features['bb_upper'] = data[column] * 1.1
                            features['bb_lower'] = data[column] * 0.9
                            features['bb_middle'] = data[column]
                            features['bb_width'] = 0.2
                            features['bb_position'] = 0.5
                        
                        try:
                            # Additional momentum indicators
                            # Stochastic Oscillator
                            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data[column])
                            stoch_k = stoch.stoch()
                            stoch_d = stoch.stoch_signal()
                            
                            features['stoch_k'] = stoch_k.clip(0, 100)
                            features['stoch_d'] = stoch_d.clip(0, 100)
                            features['stoch_cross'] = (stoch_k > stoch_d).astype(int)
                        except Exception as e:
                            logger.warning(f"Stochastic calculation failed: {e}")
                            features['stoch_k'] = 50
                            features['stoch_d'] = 50
                            features['stoch_cross'] = 0
                        
                        try:
                            # Williams %R
                            williams_r = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data[column]).williams_r()
                            features['williams_r'] = williams_r.clip(-100, 0)
                        except Exception as e:
                            logger.warning(f"Williams %R calculation failed: {e}")
                            features['williams_r'] = -50
                        
                        try:
                            # Price action features
                            features['price_range'] = (data['High'] - data['Low']) / data[column]
                            features['body_size'] = abs(data['Close'] - data['Open']) / data[column]
                            features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data[column]
                            features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data[column]
                        except Exception as e:
                            logger.warning(f"Price action features failed: {e}")
                            features['price_range'] = 0.02
                            features['body_size'] = 0.01
                            features['upper_shadow'] = 0.005
                            features['lower_shadow'] = 0.005
                            
                    elif column == 'Close':
                        # Fallback values for insufficient data
                        features['rsi'] = 50
                        features['rsi_overbought'] = 0
                        features['rsi_oversold'] = 0
                        features['macd'] = 0
                        features['macd_signal'] = 0
                        features['macd_histogram'] = 0
                        features['macd_cross'] = 0
                        features['bb_upper'] = data[column] * 1.1
                        features['bb_lower'] = data[column] * 0.9
                        features['bb_middle'] = data[column]
                        features['bb_width'] = 0.2
                        features['bb_position'] = 0.5
                        features['stoch_k'] = 50
                        features['stoch_d'] = 50
                        features['stoch_cross'] = 0
                        features['williams_r'] = -50
                        features['price_range'] = 0.02
                        features['body_size'] = 0.01
                        features['upper_shadow'] = 0.005
                        features['lower_shadow'] = 0.005
                        
                except Exception as e:
                    logger.error(f"Error processing features for {column}: {e}")
                    # Set safe default values
                    features[f'{column}_return'] = 0
                    features[f'{column}_ma_5'] = data[column].mean() if len(data) > 0 else 0
                    features[f'{column}_ma_20'] = data[column].mean() if len(data) > 0 else 0
                    features[f'{column}_std_5'] = 0
        
        # Robust NaN handling
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Final safety check - replace any remaining infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        # Additional clipping to prevent extreme values
        for col in features.columns:
            if features[col].dtype in ['float64', 'float32']:
                # Clip to reasonable ranges based on column type
                if 'return' in col:
                    features[col] = features[col].clip(-0.5, 0.5)
                elif 'rsi' in col:
                    features[col] = features[col].clip(0, 100)
                else:
                    # For other features, clip to ±3 standard deviations or reasonable bounds
                    mean_val = features[col].mean()
                    std_val = features[col].std()
                    if std_val > 0:
                        features[col] = features[col].clip(mean_val - 3*std_val, mean_val + 3*std_val)
        
        logger.info(f"Prepared {len(features.columns)} features from {len(data)} data points")
        
        # Final validation
        if self._validate_features(features):
            return features
        else:
            logger.error("Feature validation failed - returning empty DataFrame")
            return pd.DataFrame()
    
    def _validate_features(self, features: pd.DataFrame) -> bool:
        """
        Validate feature quality before model training
        
        Args:
            features: Feature DataFrame to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        try:
            # Check for infinite values
            if features.isin([np.inf, -np.inf]).any().any():
                logger.error("Features contain infinite values")
                return False
            
            # Check for NaN values
            if features.isna().any().any():
                logger.error("Features contain NaN values")
                return False
            
            # Check for reasonable value ranges
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32']:
                    # Check for extreme values
                    mean_val = features[col].mean()
                    std_val = features[col].std()
                    
                    if std_val > 0:
                        # Check if more than 10% of values are extreme outliers
                        extreme_mask = (features[col] < mean_val - 5*std_val) | (features[col] > mean_val + 5*std_val)
                        if extreme_mask.sum() > len(features) * 0.1:
                            logger.warning(f"Column {col} has many extreme values")
            
            logger.info("Feature validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False
    
    def train_models(self, data: pd.DataFrame, target_column: str = 'Close') -> None:
        """
        Train all models for all horizons
        
        Args:
            data: Historical data
            target_column: Target column for prediction
        """
        features = self.prepare_features(data)
        target = data[target_column]
        
        # Check if features are valid
        if features.empty:
            logger.error("No valid features generated - skipping model training")
            return
        
        # Track training success
        training_results = {}
        
        for horizon in self.horizons:
            logger.info(f"Training models for {horizon} horizon")
            training_results[horizon] = {}
            
            # Create target with appropriate lag
            horizon_minutes = self._parse_horizon(horizon)
            if horizon_minutes > 0:
                horizon_target = target.shift(-horizon_minutes)
            else:
                horizon_target = target
            
            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1) | horizon_target.isna())
            X = features[valid_idx]
            y = horizon_target[valid_idx]
            
            # Adjust minimum data requirements based on horizon
            min_data_required = {
                '1min': 50,
                '5min': 100,
                '15min': 200,
                '1hour': 500,
                '1day': 1000
            }
            
            min_required = min_data_required.get(horizon, 100)
            
            if len(X) < min_required:
                logger.warning(f"Insufficient data for {horizon} horizon: {len(X)} < {min_required}")
                continue
            
            logger.info(f"Training with {len(X)} samples for {horizon} horizon")
            
            # Train each model
            for model_name, model in self.models[horizon].items():
                try:
                    # Ensure we have enough data for this specific model
                    if len(X) < 10:  # Minimum required for any model
                        logger.warning(f"Insufficient data for {model_name} on {horizon}: {len(X)} samples")
                        training_results[horizon][model_name] = 'insufficient_data'
                        continue
                    
                    # Fit the model
                    model.fit(X, y)
                    
                    # Verify the model is properly fitted
                    if self._is_model_fitted(model, model_name):
                        training_results[horizon][model_name] = 'success'
                        logger.info(f"✅ Trained {model_name} for {horizon} horizon with {len(X)} samples")
                    else:
                        training_results[horizon][model_name] = 'failed_verification'
                        logger.error(f"❌ Model {model_name} for {horizon} failed verification after training")
                        
                except Exception as e:
                    training_results[horizon][model_name] = 'failed'
                    logger.error(f"❌ Error training {model_name} for {horizon}: {e}")
        
        # Log training summary
        logger.info("Training Summary:")
        for horizon, results in training_results.items():
            success_count = sum(1 for status in results.values() if status == 'success')
            failed_count = sum(1 for status in results.values() if status == 'failed')
            insufficient_count = sum(1 for status in results.values() if status == 'insufficient_data')
            verification_failed_count = sum(1 for status in results.values() if status == 'failed_verification')
            total_count = len(results)
            
            logger.info(f"  {horizon}: {success_count}/{total_count} models trained successfully")
            if failed_count > 0:
                logger.warning(f"    - {failed_count} models failed training")
            if insufficient_count > 0:
                logger.warning(f"    - {insufficient_count} models had insufficient data")
            if verification_failed_count > 0:
                logger.error(f"    - {verification_failed_count} models failed verification")
    
    def predict_all_horizons(self, features: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Make predictions for all horizons using all models
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary of predictions for each horizon and model
        """
        predictions = {}
        
        for horizon in self.horizons:
            predictions[horizon] = {}
            
            for model_name, model in self.models[horizon].items():
                try:
                    # Check if model is properly fitted
                    if self._is_model_fitted(model, model_name):
                        pred = model.predict(features.iloc[-1:])[0]
                        predictions[horizon][model_name] = pred
                    else:
                        logger.warning(f"Model {model_name} for {horizon} is not fitted")
                        predictions[horizon][model_name] = None
                except Exception as e:
                    logger.error(f"Error predicting with {model_name} for {horizon}: {e}")
                    predictions[horizon][model_name] = None
        
        return predictions
    
    def retrain_failed_models(self, data: pd.DataFrame, target_column: str = 'Close') -> None:
        """
        Retrain models that failed to fit properly with enhanced error handling
        
        Args:
            data: Historical data for training
            target_column: Target column for prediction
        """
        features = self.prepare_features(data)
        target = data[target_column]
        
        logger.info("Retraining failed models...")
        
        for horizon in self.horizons:
            horizon_minutes = self._parse_horizon(horizon)
            if horizon_minutes > 0:
                horizon_target = target.shift(-horizon_minutes)
            else:
                horizon_target = target
            
            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1) | horizon_target.isna())
            X = features[valid_idx]
            y = horizon_target[valid_idx]
            
            # Adjust minimum data requirements based on horizon
            min_data_required = {
                '1min': 100,
                '5min': 150,
                '15min': 250,
                '1hour': 600,
                '1day': 1200
            }
            
            min_required = min_data_required.get(horizon, 100)
            
            # Check if we have enough data for retraining
            if len(X) < min_required:
                logger.warning(f"Insufficient data for retraining {horizon} models: {len(X)} < {min_required}")
                continue
            
            logger.info(f"Retraining {horizon} models with {len(X)} samples")
            
            for model_name, model in self.models[horizon].items():
                # Check if model needs retraining
                if not self._is_model_fitted(model, model_name):
                    logger.info(f"Retraining {model_name} for {horizon} horizon")
                    try:
                        # Additional data preprocessing for specific models
                        if model_name == 'xgb':
                            # Ensure XGBoost gets clean data
                            X_clean = X.copy()
                            # Remove any remaining infinite values
                            X_clean = X_clean.replace([np.inf, -np.inf], 0)
                            # Ensure all values are finite
                            X_clean = X_clean.fillna(0)
                            model.fit(X_clean, y)
                        elif model_name == 'lgb':
                            # LightGBM specific preprocessing
                            X_clean = X.copy()
                            X_clean = X_clean.replace([np.inf, -np.inf], 0)
                            X_clean = X_clean.fillna(0)
                            # Ensure categorical features are handled properly
                            model.fit(X_clean, y)
                        else:
                            # Standard fitting for RandomForest
                            model.fit(X, y)
                        
                        if self._is_model_fitted(model, model_name):
                            logger.info(f"✅ Successfully retrained {model_name} for {horizon}")
                        else:
                            logger.error(f"❌ Failed to retrain {model_name} for {horizon}")
                    except Exception as e:
                        logger.error(f"❌ Failed to retrain {model_name} for {horizon}: {e}")
                        # Try to reinitialize the model if it's corrupted
                        try:
                            if model_name == 'rf':
                                self.models[horizon][model_name] = RandomForestRegressor(
                                    n_estimators=100, 
                                    max_depth=10,
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    random_state=42
                                )
                            elif model_name == 'xgb':
                                self.models[horizon][model_name] = xgb.XGBRegressor(
                                    n_estimators=100,
                                    max_depth=6,
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    random_state=42,
                                    enable_categorical=False,
                                    missing=0
                                )
                            elif model_name == 'lgb':
                                self.models[horizon][model_name] = lgb.LGBMRegressor(
                                    n_estimators=100,
                                    max_depth=6,
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    random_state=42,
                                    verbose=-1,
                                    force_col_wise=True
                                )
                            logger.info(f"Reinitialized {model_name} model for {horizon}")
                        except Exception as reinit_error:
                            logger.error(f"Failed to reinitialize {model_name} for {horizon}: {reinit_error}")
    
    def _is_model_fitted(self, model, model_name: str) -> bool:
        """
        Check if a model is properly fitted
        
        Args:
            model: The model to check
            model_name: Name of the model type
            
        Returns:
            True if model is fitted, False otherwise
        """
        try:
            if model_name == 'rf':
                # RandomForest is fitted if it has estimators_
                return hasattr(model, 'estimators_') and len(model.estimators_) > 0
            elif model_name == 'xgb':
                # XGBoost is fitted if it has booster
                return hasattr(model, 'booster') and model.booster is not None
            elif model_name == 'lgb':
                # LightGBM is fitted if it has booster
                return hasattr(model, 'booster') and model.booster is not None
            else:
                # Generic check for other models
                return hasattr(model, 'predict')
        except Exception:
            return False
    
    def predict_with_uncertainty(self, features: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Make predictions with uncertainty quantification
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary with predictions and uncertainty for each horizon and model
        """
        predictions_with_uncertainty = {}
        
        for horizon in self.horizons:
            predictions_with_uncertainty[horizon] = {}
            
            for model_name in self.models[horizon].keys():
                try:
                    # Get base prediction
                    base_pred = self.predict_all_horizons(features)[horizon].get(model_name, None)
                    
                    if base_pred is not None:
                        # Calculate uncertainty using ensemble variance
                        ensemble_predictions = []
                        
                        # Use multiple random seeds for ensemble
                        for seed in [42, 123, 456, 789, 999]:
                            # Create temporary model with different seed
                            if model_name == 'rf':
                                temp_model = RandomForestRegressor(n_estimators=50, random_state=seed)
                            elif model_name == 'xgb':
                                temp_model = xgb.XGBRegressor(n_estimators=50, random_state=seed)
                            elif model_name == 'lgb':
                                temp_model = lgb.LGBMRegressor(n_estimators=50, random_state=seed)
                            else:
                                continue
                            
                            # Quick fit and predict (using last 100 samples for speed)
                            if len(features) > 100:
                                X_temp = features.iloc[-100:]
                                y_temp = features.iloc[-100:, 0]  # Use first feature as proxy target
                                temp_model.fit(X_temp, y_temp)
                                pred = temp_model.predict(features.iloc[-1:])[0]
                                ensemble_predictions.append(pred)
                        
                        # Calculate uncertainty metrics
                        if len(ensemble_predictions) > 1:
                            uncertainty = np.std(ensemble_predictions)
                            confidence_interval = 1.96 * uncertainty  # 95% confidence interval
                        else:
                            uncertainty = 0.0
                            confidence_interval = 0.0
                        
                        predictions_with_uncertainty[horizon][model_name] = {
                            'prediction': base_pred,
                            'uncertainty': uncertainty,
                            'confidence_interval': confidence_interval,
                            'lower_bound': base_pred - confidence_interval,
                            'upper_bound': base_pred + confidence_interval
                        }
                    else:
                        predictions_with_uncertainty[horizon][model_name] = None
                        
                except Exception as e:
                    logger.error(f"Error calculating uncertainty for {model_name} on {horizon}: {e}")
                    predictions_with_uncertainty[horizon][model_name] = None
        
        return predictions_with_uncertainty
    
    def get_ensemble_prediction(self, features: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Get ensemble prediction (average of all models) for each horizon
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary with ensemble predictions and uncertainty
        """
        predictions = self.predict_with_uncertainty(features)
        ensemble_results = {}
        
        for horizon in self.horizons:
            valid_predictions = []
            uncertainties = []
            
            for model_name, result in predictions[horizon].items():
                if result is not None:
                    valid_predictions.append(result['prediction'])
                    uncertainties.append(result['uncertainty'])
            
            if valid_predictions:
                # Calculate ensemble statistics
                ensemble_pred = np.mean(valid_predictions)
                ensemble_uncertainty = np.mean(uncertainties) if uncertainties else 0.0
                ensemble_std = np.std(valid_predictions)
                
                ensemble_results[horizon] = {
                    'prediction': ensemble_pred,
                    'uncertainty': ensemble_uncertainty,
                    'std': ensemble_std,
                    'model_count': len(valid_predictions)
                }
            else:
                ensemble_results[horizon] = None
        
        return ensemble_results
    
    def _parse_horizon(self, horizon: str) -> int:
        """
        Parse horizon string to minutes
        
        Args:
            horizon: Horizon string (e.g., '1min', '1hour', '1day')
            
        Returns:
            Number of minutes
        """
        if 'min' in horizon:
            return int(horizon.replace('min', ''))
        elif 'hour' in horizon:
            return int(horizon.replace('hour', '')) * 60
        elif 'day' in horizon:
            return int(horizon.replace('day', '')) * 24 * 60
        else:
            return 0

class RealTimeGPFAPredictor:
    """
    Main class for real-time GPFA prediction system
    """
    
    def __init__(self, symbols: List[str], n_factors: int = 5):
        """
        Initialize the real-time GPFA predictor
        
        Args:
            symbols: List of stock symbols to predict
            n_factors: Number of GPFA factors
        """
        self.symbols = symbols
        self.n_factors = n_factors
        
        # Initialize components
        self.data_feed = RealTimeDataFeed(symbols)
        self.gpfa_model = RealTimeGPFA(n_factors)
        self.prediction_ensemble = PredictionEnsemble()
        
        # Initialize visualizer
        self.visualizer = RealTimeVisualizer(symbols, n_factors)
        
        # Storage for predictions and performance
        self.predictions_history = []
        self.performance_metrics = {}
        self.performance_history = []
        self.model_performance = {}
        
        logger.info(f"Initialized RealTimeGPFAPredictor for {len(symbols)} symbols")
    
    def initialize_system(self) -> None:
        """
        Initialize the system with historical data
        """
        logger.info("Initializing system with historical data...")
        
        # Fetch initial historical data
        historical_data = self._fetch_historical_data()
        
        if historical_data is not None and not historical_data.empty:
            # Fit GPFA model with multi-symbol data
            self.gpfa_model.fit_model(historical_data)
            
            # Train prediction models with OHLCV data
            if hasattr(self, 'ensemble_historical_data'):
                self.prediction_ensemble.train_models(self.ensemble_historical_data)
                logger.info("System initialization completed")
            else:
                logger.error("No ensemble data available for training")
        else:
            logger.error("Failed to initialize system - no historical data")
    
    def _fetch_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for model initialization
        Returns:
            Historical price data
        """
        import glob
        import os
        try:
            cache_dir = 'real_data_cache'
            symbol_files = {s: os.path.join(cache_dir, f"{s}.csv") for s in self.symbols}
            all_exist = all(os.path.exists(f) for f in symbol_files.values())
            gpfa_data = {}
            ensemble_data = None
            if all_exist:
                # Load each symbol's CSV and aggregate
                for i, (symbol, path) in enumerate(symbol_files.items()):
                    df = pd.read_csv(path, parse_dates=['Date'])
                    df = df.set_index('Date')
                    gpfa_data[symbol] = df['Close']
                    if i == 0:
                        ensemble_data = df.copy()
                self.gpfa_historical_data = pd.DataFrame(gpfa_data)
                self.ensemble_historical_data = ensemble_data
                return self.gpfa_historical_data
            else:
                # Fallback to yfinance (original logic)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                for i, symbol in enumerate(self.symbols):
                    ticker = yf.Ticker(symbol)
                    hist = None
                    for interval in ['1h', '1d']:
                        try:
                            hist = ticker.history(start=start_date, end=end_date, interval=interval)
                            if not hist.empty and len(hist) > 100:
                                break
                        except Exception as e:
                            continue
                    if hist is not None and not hist.empty:
                        gpfa_data[symbol] = hist['Close']
                        if i == 0:
                            ensemble_data = hist.copy()
                if gpfa_data and ensemble_data is not None:
                    self.gpfa_historical_data = pd.DataFrame(gpfa_data)
                    self.ensemble_historical_data = ensemble_data
                    return self.gpfa_historical_data
                else:
                    return None
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def run_prediction_cycle(self) -> None:
        """
        Run one complete prediction cycle with robust error handling
        """
        cycle_start = datetime.now()
        cycle_success = False
        
        try:
            logger.info("Starting prediction cycle...")
            
            # Step 1: Fetch latest data
            try:
                latest_data = self.data_feed.fetch_real_time_data()
                if not latest_data:
                    logger.warning("No data received in prediction cycle")
                    return
                logger.debug("Data fetch successful")
            except Exception as e:
                logger.error(f"Data fetch failed: {e}")
                return
            
            # Step 2: Update data buffer
            try:
                self.data_feed.update_buffer(latest_data)
                logger.debug("Data buffer updated")
            except Exception as e:
                logger.error(f"Data buffer update failed: {e}")
                return
            
            # Step 3: Get combined latest data
            try:
                combined_data = self.data_feed.get_latest_data()
                if combined_data.empty:
                    logger.warning("No combined data available")
                    return
                logger.debug("Combined data prepared")
            except Exception as e:
                logger.error(f"Data combination failed: {e}")
                return
            
            # Step 4: Prepare features
            try:
                features = self.prediction_ensemble.prepare_features(combined_data)
                if features.empty:
                    logger.warning("Feature preparation failed - empty features")
                    return
                logger.debug("Features prepared successfully")
            except Exception as e:
                logger.error(f"Feature preparation failed: {e}")
                return
            
            # Step 5: Check and retrain failed models if needed
            try:
                # Check if we have enough data for retraining
                if hasattr(self, 'ensemble_historical_data') and len(self.ensemble_historical_data) > 50:
                    # Retrain any failed models
                    self.prediction_ensemble.retrain_failed_models(self.ensemble_historical_data)
                logger.debug("Model retraining check completed")
            except Exception as e:
                logger.error(f"Model retraining check failed: {e}")
            
            # Step 6: Make predictions with error handling for each component
            predictions = {}
            predictions_with_uncertainty = {}
            ensemble_predictions = {}
            
            try:
                predictions = self.prediction_ensemble.predict_all_horizons(features)
                logger.debug("Base predictions completed")
            except Exception as e:
                logger.error(f"Base predictions failed: {e}")
                predictions = {}
            
            try:
                predictions_with_uncertainty = self.prediction_ensemble.predict_with_uncertainty(features)
                logger.debug("Uncertainty quantification completed")
            except Exception as e:
                logger.error(f"Uncertainty quantification failed: {e}")
                predictions_with_uncertainty = {}
            
            try:
                ensemble_predictions = self.prediction_ensemble.get_ensemble_prediction(features)
                logger.debug("Ensemble predictions completed")
            except Exception as e:
                logger.error(f"Ensemble predictions failed: {e}")
                ensemble_predictions = {}
            
            # Step 7: Store predictions
            try:
                prediction_record = {
                    'timestamp': datetime.now(),
                    'predictions': predictions,
                    'predictions_with_uncertainty': predictions_with_uncertainty,
                    'ensemble_predictions': ensemble_predictions,
                    'data': combined_data
                }
                self.predictions_history.append(prediction_record)
                logger.debug("Predictions stored")
            except Exception as e:
                logger.error(f"Prediction storage failed: {e}")
            
            # Step 8: Update visualizer
            try:
                self.visualizer.update_price_data(combined_data)
                self.visualizer.update_predictions(predictions, combined_data)
                
                # Update latent factors if available
                if hasattr(self.gpfa_model, 'latent_trajectories') and self.gpfa_model.latent_trajectories is not None:
                    current_factors = self.gpfa_model.latent_trajectories[:, -1]  # Latest factors
                    self.visualizer.update_factors(current_factors)
                logger.debug("Visualizer updated")
            except Exception as e:
                logger.error(f"Visualizer update failed: {e}")
            
            # Step 9: Log predictions
            try:
                self._log_predictions(predictions, ensemble_predictions)
            except Exception as e:
                logger.error(f"Prediction logging failed: {e}")
            
            # Step 9: Monitor performance (every 5 cycles)
            try:
                if len(self.predictions_history) % 5 == 0:
                    self.monitor_performance()
                    
                    # Check for model drift
                    drift_indicators = self.detect_model_drift()
                    if drift_indicators:
                        logger.warning("Model Drift Detection:")
                        for horizon, models in drift_indicators.items():
                            for model, drift in models.items():
                                if drift and drift['drift_detected']:
                                    logger.warning(f"  {horizon} - {model}: Drift detected (ratio: {drift['drift_ratio']:.2f})")
            except Exception as e:
                logger.error(f"Performance monitoring failed: {e}")
            
            cycle_success = True
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Prediction cycle completed successfully in {cycle_duration:.2f}s")
            
        except Exception as e:
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.error(f"Critical error in prediction cycle after {cycle_duration:.2f}s: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Log additional error context
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # Always log cycle status
            if not cycle_success:
                logger.warning("Prediction cycle failed - system will continue with next cycle")
    
    def _log_predictions(self, predictions: Dict, ensemble_predictions: Dict = None) -> None:
        """
        Log current predictions with uncertainty
        
        Args:
            predictions: Prediction dictionary
            ensemble_predictions: Ensemble predictions with uncertainty
        """
        logger.info("Current Predictions:")
        for horizon, models in predictions.items():
            logger.info(f"  {horizon}:")
            for model, pred in models.items():
                if pred is not None:
                    logger.info(f"    {model}: {pred:.4f}")
            
            # Log ensemble prediction if available
            if ensemble_predictions and horizon in ensemble_predictions and ensemble_predictions[horizon]:
                ensemble = ensemble_predictions[horizon]
                logger.info(f"    ENSEMBLE: {ensemble['prediction']:.4f} ± {ensemble['uncertainty']:.4f} ({ensemble['model_count']} models)")
    
    def start_real_time_loop(self, interval_seconds: int = 60) -> None:
        """
        Start the real-time prediction loop
        
        Args:
            interval_seconds: Update interval in seconds
        """
        logger.info(f"Starting real-time prediction loop with {interval_seconds}s interval")
        
        # Schedule the prediction cycle
        schedule.every(interval_seconds).seconds.do(self.run_prediction_cycle)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Real-time loop stopped by user")
        except Exception as e:
            logger.error(f"Error in real-time loop: {e}")
    
    def create_visualizations(self, save_dir: str = './') -> None:
        """
        Create comprehensive visualizations
        
        Args:
            save_dir: Directory to save visualization files
        """
        logger.info("Creating visualizations...")
        
        try:
            # Create real-time dashboard
            dashboard_path = f"{save_dir}realtime_dashboard.html"
            self.visualizer.create_real_time_dashboard(dashboard_path)
            
            # Create price prediction plots for each symbol
            for symbol in self.symbols:
                plot_path = f"{save_dir}{symbol}_predictions.png"
                self.visualizer.plot_price_predictions(symbol, plot_path)
            
            # Create latent factors plot
            factors_path = f"{save_dir}latent_factors.png"
            self.visualizer.plot_latent_factors(factors_path)
            
            # Create performance summary
            summary_path = f"{save_dir}performance_summary.html"
            self.visualizer.create_performance_summary(summary_path)
            
            logger.info("All visualizations created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def run_prediction_test(self, duration_minutes: int = 5) -> None:
        """
        Run a prediction test for a specified duration
        
        Args:
            duration_minutes: Duration of the test in minutes
        """
        logger.info(f"Starting prediction test for {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                self.run_prediction_cycle()
                time.sleep(60)  # Wait 1 minute between cycles
                
                # Create visualizations every 2 minutes
                if int(time.time() - start_time) % 120 == 0:
                    self.create_visualizations()
            
            # Final visualizations
            self.create_visualizations()
            logger.info("Prediction test completed!")
            
        except KeyboardInterrupt:
            logger.info("Prediction test interrupted by user")
            self.create_visualizations()  # Create final visualizations
        except Exception as e:
            logger.error(f"Error during prediction test: {e}")
            self.create_visualizations()  # Create visualizations even if error occurs

def main():
    """
    Main function to run the real-time GPFA predictor with visualization
    """
    # Example usage
    symbols = ['AAPL', 'GOOGL', 'MSFT']  # Example symbols
    
    # Initialize predictor
    predictor = RealTimeGPFAPredictor(symbols, n_factors=3)
    
    # Initialize system
    predictor.initialize_system()
    
    print("=" * 60)
    print("Real-time GPFA Predictor with Enhanced Visualization")
    print("=" * 60)
    print("Options:")
    print("1. Run prediction test (5 minutes with visualizations)")
    print("2. Start real-time loop (continuous)")
    print("3. Create sample visualizations only")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\nStarting prediction test with visualizations...")
                predictor.run_prediction_test(duration_minutes=5)
                print("\nTest completed! Check the generated visualization files:")
                print("- realtime_dashboard.html (Interactive dashboard)")
                print("- *_predictions.png (Price prediction plots)")
                print("- latent_factors.png (Latent factors)")
                print("- performance_summary.html (Performance summary)")
                break
                
            elif choice == '2':
                print("\nStarting real-time loop...")
                print("Press Ctrl+C to stop")
                predictor.start_real_time_loop(interval_seconds=60)
                break
                
            elif choice == '3':
                print("\nCreating sample visualizations...")
                predictor.create_visualizations()
                print("Sample visualizations created!")
                break
                
            elif choice == '4':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Real-time GPFA predictor completed!")

if __name__ == "__main__":
    main() 