# Enhanced GPFA Predictor with Real Data Integration
"""
Enhanced GPFA predictor that integrates with the real data system
and includes improved model training, validation, and prediction capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Real-time processing
from typing import Dict, List, Tuple, Optional
import logging

# Import our enhanced data system
from enhanced_real_data_system import EnhancedRealDataFeed, RealTimeDataStreamer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedGPFA:
    """
    Enhanced GPFA model with improved factor extraction and prediction
    """
    
    def __init__(self, n_factors: int = 5, kernel_type: str = 'rbf'):
        """
        Initialize enhanced GPFA model
        
        Args:
            n_factors: Number of latent factors
            kernel_type: Type of Gaussian Process kernel ('rbf', 'matern')
        """
        self.n_factors = n_factors
        self.kernel_type = kernel_type
        
        # Initialize PCA
        self.pca = PCA(n_components=n_factors, random_state=42)
        
        # Initialize Gaussian Process models
        self.gp_models = []
        self._initialize_gp_models()
        
        # Model state
        self.factor_loadings = None
        self.latent_trajectories = None
        self.explained_variance_ratio = None
        self.last_update = None
        
        # Data preprocessing
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized EnhancedGPFA with {n_factors} factors and {kernel_type} kernel")
    
    def _initialize_gp_models(self):
        """Initialize Gaussian Process models with appropriate kernels"""
        for i in range(self.n_factors):
            if self.kernel_type == 'rbf':
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            elif self.kernel_type == 'matern':
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            
            gp_model = GaussianProcessRegressor(
                kernel=kernel, 
                random_state=42,
                n_restarts_optimizer=5,
                alpha=1e-6
            )
            self.gp_models.append(gp_model)
    
    def fit_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Fit the enhanced GPFA model to historical data
        
        Args:
            data: Historical price data (symbols as columns, time as index)
            
        Returns:
            Dictionary with fitting metrics
        """
        try:
            # Prepare data
            price_data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Standardize the data
            price_data_scaled = self.scaler.fit_transform(price_data)
            
            # Apply PCA to extract factors
            self.factor_loadings = self.pca.fit_transform(price_data_scaled.T)
            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            
            # Get latent trajectories
            latent_data = self.pca.transform(price_data_scaled.T).T
            
            # Fit Gaussian Process models for each factor
            time_points = np.arange(len(latent_data[0])).reshape(-1, 1)
            fitting_metrics = {}
            
            for i, gp_model in enumerate(self.gp_models):
                # Fit the model
                gp_model.fit(time_points, latent_data[i])
                
                # Calculate fitting metrics
                predictions, _ = gp_model.predict(time_points, return_std=True)
                mse = mean_squared_error(latent_data[i], predictions)
                r2 = r2_score(latent_data[i], predictions)
                
                fitting_metrics[f'factor_{i}_mse'] = mse
                fitting_metrics[f'factor_{i}_r2'] = r2
            
            self.latent_trajectories = latent_data
            self.last_update = datetime.now()
            
            # Calculate overall metrics
            total_explained_variance = np.sum(self.explained_variance_ratio)
            fitting_metrics['total_explained_variance'] = total_explained_variance
            fitting_metrics['avg_factor_r2'] = np.mean([fitting_metrics[f'factor_{i}_r2'] for i in range(self.n_factors)])
            
            logger.info(f"Fitted EnhancedGPFA model with {total_explained_variance:.2%} explained variance")
            logger.info(f"Average factor R²: {fitting_metrics['avg_factor_r2']:.3f}")
            
            return fitting_metrics
            
        except Exception as e:
            logger.error(f"Error fitting EnhancedGPFA model: {e}")
            return {}
    
    def predict_factors(self, time_horizon: int, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict latent factors for a given time horizon
        
        Args:
            time_horizon: Number of time steps to predict ahead
            return_std: Whether to return prediction uncertainties
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.gp_models or self.latent_trajectories is None:
            raise ValueError("Model not fitted yet")
        
        try:
            # Get current time point
            current_time = len(self.latent_trajectories[0])
            future_times = np.arange(current_time, current_time + time_horizon).reshape(-1, 1)
            
            # Predict each factor
            predictions = []
            uncertainties = []
            
            for gp_model in self.gp_models:
                if return_std:
                    pred, std = gp_model.predict(future_times, return_std=True)
                    predictions.append(pred)
                    uncertainties.append(std)
                else:
                    pred = gp_model.predict(future_times)
                    predictions.append(pred)
            
            predictions_array = np.array(predictions)
            uncertainties_array = np.array(uncertainties) if return_std else None
            
            return predictions_array, uncertainties_array
            
        except Exception as e:
            logger.error(f"Error predicting factors: {e}")
            raise
    
    def reconstruct_prices(self, latent_factors: np.ndarray) -> np.ndarray:
        """
        Reconstruct prices from latent factors
        
        Args:
            latent_factors: Predicted latent factors
            
        Returns:
            Reconstructed price data
        """
        try:
            # Transform latent factors back to price space
            reconstructed_scaled = self.pca.inverse_transform(latent_factors.T)
            
            # Inverse transform the scaling
            reconstructed_prices = self.scaler.inverse_transform(reconstructed_scaled)
            
            return reconstructed_prices
            
        except Exception as e:
            logger.error(f"Error reconstructing prices: {e}")
            raise
    
    def get_factor_importance(self) -> Dict[str, float]:
        """
        Get factor importance based on explained variance
        
        Returns:
            Dictionary with factor importance scores
        """
        if self.explained_variance_ratio is None:
            return {}
        
        importance = {}
        for i, var_ratio in enumerate(self.explained_variance_ratio):
            importance[f'factor_{i}'] = var_ratio
        
        return importance

class EnhancedPredictionEnsemble:
    """
    Enhanced prediction ensemble with improved model training and validation
    """
    
    def __init__(self, horizons: List[str] = ['1min', '5min', '15min', '1hour', '1day']):
        """
        Initialize enhanced prediction ensemble
        
        Args:
            horizons: List of prediction horizons
        """
        self.horizons = horizons
        self.models = {horizon: {} for horizon in horizons}
        self.scalers = {horizon: StandardScaler() for horizon in horizons}
        self.feature_importance = {horizon: {} for horizon in horizons}
        self.model_metrics = {horizon: {} for horizon in horizons}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'enable_categorical': False,
                'missing': 0,
                'n_jobs': -1
            },
            'lightgbm': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True,
                'n_jobs': -1
            }
        }
        
        logger.info(f"Initialized EnhancedPredictionEnsemble with {len(horizons)} horizons")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare enhanced features for prediction
        
        Args:
            data: Raw price data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            features = pd.DataFrame()
            
            # Basic price features
            features['close'] = data['Close']
            features['open'] = data['Open']
            features['high'] = data['High']
            features['low'] = data['Low']
            features['volume'] = data['Volume']
            
            # Price changes
            features['price_change'] = data['Close'].pct_change()
            features['price_change_abs'] = features['price_change'].abs()
            
            # OHLC features
            features['body_size'] = (data['Close'] - data['Open']) / data['Open']
            features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Open']
            features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Open']
            features['high_low_range'] = (data['High'] - data['Low']) / data['Open']
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = data['Close'].rolling(period).mean()
                features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
                features[f'price_sma_{period}_ratio'] = data['Close'] / features[f'sma_{period}']
            
            # Volatility features
            for period in [5, 10, 20]:
                features[f'volatility_{period}'] = data['Close'].rolling(period).std()
                features[f'volatility_{period}_pct'] = features[f'volatility_{period}'] / data['Close']
            
            # Volume features
            features['volume_sma_20'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma_20']
            
            # Technical indicators
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_sma = data['Close'].rolling(bb_period).mean()
            bb_std_dev = data['Close'].rolling(bb_period).std()
            features['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
            features['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (data['Close'] - features['bb_lower']) / features['bb_width']
            
            # Stochastic Oscillator
            stoch_period = 14
            lowest_low = data['Low'].rolling(stoch_period).min()
            highest_high = data['High'].rolling(stoch_period).max()
            features['stoch_k'] = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()
            
            # Williams %R
            features['williams_r'] = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))
            
            # Momentum features
            for period in [1, 3, 5, 10]:
                features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            
            # Mean reversion features
            for period in [5, 10, 20]:
                features[f'mean_reversion_{period}'] = (data['Close'] - data['Close'].rolling(period).mean()) / data['Close'].rolling(period).std()
            
            # Time-based features
            features['hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
            features['day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
            
            # Clean up infinite values and NaNs
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.info(f"Prepared {len(features.columns)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def train_models(self, data: pd.DataFrame, target_column: str = 'Close') -> Dict[str, Dict[str, float]]:
        """
        Train enhanced models for all horizons
        
        Args:
            data: Training data
            target_column: Target column for prediction
            
        Returns:
            Dictionary with training metrics for each horizon
        """
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            if features.empty:
                logger.error("No features prepared, cannot train models")
                return {}
            
            # Prepare target
            target = data[target_column]
            
            # Remove rows with NaN values
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            features_clean = features[valid_indices]
            target_clean = target[valid_indices]
            
            if len(features_clean) < 100:
                logger.error(f"Insufficient data for training: {len(features_clean)} samples")
                return {}
            
            training_metrics = {}
            
            for horizon in self.horizons:
                logger.info(f"Training models for {horizon} horizon...")
                
                # Create target for this horizon
                horizon_steps = self._parse_horizon(horizon)
                if horizon_steps >= len(target_clean):
                    logger.warning(f"Horizon {horizon} too large for data size, skipping")
                    continue
                
                target_horizon = target_clean.shift(-horizon_steps)
                
                # Remove rows where target is NaN
                valid_target_indices = ~target_horizon.isnull()
                features_horizon = features_clean[valid_target_indices]
                target_horizon_clean = target_horizon[valid_target_indices]
                
                if len(features_horizon) < 50:
                    logger.warning(f"Insufficient data for {horizon} horizon: {len(features_horizon)} samples")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    features_horizon, target_horizon_clean, 
                    test_size=0.2, random_state=42, shuffle=False
                )
                
                # Scale features
                X_train_scaled = self.scalers[horizon].fit_transform(X_train)
                X_test_scaled = self.scalers[horizon].transform(X_test)
                
                horizon_metrics = {}
                
                # Train Random Forest
                try:
                    rf_model = RandomForestRegressor(**self.model_configs['random_forest'])
                    rf_model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    rf_pred = rf_model.predict(X_test_scaled)
                    
                    # Metrics
                    rf_mse = mean_squared_error(y_test, rf_pred)
                    rf_mae = mean_absolute_error(y_test, rf_pred)
                    rf_r2 = r2_score(y_test, rf_pred)
                    
                    horizon_metrics['random_forest'] = {
                        'mse': rf_mse,
                        'mae': rf_mae,
                        'r2': rf_r2
                    }
                    
                    self.models[horizon]['random_forest'] = rf_model
                    self.feature_importance[horizon]['random_forest'] = rf_model.feature_importances_
                    
                    logger.info(f"Random Forest {horizon}: R² = {rf_r2:.3f}, MSE = {rf_mse:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error training Random Forest for {horizon}: {e}")
                
                # Train XGBoost
                try:
                    xgb_model = xgb.XGBRegressor(**self.model_configs['xgboost'])
                    xgb_model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    xgb_pred = xgb_model.predict(X_test_scaled)
                    
                    # Metrics
                    xgb_mse = mean_squared_error(y_test, xgb_pred)
                    xgb_mae = mean_absolute_error(y_test, xgb_pred)
                    xgb_r2 = r2_score(y_test, xgb_pred)
                    
                    horizon_metrics['xgboost'] = {
                        'mse': xgb_mse,
                        'mae': xgb_mae,
                        'r2': xgb_r2
                    }
                    
                    self.models[horizon]['xgboost'] = xgb_model
                    self.feature_importance[horizon]['xgboost'] = xgb_model.feature_importances_
                    
                    logger.info(f"XGBoost {horizon}: R² = {xgb_r2:.3f}, MSE = {xgb_mse:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error training XGBoost for {horizon}: {e}")
                
                # Train LightGBM
                try:
                    lgb_model = lgb.LGBMRegressor(**self.model_configs['lightgbm'])
                    lgb_model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    lgb_pred = lgb_model.predict(X_test_scaled)
                    
                    # Metrics
                    lgb_mse = mean_squared_error(y_test, lgb_pred)
                    lgb_mae = mean_absolute_error(y_test, lgb_pred)
                    lgb_r2 = r2_score(y_test, lgb_pred)
                    
                    horizon_metrics['lightgbm'] = {
                        'mse': lgb_mse,
                        'mae': lgb_mae,
                        'r2': lgb_r2
                    }
                    
                    self.models[horizon]['lightgbm'] = lgb_model
                    self.feature_importance[horizon]['lightgbm'] = lgb_model.feature_importances_
                    
                    logger.info(f"LightGBM {horizon}: R² = {lgb_r2:.3f}, MSE = {lgb_mse:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error training LightGBM for {horizon}: {e}")
                
                training_metrics[horizon] = horizon_metrics
                self.model_metrics[horizon] = horizon_metrics
            
            logger.info(f"Training completed for {len(training_metrics)} horizons")
            return training_metrics
            
        except Exception as e:
            logger.error(f"Error in train_models: {e}")
            return {}
    
    def predict_all_horizons(self, features: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Make predictions for all horizons using all models
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary with predictions for each horizon and model
        """
        try:
            predictions = {}
            
            for horizon in self.horizons:
                if horizon not in self.models:
                    continue
                
                horizon_predictions = {}
                
                # Scale features
                features_scaled = self.scalers[horizon].transform(features)
                
                for model_name, model in self.models[horizon].items():
                    try:
                        pred = model.predict(features_scaled)
                        horizon_predictions[model_name] = float(pred[0])
                    except Exception as e:
                        logger.error(f"Error predicting with {model_name} for {horizon}: {e}")
                        continue
                
                if horizon_predictions:
                    predictions[horizon] = horizon_predictions
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_all_horizons: {e}")
            return {}
    
    def get_ensemble_prediction(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Get ensemble prediction (average of all models)
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary with ensemble predictions for each horizon
        """
        try:
            all_predictions = self.predict_all_horizons(features)
            ensemble_predictions = {}
            
            for horizon, model_predictions in all_predictions.items():
                if model_predictions:
                    # Calculate weighted average based on model performance
                    weights = []
                    predictions = []
                    
                    for model_name, pred in model_predictions.items():
                        if horizon in self.model_metrics and model_name in self.model_metrics[horizon]:
                            # Use R² score as weight (higher is better)
                            weight = max(0, self.model_metrics[horizon][model_name]['r2'])
                            weights.append(weight)
                            predictions.append(pred)
                        else:
                            # Equal weight if no metrics available
                            weights.append(1.0)
                            predictions.append(pred)
                    
                    if weights and predictions:
                        # Normalize weights
                        total_weight = sum(weights)
                        if total_weight > 0:
                            normalized_weights = [w / total_weight for w in weights]
                            ensemble_pred = sum(p * w for p, w in zip(predictions, normalized_weights))
                            ensemble_predictions[horizon] = ensemble_pred
                        else:
                            # Fallback to simple average
                            ensemble_predictions[horizon] = np.mean(predictions)
            
            return ensemble_predictions
            
        except Exception as e:
            logger.error(f"Error in get_ensemble_prediction: {e}")
            return {}
    
    def _parse_horizon(self, horizon: str) -> int:
        """Parse horizon string to number of time steps"""
        if horizon == '1min':
            return 1
        elif horizon == '5min':
            return 5
        elif horizon == '15min':
            return 15
        elif horizon == '1hour':
            return 60
        elif horizon == '1day':
            return 1440  # Assuming 1-minute data
        else:
            return 1

class EnhancedGPFAPredictor:
    """
    Enhanced GPFA predictor with real data integration
    """
    
    def __init__(self, symbols: List[str], n_factors: int = 5, use_real_data: bool = True):
        """
        Initialize enhanced GPFA predictor
        
        Args:
            symbols: List of stock symbols
            n_factors: Number of GPFA factors
            use_real_data: Whether to use real data or simulated data
        """
        self.symbols = symbols
        self.n_factors = n_factors
        self.use_real_data = use_real_data
        
        # Initialize components
        if use_real_data:
            self.data_feed = EnhancedRealDataFeed(symbols)
            self.data_streamer = RealTimeDataStreamer(self.data_feed)
        else:
            self.data_feed = None
            self.data_streamer = None
        
        self.gpfa_model = EnhancedGPFA(n_factors)
        self.prediction_ensemble = EnhancedPredictionEnsemble()
        
        # State tracking
        self.is_initialized = False
        self.last_prediction = None
        self.prediction_history = []
        
        logger.info(f"Initialized EnhancedGPFAPredictor for {len(symbols)} symbols")
    
    def initialize_system(self) -> bool:
        """
        Initialize the prediction system
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing EnhancedGPFAPredictor system...")
            
            # Fetch historical data
            if self.use_real_data:
                historical_data = self._fetch_historical_data()
            else:
                historical_data = self._generate_simulated_data()
            
            if historical_data is None or historical_data.empty:
                logger.error("Failed to get historical data for initialization")
                return False
            
            # Fit GPFA model
            gpfa_metrics = self.gpfa_model.fit_model(historical_data)
            if not gpfa_metrics:
                logger.error("Failed to fit GPFA model")
                return False
            
            # Train prediction ensemble
            ensemble_metrics = self.prediction_ensemble.train_models(historical_data)
            if not ensemble_metrics:
                logger.error("Failed to train prediction ensemble")
                return False
            
            # Start real-time data streaming if using real data
            if self.use_real_data and self.data_streamer:
                self.data_streamer.start_streaming()
            
            self.is_initialized = True
            logger.info("EnhancedGPFAPredictor system initialized successfully")
            
            # Log initialization metrics
            logger.info(f"GPFA explained variance: {gpfa_metrics.get('total_explained_variance', 0):.2%}")
            for horizon, metrics in ensemble_metrics.items():
                avg_r2 = np.mean([m.get('r2', 0) for m in metrics.values()])
                logger.info(f"Ensemble {horizon} average R²: {avg_r2:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False
    
    def run_prediction_cycle(self) -> Dict[str, any]:
        """
        Run a complete prediction cycle
        
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            if not self.is_initialized:
                logger.error("System not initialized")
                return {}
            
            # Get latest data
            if self.use_real_data:
                latest_data = self._get_latest_real_data()
            else:
                latest_data = self._generate_simulated_data()
            
            if latest_data is None or latest_data.empty:
                logger.warning("No data available for prediction")
                return {}
            
            # Prepare features
            features = self.prediction_ensemble.prepare_features(latest_data)
            if features.empty:
                logger.error("Failed to prepare features")
                return {}
            
            # Get ensemble predictions
            ensemble_predictions = self.prediction_ensemble.get_ensemble_prediction(features)
            
            # Get GPFA predictions
            gpfa_predictions = self._get_gpfa_predictions()
            
            # Combine predictions
            combined_predictions = {
                'ensemble': ensemble_predictions,
                'gpfa': gpfa_predictions,
                'timestamp': datetime.now(),
                'symbols': self.symbols,
                'data_points': len(latest_data)
            }
            
            # Store prediction history
            self.prediction_history.append(combined_predictions)
            self.last_prediction = combined_predictions
            
            # Log predictions
            self._log_predictions(combined_predictions)
            
            return combined_predictions
            
        except Exception as e:
            logger.error(f"Error in prediction cycle: {e}")
            return {}
    
    def _fetch_historical_data(self) -> Optional[pd.DataFrame]:
        """Fetch historical data for initialization"""
        try:
            # Fetch 1 year of daily data for each symbol
            all_data = []
            
            for symbol in self.symbols:
                hist_data = self.data_feed.fetch_historical_data(symbol, period='1y', interval='1d')
                if hist_data is not None and not hist_data.empty:
                    # Add symbol column
                    hist_data['Symbol'] = symbol
                    all_data.append(hist_data)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def _get_latest_real_data(self) -> Optional[pd.DataFrame]:
        """Get latest real-time data"""
        try:
            if self.data_streamer:
                latest_data = self.data_streamer.get_latest_data()
                if latest_data:
                    # Combine data from all symbols
                    all_data = []
                    for symbol, df in latest_data.items():
                        df['Symbol'] = symbol
                        all_data.append(df)
                    
                    if all_data:
                        return pd.concat(all_data, ignore_index=True)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest real data: {e}")
            return None
    
    def _generate_simulated_data(self) -> pd.DataFrame:
        """Generate simulated data for testing"""
        # This would integrate with your existing data simulator
        # For now, return a simple DataFrame
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = []
        
        for symbol in self.symbols:
            base_price = 100.0
            for i, date in enumerate(dates):
                price_change = np.random.normal(0, 0.02)
                base_price *= (1 + price_change)
                
                data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Open': base_price * 0.999,
                    'High': base_price * 1.005,
                    'Low': base_price * 0.995,
                    'Close': base_price,
                    'Volume': np.random.randint(1000000, 10000000)
                })
        
        return pd.DataFrame(data)
    
    def _get_gpfa_predictions(self) -> Dict[str, float]:
        """Get GPFA predictions for different horizons"""
        try:
            gpfa_predictions = {}
            
            for horizon in self.prediction_ensemble.horizons:
                horizon_steps = self.prediction_ensemble._parse_horizon(horizon)
                
                # Predict factors
                factor_predictions, _ = self.gpfa_model.predict_factors(horizon_steps)
                
                # Reconstruct prices
                reconstructed_prices = self.gpfa_model.reconstruct_prices(factor_predictions)
                
                # Take the mean of reconstructed prices as prediction
                if reconstructed_prices.size > 0:
                    gpfa_predictions[horizon] = float(np.mean(reconstructed_prices))
            
            return gpfa_predictions
            
        except Exception as e:
            logger.error(f"Error getting GPFA predictions: {e}")
            return {}
    
    def _log_predictions(self, predictions: Dict):
        """Log prediction results"""
        try:
            ensemble_preds = predictions.get('ensemble', {})
            gpfa_preds = predictions.get('gpfa', {})
            
            logger.info("=== Prediction Results ===")
            for horizon in self.prediction_ensemble.horizons:
                ensemble_pred = ensemble_preds.get(horizon, 'N/A')
                gpfa_pred = gpfa_preds.get(horizon, 'N/A')
                
                logger.info(f"{horizon}: Ensemble={ensemble_pred:.2f}, GPFA={gpfa_pred:.2f}")
            
        except Exception as e:
            logger.error(f"Error logging predictions: {e}")
    
    def run_test(self, duration_minutes: int = 10) -> Dict[str, any]:
        """
        Run a test of the prediction system
        
        Args:
            duration_minutes: Test duration in minutes
            
        Returns:
            Test results
        """
        try:
            logger.info(f"Starting {duration_minutes}-minute test...")
            
            # Initialize system
            if not self.initialize_system():
                return {'success': False, 'error': 'Initialization failed'}
            
            # Run prediction cycles
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            cycle_count = 0
            
            while time.time() < end_time:
                cycle_count += 1
                logger.info(f"Test cycle {cycle_count}")
                
                predictions = self.run_prediction_cycle()
                if predictions:
                    logger.info(f"✓ Cycle {cycle_count} completed")
                else:
                    logger.warning(f"✗ Cycle {cycle_count} failed")
                
                # Wait between cycles
                time.sleep(60)  # 1 minute between cycles
            
            # Compile test results
            test_results = {
                'success': True,
                'duration_minutes': duration_minutes,
                'cycles_completed': cycle_count,
                'predictions_made': len(self.prediction_history),
                'symbols_tested': self.symbols,
                'final_predictions': self.last_prediction
            }
            
            logger.info(f"Test completed: {cycle_count} cycles, {len(self.prediction_history)} predictions")
            return test_results
            
        except Exception as e:
            logger.error(f"Error in test: {e}")
            return {'success': False, 'error': str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.data_streamer:
                self.data_streamer.stop_streaming()
            logger.info("EnhancedGPFAPredictor cleanup completed")
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

def main():
    """Test the enhanced GPFA predictor"""
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Initialize predictor with real data
    predictor = EnhancedGPFAPredictor(symbols, n_factors=3, use_real_data=True)
    
    # Run a short test
    test_results = predictor.run_test(duration_minutes=5)
    
    if test_results['success']:
        print("✓ Enhanced GPFA predictor test completed successfully!")
        print(f"Cycles completed: {test_results['cycles_completed']}")
        print(f"Predictions made: {test_results['predictions_made']}")
    else:
        print(f"✗ Test failed: {test_results.get('error', 'Unknown error')}")
    
    # Cleanup
    predictor.cleanup()

if __name__ == "__main__":
    main() 