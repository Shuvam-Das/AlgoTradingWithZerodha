"""ML-enhanced fundamental analysis module"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
from dataclasses import dataclass
import os

logger = logging.getLogger('atwz.ml_fundamental')

@dataclass
class MLPrediction:
    symbol: str
    target: str
    prediction: float
    confidence: float
    features_importance: Dict[str, float]
    model_metadata: Dict[str, Any]
    timestamp: datetime

class MLFundamentalAnalyzer:
    """Machine Learning enhanced fundamental analysis"""
    
    def __init__(self, model_dir: str = 'models/fundamental'):
        self.model_dir = model_dir
        self.scalers: Dict[str, StandardScaler] = {}
        self.models: Dict[str, Any] = {}
        self._load_or_create_models()
    
    def _load_or_create_models(self):
        """Load existing models or create new ones"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        model_types = ['valuation', 'growth', 'risk', 'esg_impact']
        
        for model_type in model_types:
            model_path = os.path.join(self.model_dir, f'{model_type}_model.joblib')
            scaler_path = os.path.join(self.model_dir, f'{model_type}_scaler.joblib')
            
            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_type] = joblib.load(model_path)
                    self.scalers[model_type] = joblib.load(scaler_path)
                else:
                    # Create new models
                    if model_type == 'valuation':
                        self.models[model_type] = self._create_valuation_model()
                    elif model_type == 'growth':
                        self.models[model_type] = self._create_growth_model()
                    elif model_type == 'risk':
                        self.models[model_type] = self._create_risk_model()
                    elif model_type == 'esg_impact':
                        self.models[model_type] = self._create_esg_model()
                    
                    self.scalers[model_type] = StandardScaler()
            
            except Exception as e:
                logger.error(f"Error loading model {model_type}: {e}")
                self.models[model_type] = None
                self.scalers[model_type] = None
    
    def _create_valuation_model(self) -> tf.keras.Model:
        """Create deep learning model for valuation prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_growth_model(self) -> RandomForestRegressor:
        """Create random forest model for growth prediction"""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _create_risk_model(self) -> tf.keras.Model:
        """Create deep learning model for risk assessment"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(15,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_esg_model(self) -> RandomForestRegressor:
        """Create random forest model for ESG impact prediction"""
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
    
    def prepare_features(self, metrics: Dict[str, Any], model_type: str) -> np.ndarray:
        """Prepare features for model input"""
        feature_sets = {
            'valuation': [
                'market_cap', 'pe_ratio', 'pb_ratio', 'debt_to_equity',
                'roe', 'roa', 'eps_growth', 'revenue_growth',
                'dividend_yield', 'payout_ratio', 'altman_z_score',
                'piotroski_score', 'interest_coverage', 'cash_ratio',
                'environmental_score', 'social_score', 'governance_score',
                'controversy_score', 'sector_pe', 'sector_growth'
            ],
            'growth': [
                'eps_growth', 'revenue_growth', 'profit_margin',
                'operating_margin', 'roa', 'roe', 'debt_to_equity',
                'current_ratio', 'quick_ratio', 'inventory_turnover'
            ],
            'risk': [
                'altman_z_score', 'beneish_m_score', 'leverage_ratio',
                'interest_coverage', 'cash_ratio', 'current_ratio',
                'debt_to_equity', 'beta', 'volatility', 'volume_variance'
            ],
            'esg_impact': [
                'environmental_score', 'social_score', 'governance_score',
                'controversy_score', 'market_cap', 'volume', 'volatility',
                'sector_rank', 'peer_percentile', 'news_sentiment'
            ]
        }
        
        features = []
        for feature in feature_sets[model_type]:
            features.append(metrics.get(feature, 0))
        
        return np.array(features).reshape(1, -1)
    
    def predict_valuation(self, metrics: Dict[str, Any]) -> MLPrediction:
        """Predict fair value using ML model"""
        try:
            features = self.prepare_features(metrics, 'valuation')
            scaled_features = self.scalers['valuation'].transform(features)
            
            prediction = self.models['valuation'].predict(scaled_features)[0][0]
            
            # Get feature importance if available
            if isinstance(self.models['valuation'], RandomForestRegressor):
                importance = dict(zip(
                    features.columns,
                    self.models['valuation'].feature_importances_
                ))
            else:
                importance = {}
            
            return MLPrediction(
                symbol=metrics['symbol'],
                target='valuation',
                prediction=prediction,
                confidence=0.8,  # Can be calculated from model uncertainty
                features_importance=importance,
                model_metadata={'model_type': 'deep_learning'},
                timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Error predicting valuation: {e}")
            return MLPrediction(
                symbol=metrics['symbol'],
                target='valuation',
                prediction=0,
                confidence=0,
                features_importance={},
                model_metadata={},
                timestamp=datetime.now()
            )
    
    def predict_growth(self, metrics: Dict[str, Any]) -> MLPrediction:
        """Predict growth metrics using ML model"""
        try:
            features = self.prepare_features(metrics, 'growth')
            scaled_features = self.scalers['growth'].transform(features)
            
            prediction = self.models['growth'].predict(scaled_features)[0]
            importance = dict(zip(
                self.models['growth'].feature_names_in_,
                self.models['growth'].feature_importances_
            ))
            
            return MLPrediction(
                symbol=metrics['symbol'],
                target='growth',
                prediction=prediction,
                confidence=0.75,
                features_importance=importance,
                model_metadata={'model_type': 'random_forest'},
                timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Error predicting growth: {e}")
            return MLPrediction(
                symbol=metrics['symbol'],
                target='growth',
                prediction=0,
                confidence=0,
                features_importance={},
                model_metadata={},
                timestamp=datetime.now()
            )
    
    def predict_risk(self, metrics: Dict[str, Any]) -> MLPrediction:
        """Predict risk metrics using ML model"""
        try:
            features = self.prepare_features(metrics, 'risk')
            scaled_features = self.scalers['risk'].transform(features)
            
            prediction = self.models['risk'].predict(scaled_features)[0][0]
            
            return MLPrediction(
                symbol=metrics['symbol'],
                target='risk',
                prediction=prediction,
                confidence=0.7,
                features_importance={},
                model_metadata={'model_type': 'deep_learning'},
                timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return MLPrediction(
                symbol=metrics['symbol'],
                target='risk',
                prediction=0,
                confidence=0,
                features_importance={},
                model_metadata={},
                timestamp=datetime.now()
            )
    
    def predict_esg_impact(self, metrics: Dict[str, Any]) -> MLPrediction:
        """Predict ESG impact on valuation"""
        try:
            features = self.prepare_features(metrics, 'esg_impact')
            scaled_features = self.scalers['esg_impact'].transform(features)
            
            prediction = self.models['esg_impact'].predict(scaled_features)[0]
            importance = dict(zip(
                self.models['esg_impact'].feature_names_in_,
                self.models['esg_impact'].feature_importances_
            ))
            
            return MLPrediction(
                symbol=metrics['symbol'],
                target='esg_impact',
                prediction=prediction,
                confidence=0.65,
                features_importance=importance,
                model_metadata={'model_type': 'random_forest'},
                timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Error predicting ESG impact: {e}")
            return MLPrediction(
                symbol=metrics['symbol'],
                target='esg_impact',
                prediction=0,
                confidence=0,
                features_importance={},
                model_metadata={},
                timestamp=datetime.now()
            )
    
    def train_models(self, training_data: pd.DataFrame, target_type: str):
        """Train or update ML models with new data"""
        try:
            features = self.prepare_features(training_data, target_type)
            targets = training_data[f'{target_type}_target'].values
            
            # Scale features
            scaled_features = self.scalers[target_type].fit_transform(features)
            
            # Train model
            if isinstance(self.models[target_type], tf.keras.Model):
                self.models[target_type].fit(
                    scaled_features, targets,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
            else:
                self.models[target_type].fit(scaled_features, targets)
            
            # Save updated models
            joblib.dump(self.models[target_type],
                       os.path.join(self.model_dir, f'{target_type}_model.joblib'))
            joblib.dump(self.scalers[target_type],
                       os.path.join(self.model_dir, f'{target_type}_scaler.joblib'))
            
        except Exception as e:
            logger.error(f"Error training {target_type} model: {e}")
            
    def evaluate_model(self, target_type: str, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            features = self.prepare_features(test_data, target_type)
            targets = test_data[f'{target_type}_target'].values
            
            scaled_features = self.scalers[target_type].transform(features)
            predictions = self.models[target_type].predict(scaled_features)
            
            # Calculate metrics
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            r2 = 1 - (np.sum((targets - predictions) ** 2) /
                     np.sum((targets - np.mean(targets)) ** 2))
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {target_type} model: {e}")
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'r2': 0,
                'timestamp': datetime.now()
            }