"""Advanced ML models for stock prediction and reinforcement learning trading strategies.

This module implements advanced machine learning models for stock prediction and trading:
1. Deep Learning models (LSTM, Transformer) for price prediction
2. Reinforcement Learning for trading policy optimization 
3. Sentiment analysis for news/social media
4. Ensemble methods combining multiple signals
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import joblib
import os
import logging
from datetime import datetime
from prophet import Prophet

logger = logging.getLogger('atwz.ml.advanced')

class LSTMPredictor(nn.Module):
    """LSTM model for sequence prediction"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TransformerPredictor(tf.keras.Model):
    """Transformer model for sequence prediction"""
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, output_dim: int):
        super().__init__()
        self.input_proj = tf.keras.layers.Dense(d_model)
        self.transformer = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model*4, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(output_dim)
        
    def call(self, inputs):
        x = self.input_proj(inputs)
        attn_output = self.transformer(x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return self.output_layer(out2)

class ReinforcementLearningTrader:
    """RL model for trading policy optimization"""
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.model.predict(state)
        return np.argmax(action_probs[0])
    
    def train(self, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        self.model.train_on_batch(states, actions)

class SentimentAnalyzer:
    """Sentiment analysis for news and social media"""
    def __init__(self):
        from transformers import pipeline
        self.analyzer = pipeline("sentiment-analysis")
    
    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        return self.analyzer(texts)

class EnsemblePredictor:
    """Ensemble model combining multiple signals"""
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100),
            'gb': GradientBoostingRegressor(n_estimators=100)
        }
        self.prophet = Prophet(daily_seasonality=True)
        self.scaler = MinMaxScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and prepare features"""
        features = pd.DataFrame()
        # Price-based
        features['returns'] = df['close'].pct_change()
        features['volatility'] = features['returns'].rolling(window=20).std()
        # Volume
        features['volume_ma'] = df['volume'].rolling(window=20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        # Technical indicators
        features['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        features['macd'] = ta.trend.MACD(df['close']).macd()
        # Clean and normalize
        features = features.dropna()
        features = pd.DataFrame(self.scaler.fit_transform(features), 
                              columns=features.columns,
                              index=features.index)
        return features
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train all models in ensemble"""
        for name, model in self.models.items():
            model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions from all models"""
        predictions = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"Prediction failed for model {name}: {e}")
        return predictions

class MLModelRegistry:
    """Model versioning and lifecycle management"""
    def __init__(self, base_dir: str = 'models'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_model(self, model: Any, name: str, metadata: Dict[str, Any]) -> str:
        """Save model with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = os.path.join(self.base_dir, f"{name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return model_dir
    
    def load_model(self, model_dir: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata"""
        model_path = os.path.join(model_dir, 'model.joblib')
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata

# Helper functions for feature engineering
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate common technical indicators"""
    import talib
    
    df = df.copy()
    
    # Basic price indicators
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'])
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal
    df['MACD_hist'] = macdhist
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'])
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    
    # Volume indicators
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    return df

def prepare_sequences(data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for time series models"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:(i + seq_len)])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate common ML metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return results