"""Machine Learning predictor integration for dashboard"""
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ..ml.model import get_latest_predictions
from ..ml.advanced_model import TransformerModel, LSTMModel


class MLDashboardIntegrator:
    """Integrates ML predictions and insights into the dashboard"""
    
    def __init__(self):
        self.transformer_model = TransformerModel()
        self.lstm_model = LSTMModel()
        self.last_predictions: Dict[str, Dict[str, Any]] = {}
        self.prediction_confidence: Dict[str, float] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
    async def update_predictions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get updated predictions from ML models"""
        # Get predictions from multiple models
        transformer_pred = await self.transformer_model.predict_async(market_data)
        lstm_pred = await self.lstm_model.predict_async(market_data)
        
        # Ensemble the predictions
        ensemble_pred = self._ensemble_predictions([
            ('transformer', transformer_pred, 0.6),  # Higher weight for transformer
            ('lstm', lstm_pred, 0.4)
        ])
        
        # Update model performance metrics
        self._update_model_performance()
        
        return {
            'predictions': ensemble_pred,
            'confidence': self.prediction_confidence,
            'model_performance': self.model_performance
        }
    
    def _ensemble_predictions(self, 
                            model_preds: List[tuple[str, Dict[str, Any], float]]
                            ) -> Dict[str, Any]:
        """Combine predictions from multiple models using weighted average"""
        ensemble_results = {}
        
        for symbol in model_preds[0][1].keys():
            weighted_pred = 0
            weighted_conf = 0
            total_weight = 0
            
            for model_name, preds, weight in model_preds:
                if symbol in preds:
                    pred_value = preds[symbol]['prediction']
                    conf_value = preds[symbol]['confidence']
                    
                    weighted_pred += pred_value * weight * conf_value
                    weighted_conf += conf_value * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_results[symbol] = {
                    'prediction': weighted_pred / total_weight,
                    'confidence': weighted_conf / total_weight
                }
                
                self.prediction_confidence[symbol] = weighted_conf / total_weight
        
        return ensemble_results
    
    def _update_model_performance(self):
        """Update model performance metrics"""
        for model_name in ['transformer', 'lstm']:
            metrics = self._calculate_model_metrics(model_name)
            self.model_performance[model_name] = metrics
    
    def _calculate_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Calculate performance metrics for a specific model"""
        # In production, this would calculate based on actual predictions vs outcomes
        return {
            'accuracy': 0.85,  # Placeholder
            'precision': 0.83,
            'recall': 0.82,
            'f1_score': 0.84,
            'roi': 0.12
        }
    
    def get_trading_insights(self) -> List[Dict[str, Any]]:
        """Generate trading insights based on ML predictions"""
        insights = []
        
        for symbol, pred_data in self.last_predictions.items():
            confidence = self.prediction_confidence.get(symbol, 0)
            
            if confidence > 0.8:  # High confidence threshold
                insight = {
                    'symbol': symbol,
                    'type': 'strong_signal',
                    'direction': 'buy' if pred_data['prediction'] > 0 else 'sell',
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'source': 'ml_ensemble'
                }
                insights.append(insight)
        
        return insights
    
    def get_risk_adjustments(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML-based risk adjustment recommendations"""
        adjustments = {}
        
        for symbol, position in portfolio.items():
            if symbol in self.last_predictions:
                pred = self.last_predictions[symbol]
                conf = self.prediction_confidence[symbol]
                
                # Calculate recommended position adjustments
                current_size = position['size']
                target_size = self._calculate_optimal_size(
                    current_size, pred['prediction'], conf)
                
                if abs(target_size - current_size) > current_size * 0.1:  # 10% threshold
                    adjustments[symbol] = {
                        'current_size': current_size,
                        'target_size': target_size,
                        'confidence': conf,
                        'reason': 'ml_risk_adjustment'
                    }
        
        return adjustments
    
    def _calculate_optimal_size(self, 
                              current_size: float, 
                              prediction: float, 
                              confidence: float) -> float:
        """Calculate optimal position size based on ML signals"""
        # Basic position sizing based on prediction and confidence
        adjustment_factor = prediction * confidence
        return current_size * (1 + adjustment_factor)