"""Integration tests for ML pipeline"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import asyncio
from typing import List, Dict, Any

from src.ml.advanced_model import TransformerModel, LSTMModel
from src.ml.model import load_model, save_model, predict, train_model
from src.data_fetcher import fetch_history
from src.storage import store_prediction

class TestMLPipeline:
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2025-01-01', end='2025-10-01', freq='D')
        return pd.DataFrame({
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.normal(1000000, 100000, len(dates))
        }, index=dates)

    @pytest.fixture
    def models(self):
        """Initialize models for testing"""
        return {
            'transformer': TransformerModel(),
            'lstm': LSTMModel()
        }

    async def test_model_training(self, sample_data, models):
        """Test model training pipeline"""
        for name, model in models.items():
            # Train model
            train_data = sample_data[:'2025-08-01']
            val_data = sample_data['2025-08-01':]
            
            result = await train_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                epochs=2
            )
            
            assert 'train_loss' in result
            assert 'val_loss' in result
            assert result['train_loss'][-1] < result['train_loss'][0]

    async def test_model_prediction(self, sample_data, models):
        """Test prediction pipeline"""
        for name, model in models.items():
            predictions = await model.predict_async(sample_data)
            
            assert 'predictions' in predictions
            assert 'confidence' in predictions
            assert len(predictions['predictions']) > 0
            assert all(0 <= conf <= 1 for conf in predictions['confidence'])

    async def test_model_persistence(self, models, tmp_path):
        """Test model saving and loading"""
        for name, model in models.items():
            model_path = tmp_path / f"{name}_test.pt"
            
            # Save model
            save_model(model, str(model_path))
            assert model_path.exists()
            
            # Load model
            loaded_model = load_model(model.__class__, str(model_path))
            assert isinstance(loaded_model, model.__class__)
            
            # Verify weights are the same
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                assert torch.equal(p1, p2)

    def test_prediction_storage(self, sample_data, models):
        """Test prediction storage and retrieval"""
        symbol = "TEST"
        
        for name, model in models.items():
            predictions = {
                'predictions': np.array([1.0, 2.0, 3.0]),
                'confidence': np.array([0.8, 0.9, 0.7])
            }
            
            # Store predictions
            store_prediction(
                symbol=symbol,
                model_name=name,
                predictions=predictions['predictions'],
                confidence=predictions['confidence'],
                timestamp=datetime.now()
            )
            
            # Verify storage was successful
            # Add verification based on your storage implementation

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, sample_data, models):
        """Test complete ML pipeline end-to-end"""
        symbol = "TEST"
        
        for name, model in models.items():
            # Train
            train_result = await train_model(
                model=model,
                train_data=sample_data,
                val_data=sample_data,
                epochs=1
            )
            assert 'train_loss' in train_result
            
            # Predict
            predictions = await model.predict_async(sample_data)
            assert 'predictions' in predictions
            
            # Store
            store_prediction(
                symbol=symbol,
                model_name=name,
                predictions=predictions['predictions'],
                confidence=predictions['confidence'],
                timestamp=datetime.now()
            )

    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, sample_data, models):
        """Test concurrent prediction processing"""
        symbols = ["TEST1", "TEST2", "TEST3"]
        
        async def predict_symbol(model, symbol):
            return await model.predict_async(sample_data)
        
        for name, model in models.items():
            # Run predictions concurrently
            tasks = [predict_symbol(model, sym) for sym in symbols]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(symbols)
            for result in results:
                assert 'predictions' in result
                assert 'confidence' in result