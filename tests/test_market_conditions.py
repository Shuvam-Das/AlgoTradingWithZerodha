"""Market simulation and stress tests"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any

from src.strategy import MACrossoverStrategy, RSIOLStrategy
from src.executor import Executor
from src.order_manager import OrderManager
from .test_helpers import MarketSimulator, TestDataGenerator

class TestMarketConditions:
    @pytest.fixture
    def market_data(self):
        """Generate test market data for different conditions"""
        return TestDataGenerator.generate_market_regimes()
    
    @pytest.fixture
    def strategies(self):
        return {
            'mac': MACrossoverStrategy(),
            'rsi': RSIOLStrategy()
        }
    
    @pytest.fixture
    def executor(self):
        return Executor(simulated=True)
    
    def test_trending_market(self, market_data, strategies):
        """Test strategy performance in trending markets"""
        data = market_data['trending_up']
        
        for name, strategy in strategies.items():
            signals = []
            for i in range(len(data)-50):
                window = data.iloc[i:i+50]
                signal = strategy.generate('TEST', window)
                signals.append(signal.signal)
            
            # Calculate signal consistency
            signal_changes = sum(1 for i in range(1, len(signals))
                               if signals[i] != signals[i-1])
            
            # In trending markets, expect fewer signal changes
            assert signal_changes < len(signals) * 0.3
    
    def test_volatile_market(self, market_data, strategies):
        """Test strategy behavior in volatile markets"""
        data = market_data['volatile']
        
        for name, strategy in strategies.items():
            confidence_values = []
            for i in range(len(data)-50):
                window = data.iloc[i:i+50]
                signal = strategy.generate('TEST', window)
                confidence_values.append(signal.confidence)
            
            # In volatile markets, expect lower average confidence
            avg_confidence = np.mean(confidence_values)
            assert avg_confidence < 0.7
    
    def test_gap_handling(self, market_data, strategies):
        """Test handling of price gaps"""
        data = market_data['gap_up']
        
        for name, strategy in strategies.items():
            for i in range(len(data)-50):
                window = data.iloc[i:i+50]
                
                # Check for gaps
                price_changes = window['open'] / window['close'].shift(1) - 1
                gaps = price_changes[abs(price_changes) > 0.02]
                
                if not gaps.empty:
                    # Test strategy response to gaps
                    signal = strategy.generate('TEST', window)
                    
                    # Verify signal is generated with appropriate confidence
                    assert signal.confidence > 0
                    if abs(gaps.iloc[-1]) > 0.05:
                        assert signal.confidence < 0.5  # Lower confidence for large gaps
    
    async def test_rapid_reversal(self, market_data, strategies, executor):
        """Test system behavior during rapid market reversals"""
        # Combine up trend followed by down trend
        up_data = market_data['trending_up']
        down_data = market_data['trending_down']
        data = pd.concat([up_data.iloc[:500], down_data.iloc[500:]])
        
        positions = []
        for i in range(len(data)-50):
            window = data.iloc[i:i+50]
            
            # Get signals from all strategies
            signals = [
                strategy.generate('TEST', window)
                for strategy in strategies.values()
            ]
            
            # Execute signals
            for signal in signals:
                result = await executor.execute_signal(signal.to_dict())
                if result.get('status') == 'completed':
                    positions.append({
                        'entry_price': result['price'],
                        'quantity': result['quantity'],
                        'timestamp': window.index[-1]
                    })
        
        # Calculate drawdown
        if positions:
            prices = data['close']
            max_drawdown = 0
            peak = float('-inf')
            
            for pos in positions:
                price = prices[pos['timestamp']]
                pnl = (price - pos['entry_price']) * pos['quantity']
                peak = max(peak, pnl)
                drawdown = (peak - pnl) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # Assert reasonable drawdown management
            assert max_drawdown < 0.3  # Max 30% drawdown
    
    def test_market_regime_detection(self, market_data):
        """Test market regime detection accuracy"""
        def detect_regime(data: pd.DataFrame) -> str:
            returns = data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            trend = returns.mean() * 252
            
            if volatility > 0.3:
                return 'volatile'
            elif abs(trend) > 0.15:
                return 'trending'
            else:
                return 'sideways'
        
        # Test detection on different regimes
        results = {}
        for regime_name, data in market_data.items():
            detected = detect_regime(data)
            results[regime_name] = detected
        
        # Verify detection accuracy
        assert results['volatile'] == 'volatile'
        assert results['trending_up'] == 'trending'
        assert results['sideways'] == 'sideways'
    
    async def test_liquidity_stress(self, executor):
        """Test system behavior under liquidity stress"""
        simulator = MarketSimulator(volatility=0.05)
        
        # Simulate declining liquidity
        for _ in range(10):
            data = simulator.generate_ohlcv(50)
            signal = {
                'symbol': 'TEST',
                'signal': 'buy',
                'confidence': 0.8
            }
            
            # Attempt increasingly larger orders
            for size in [100, 1000, 10000, 100000]:
                result = await executor.execute_signal(
                    signal,
                    size=size
                )
                
                # Verify proper handling of large orders
                assert result['status'] in ['completed', 'rejected', 'partial']
                if result['status'] == 'rejected':
                    assert size > 10000  # Expect rejection of very large orders
    
    def test_correlation_impact(self, market_data, strategies):
        """Test strategy performance with correlated instruments"""
        # Create correlated data
        base_data = market_data['trending_up']
        correlated_data = base_data * 1.1 + np.random.normal(0, 0.001, len(base_data))
        
        results = []
        for name, strategy in strategies.items():
            # Test on base instrument
            base_signal = strategy.generate('BASE', base_data)
            
            # Test on correlated instrument
            corr_signal = strategy.generate('CORR', correlated_data)
            
            # Compare signals
            results.append({
                'strategy': name,
                'base_signal': base_signal.signal,
                'corr_signal': corr_signal.signal,
                'correlation': np.corrcoef(base_data['close'], correlated_data['close'])[0,1]
            })
        
        # Verify correlation awareness
        for result in results:
            assert abs(result['correlation']) > 0.9  # High correlation
            # Signals should generally align for highly correlated instruments
            assert result['base_signal'] == result['corr_signal']