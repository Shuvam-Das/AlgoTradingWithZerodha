"""Test helper utilities for the trading system"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import logging
import json

class MarketSimulator:
    """Simulates market conditions and data generation"""
    
    def __init__(self, 
                volatility: float = 0.02,
                trend: float = 0.0,
                gap_probability: float = 0.05):
        self.volatility = volatility
        self.trend = trend
        self.gap_probability = gap_probability
        self.last_price = 100.0
        
    def generate_ohlcv(self, periods: int) -> pd.DataFrame:
        """Generate OHLCV data with realistic market behavior"""
        dates = pd.date_range(start='2025-01-01', periods=periods, freq='1min')
        data = []
        
        for i in range(periods):
            # Add trend and random walk
            random_walk = np.random.normal(self.trend, self.volatility)
            
            # Simulate price gaps
            if np.random.random() < self.gap_probability:
                random_walk *= 3
            
            self.last_price *= (1 + random_walk)
            
            # Generate realistic OHLCV
            high_low_spread = abs(np.random.normal(0, self.volatility))
            open_close_spread = abs(np.random.normal(0, self.volatility/2))
            
            open_price = self.last_price * (1 + np.random.normal(0, open_close_spread))
            close_price = self.last_price
            high_price = max(open_price, close_price) * (1 + high_low_spread)
            low_price = min(open_price, close_price) * (1 - high_low_spread)
            volume = abs(int(np.random.normal(100000, 20000)))
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)

    def generate_tick_stream(self) -> Dict[str, Any]:
        """Generate realistic tick data"""
        tick_change = np.random.normal(0, self.volatility/10)
        self.last_price *= (1 + tick_change)
        
        return {
            'last_price': self.last_price,
            'volume': abs(int(np.random.normal(1000, 200))),
            'buy_quantity': abs(int(np.random.normal(500, 100))),
            'sell_quantity': abs(int(np.random.normal(500, 100))),
            'timestamp': pd.Timestamp.now()
        }

class MockBrokerAPI:
    """Mock broker API for testing"""
    
    def __init__(self, latency: float = 0.05, error_rate: float = 0.01):
        self.latency = latency
        self.error_rate = error_rate
        self.orders: List[Dict[str, Any]] = []
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.rate_limits = {
            'orders_per_second': 5,
            'last_order_time': datetime.now(),
            'order_count': 0
        }
    
    async def place_order(self, **kwargs) -> Dict[str, Any]:
        """Simulate order placement with realistic behavior"""
        # Simulate network latency
        await asyncio.sleep(self.latency)
        
        # Rate limit check
        now = datetime.now()
        if (now - self.rate_limits['last_order_time']).total_seconds() < 1:
            if self.rate_limits['order_count'] >= self.rate_limits['orders_per_second']:
                raise Exception("Rate limit exceeded")
        else:
            self.rate_limits['order_count'] = 0
            self.rate_limits['last_order_time'] = now
        
        self.rate_limits['order_count'] += 1
        
        # Random error simulation
        if np.random.random() < self.error_rate:
            raise Exception("Simulated broker API error")
        
        order_id = f"ORDER_{len(self.orders)}"
        order = {
            'order_id': order_id,
            'status': 'COMPLETE',
            **kwargs
        }
        self.orders.append(order)
        
        # Update positions
        symbol = kwargs.get('symbol')
        qty = kwargs.get('quantity', 0)
        if symbol:
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0}
            self.positions[symbol]['quantity'] += qty
        
        return order
    
    def get_positions(self) -> Dict[str, Any]:
        """Return current positions"""
        return self.positions
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Simulate order cancellation"""
        await asyncio.sleep(self.latency)
        
        for order in self.orders:
            if order['order_id'] == order_id:
                order['status'] = 'CANCELLED'
                return order
        
        raise Exception("Order not found")

class TestDataGenerator:
    """Generates test data for various scenarios"""
    
    @staticmethod
    def generate_market_regimes() -> Dict[str, pd.DataFrame]:
        """Generate data for different market regimes"""
        regimes = {
            'trending_up': MarketSimulator(trend=0.001, volatility=0.01),
            'trending_down': MarketSimulator(trend=-0.001, volatility=0.01),
            'volatile': MarketSimulator(volatility=0.03, trend=0),
            'sideways': MarketSimulator(volatility=0.005, trend=0),
            'gap_up': MarketSimulator(gap_probability=0.1, trend=0.002),
            'gap_down': MarketSimulator(gap_probability=0.1, trend=-0.002)
        }
        
        return {
            name: simulator.generate_ohlcv(1000)
            for name, simulator in regimes.items()
        }
    
    @staticmethod
    def generate_order_scenarios() -> List[Dict[str, Any]]:
        """Generate various order scenarios"""
        return [
            {
                'type': 'market_buy',
                'symbol': 'TEST',
                'quantity': 100,
                'price': None
            },
            {
                'type': 'limit_buy',
                'symbol': 'TEST',
                'quantity': 100,
                'price': 99.5
            },
            {
                'type': 'stop_loss',
                'symbol': 'TEST',
                'quantity': -100,
                'trigger_price': 95.0
            }
        ]
    
    @staticmethod
    def generate_error_scenarios() -> List[Dict[str, Any]]:
        """Generate error test scenarios"""
        return [
            {
                'type': 'invalid_symbol',
                'symbol': 'INVALID',
                'expected_error': 'Invalid symbol'
            },
            {
                'type': 'insufficient_funds',
                'symbol': 'TEST',
                'quantity': 1000000,
                'expected_error': 'Insufficient funds'
            },
            {
                'type': 'rate_limit',
                'orders': [{'symbol': 'TEST', 'quantity': 100}] * 10,
                'expected_error': 'Rate limit exceeded'
            }
        ]

@pytest.fixture
def market_simulator():
    """Fixture for market simulator"""
    return MarketSimulator()

@pytest.fixture
def mock_broker():
    """Fixture for mock broker"""
    return MockBrokerAPI()

@pytest.fixture
def test_data():
    """Fixture for test data generator"""
    return TestDataGenerator()

def assert_dataframe_valid(df: pd.DataFrame):
    """Validate DataFrame structure and data"""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_columns)
    assert not df.empty
    assert not df.isnull().any().any()
    assert (df['high'] >= df['low']).all()
    assert (df['high'] >= df['open']).all()
    assert (df['high'] >= df['close']).all()
    assert (df['low'] <= df['open']).all()
    assert (df['low'] <= df['close']).all()
    assert (df['volume'] >= 0).all()

def assert_order_valid(order: Dict[str, Any]):
    """Validate order structure"""
    required_fields = ['order_id', 'status', 'symbol', 'quantity']
    assert all(field in order for field in required_fields)
    assert order['status'] in ['OPEN', 'COMPLETE', 'CANCELLED', 'REJECTED']
    assert isinstance(order['quantity'], (int, float))
    assert isinstance(order['symbol'], str)

def assert_position_valid(position: Dict[str, Any]):
    """Validate position structure"""
    required_fields = ['quantity']
    assert all(field in position for field in required_fields)
    assert isinstance(position['quantity'], (int, float))