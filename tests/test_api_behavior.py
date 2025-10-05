"""Rate limit and API behavior tests"""
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

from src.kite_client import KiteClient
from src.order_manager import OrderManager
from src.executor import Executor
from .test_helpers import MockBrokerAPI

class TestAPIBehavior:
    @pytest.fixture
    def mock_broker(self):
        return MockBrokerAPI(latency=0.05, error_rate=0.01)
    
    @pytest.fixture
    def order_manager(self, mock_broker):
        order_manager = OrderManager(simulated=True)
        order_manager.client = mock_broker
        return order_manager
    
    async def test_rate_limit_handling(self, order_manager):
        """Test handling of API rate limits"""
        # Configure rate limits
        rate_limit = 5  # orders per second
        orders_to_test = 20
        
        start_time = time.time()
        success_count = 0
        rate_limit_errors = 0
        
        # Try to place orders rapidly
        for i in range(orders_to_test):
            try:
                await order_manager.place_order(
                    symbol='TEST',
                    quantity=100,
                    order_type='MARKET',
                    transaction_type='BUY'
                )
                success_count += 1
            except Exception as e:
                if "Rate limit exceeded" in str(e):
                    rate_limit_errors += 1
                await asyncio.sleep(0.2)  # Back off on rate limit
        
        duration = time.time() - start_time
        orders_per_second = success_count / duration
        
        print(f"Orders per second: {orders_per_second:.2f}")
        print(f"Rate limit errors: {rate_limit_errors}")
        
        # Verify rate limiting worked
        assert orders_per_second <= rate_limit * 1.1  # Allow 10% margin
        assert rate_limit_errors > 0  # Should hit some rate limits
    
    async def test_concurrent_order_limits(self, order_manager):
        """Test behavior with concurrent order placement"""
        async def place_order():
            return await order_manager.place_order(
                symbol='TEST',
                quantity=100,
                order_type='MARKET',
                transaction_type='BUY'
            )
        
        # Try to place multiple orders concurrently
        tasks = [place_order() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successes = len([r for r in results if not isinstance(r, Exception)])
        rate_limits = len([r for r in results if isinstance(r, Exception) and "Rate limit" in str(r)])
        
        assert successes <= 5  # Should not exceed rate limit
        assert rate_limits > 0  # Should hit rate limits
    
    def test_api_throttling(self, order_manager):
        """Test API throttling mechanism"""
        calls = []
        start_time = time.time()
        
        # Make API calls with tracking
        for _ in range(10):
            try:
                order_manager.get_positions()
                calls.append(time.time())
            except Exception as e:
                assert "Rate limit" in str(e)
        
        # Calculate intervals between calls
        intervals = np.diff(calls)
        
        # Verify minimum interval is maintained
        assert min(intervals) >= 0.2  # Minimum 200ms between calls
    
    async def test_error_retry_mechanism(self, order_manager):
        """Test error retry behavior"""
        retries = []
        
        # Mock error response
        async def failing_api_call():
            retries.append(time.time())
            if len(retries) < 3:
                raise Exception("API Error")
            return {"status": "success"}
        
        # Test retry mechanism
        result = await order_manager._retry_api_call(
            failing_api_call,
            max_retries=3,
            retry_delay=0.1
        )
        
        assert len(retries) == 3  # Should take 3 attempts
        assert result["status"] == "success"
        
        # Verify retry intervals
        intervals = np.diff(retries)
        assert all(interval >= 0.1 for interval in intervals)
    
    def test_api_timeout_handling(self, order_manager):
        """Test handling of API timeouts"""
        async def slow_api_call():
            await asyncio.sleep(2)  # Simulate slow response
            return {"status": "success"}
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_api_call(), timeout=1.0)
    
    @pytest.mark.parametrize("error_rate", [0.2, 0.5, 0.8])
    async def test_api_reliability(self, error_rate, mock_broker):
        """Test system behavior under different API error rates"""
        mock_broker.error_rate = error_rate
        order_manager = OrderManager(simulated=True)
        order_manager.client = mock_broker
        
        attempts = 50
        successes = 0
        
        for _ in range(attempts):
            try:
                await order_manager.place_order(
                    symbol='TEST',
                    quantity=100,
                    order_type='MARKET',
                    transaction_type='BUY'
                )
                successes += 1
            except Exception:
                pass
        
        success_rate = successes / attempts
        expected_success_rate = 1 - error_rate
        
        # Allow 10% deviation from expected rate
        assert abs(success_rate - expected_success_rate) < 0.1
    
    async def test_api_load_recovery(self, order_manager):
        """Test API recovery after high load"""
        # Generate high load
        tasks = []
        for _ in range(20):
            task = order_manager.place_order(
                symbol='TEST',
                quantity=100,
                order_type='MARKET',
                transaction_type='BUY'
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        initial_errors = len([r for r in results if isinstance(r, Exception)])
        
        # Wait for recovery period
        await asyncio.sleep(2)
        
        # Test after recovery
        success = await order_manager.place_order(
            symbol='TEST',
            quantity=100,
            order_type='MARKET',
            transaction_type='BUY'
        )
        
        assert success['status'] == 'COMPLETE'
        assert initial_errors > 0  # Should have hit rate limits initially