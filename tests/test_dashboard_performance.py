"""Performance tests for real-time dashboard"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import logging

from src.dashboard.realtime import DashboardServer
from src.ml.advanced_model import TransformerModel, LSTMModel

class TestDashboardPerformance:
    @pytest.fixture
    def dashboard(self):
        """Initialize dashboard server for testing"""
        return DashboardServer(update_interval=1000)

    @pytest.fixture
    def sample_data_stream(self):
        """Create sample data stream for testing"""
        def generate_tick():
            return {
                'symbol': 'TEST',
                'price': np.random.normal(100, 1),
                'volume': np.random.normal(1000, 100),
                'timestamp': datetime.now()
            }
        return generate_tick

    def measure_response_time(self, func):
        """Measure function execution time"""
        start = time.time()
        result = func()
        end = time.time()
        return end - start, result

    def measure_memory_usage(self):
        """Measure current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def test_update_performance(self, dashboard, sample_data_stream):
        """Test dashboard update performance under load"""
        update_times = []
        memory_usage = []
        
        # Simulate 1000 updates
        for _ in range(1000):
            # Generate test data
            data = sample_data_stream()
            
            # Measure update time
            time_taken, _ = self.measure_response_time(
                lambda: dashboard.data_queue.put(data)
            )
            update_times.append(time_taken)
            
            # Measure memory
            memory_usage.append(self.measure_memory_usage())
        
        avg_update_time = np.mean(update_times)
        max_update_time = np.max(update_times)
        p95_update_time = np.percentile(update_times, 95)
        
        # Log performance metrics
        logging.info(f"Average update time: {avg_update_time:.4f}s")
        logging.info(f"95th percentile update time: {p95_update_time:.4f}s")
        logging.info(f"Max update time: {max_update_time:.4f}s")
        logging.info(f"Final memory usage: {memory_usage[-1]:.2f}MB")
        
        # Assert performance requirements
        assert avg_update_time < 0.1  # Updates should be under 100ms
        assert p95_update_time < 0.2  # 95% of updates under 200ms
        assert max_update_time < 0.5  # No single update over 500ms

    async def test_concurrent_updates(self, dashboard, sample_data_stream):
        """Test dashboard performance with concurrent updates"""
        async def update_task():
            for _ in range(100):
                data = sample_data_stream()
                dashboard.data_queue.put(data)
                await asyncio.sleep(0.01)
        
        # Run 10 concurrent update streams
        tasks = [update_task() for _ in range(10)]
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Calculate throughput
        total_updates = 1000  # 10 tasks * 100 updates
        throughput = total_updates / total_time
        
        logging.info(f"Concurrent throughput: {throughput:.2f} updates/second")
        assert throughput > 100  # Should handle at least 100 updates per second

    def test_memory_leak(self, dashboard, sample_data_stream):
        """Test for memory leaks during extended operation"""
        initial_memory = self.measure_memory_usage()
        memory_samples = []
        
        # Run for 100 iterations
        for _ in range(100):
            # Generate and process batch of updates
            for _ in range(100):
                data = sample_data_stream()
                dashboard.data_queue.put(data)
            
            # Measure memory
            memory_samples.append(self.measure_memory_usage())
            time.sleep(0.1)  # Brief pause between batches
        
        # Calculate memory growth
        memory_growth = memory_samples[-1] - initial_memory
        growth_rate = memory_growth / len(memory_samples)
        
        logging.info(f"Memory growth: {memory_growth:.2f}MB")
        logging.info(f"Growth rate: {growth_rate:.4f}MB/iteration")
        
        # Assert reasonable memory usage
        assert growth_rate < 0.1  # Less than 0.1MB growth per iteration
        assert memory_growth < 50  # Less than 50MB total growth

    def test_chart_rendering_performance(self, dashboard):
        """Test chart rendering performance"""
        def generate_test_data(size):
            dates = pd.date_range('2025-01-01', periods=size, freq='1min')
            return pd.DataFrame({
                'open': np.random.normal(100, 1, size),
                'high': np.random.normal(101, 1, size),
                'low': np.random.normal(99, 1, size),
                'close': np.random.normal(100, 1, size),
                'volume': np.random.normal(1000, 100, size)
            }, index=dates)
        
        data_sizes = [100, 1000, 10000]
        render_times = {}
        
        for size in data_sizes:
            data = generate_test_data(size)
            time_taken, _ = self.measure_response_time(
                lambda: dashboard._create_charts(data)
            )
            render_times[size] = time_taken
            
            logging.info(f"Chart rendering time for {size} points: {time_taken:.4f}s")
            
            # Assert rendering time limits
            assert time_taken < size / 1000  # Should render 1000 points per second

    def test_ml_integration_performance(self, dashboard):
        """Test ML model integration performance"""
        model = TransformerModel()
        data = pd.DataFrame(np.random.randn(1000, 5))
        
        # Measure prediction time
        time_taken, _ = self.measure_response_time(
            lambda: model.predict(data)
        )
        
        logging.info(f"ML prediction time: {time_taken:.4f}s")
        assert time_taken < 1.0  # Predictions should complete within 1 second