"""Load testing for data processing components"""
import pytest
import asyncio
import threading
import queue
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
from typing import Dict, List, Any

from src.data_fetcher import fetch_history
from src.storage import store_tick_data, store_order
from src.kite_ws import KiteWebsocket
from src.order_manager import OrderManager
from src.executor import Executor

class TestDataProcessing:
    @pytest.fixture
    def sample_tick_generator(self):
        """Generate sample market ticks"""
        def generate_tick(symbol: str) -> Dict[str, Any]:
            return {
                'tradingsymbol': symbol,
                'last_price': np.random.normal(100, 1),
                'volume': int(np.random.normal(1000, 100)),
                'buy_quantity': int(np.random.normal(500, 50)),
                'sell_quantity': int(np.random.normal(500, 50)),
                'timestamp': pd.Timestamp.now()
            }
        return generate_tick

    def test_tick_processing_throughput(self, sample_tick_generator):
        """Test tick data processing throughput"""
        symbols = [f"TEST{i}" for i in range(100)]
        processed_ticks = 0
        processing_times = []
        
        start_time = time.time()
        
        # Process ticks for 10 seconds
        while time.time() - start_time < 10:
            for symbol in symbols:
                tick = sample_tick_generator(symbol)
                
                tick_start = time.time()
                store_tick_data(tick)
                processing_times.append(time.time() - tick_start)
                
                processed_ticks += 1
        
        total_time = time.time() - start_time
        throughput = processed_ticks / total_time
        avg_latency = np.mean(processing_times) * 1000  # ms
        p99_latency = np.percentile(processing_times, 99) * 1000  # ms
        
        print(f"Tick Processing Results:")
        print(f"Throughput: {throughput:.2f} ticks/second")
        print(f"Average Latency: {avg_latency:.2f}ms")
        print(f"99th Percentile Latency: {p99_latency:.2f}ms")
        
        assert throughput > 1000  # Should handle 1000+ ticks per second
        assert avg_latency < 10  # Average latency under 10ms
        assert p99_latency < 50  # 99th percentile under 50ms

    async def test_concurrent_order_processing(self):
        """Test concurrent order processing capacity"""
        order_manager = OrderManager(simulated=True)
        executor = Executor(simulated=True)
        
        async def place_order(symbol: str, quantity: int):
            return await order_manager.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type='MARKET',
                transaction_type='BUY'
            )
        
        # Place 100 concurrent orders
        symbols = [f"TEST{i}" for i in range(100)]
        quantities = [100] * 100
        
        start_time = time.time()
        orders = await asyncio.gather(*[
            place_order(sym, qty) for sym, qty in zip(symbols, quantities)
        ])
        total_time = time.time() - start_time
        
        successful_orders = len([o for o in orders if o.get('status') == 'COMPLETE'])
        throughput = len(orders) / total_time
        
        print(f"Order Processing Results:")
        print(f"Throughput: {throughput:.2f} orders/second")
        print(f"Success Rate: {successful_orders/len(orders)*100:.2f}%")
        
        assert throughput > 10  # Should handle 10+ orders per second
        assert successful_orders / len(orders) > 0.95  # 95% success rate

    def test_historical_data_load(self):
        """Test historical data loading performance"""
        symbols = [f"TEST{i}" for i in range(10)]
        lookback_days = 365
        
        def fetch_symbol_data(symbol: str) -> pd.DataFrame:
            return fetch_history(
                symbol=symbol,
                start_date=pd.Timestamp.now() - pd.Timedelta(days=lookback_days),
                end_date=pd.Timestamp.now()
            )
        
        start_time = time.time()
        
        # Fetch data concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(fetch_symbol_data, symbols))
        
        total_time = time.time() - start_time
        total_data_points = sum(len(df) for df in results if df is not None)
        throughput = total_data_points / total_time
        
        print(f"Historical Data Loading Results:")
        print(f"Total Data Points: {total_data_points}")
        print(f"Loading Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} points/second")
        
        assert throughput > 10000  # Should load 10k+ data points per second

    def test_memory_efficiency(self, sample_tick_generator):
        """Test memory efficiency under load"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        tick_queue = queue.Queue()
        
        def tick_producer():
            for _ in range(10000):
                tick = sample_tick_generator("TEST")
                tick_queue.put(tick)
                time.sleep(0.001)
        
        def tick_consumer():
            while True:
                try:
                    tick = tick_queue.get(timeout=1)
                    store_tick_data(tick)
                except queue.Empty:
                    break
        
        # Start producer and consumer threads
        producer = threading.Thread(target=tick_producer)
        consumer = threading.Thread(target=tick_consumer)
        
        producer.start()
        consumer.start()
        
        memory_samples = []
        start_time = time.time()
        
        # Monitor memory usage
        while producer.is_alive() or consumer.is_alive():
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(memory)
            time.sleep(0.1)
        
        producer.join()
        consumer.join()
        
        final_memory = memory_samples[-1]
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory
        
        print(f"Memory Usage Results:")
        print(f"Initial Memory: {initial_memory:.2f}MB")
        print(f"Final Memory: {final_memory:.2f}MB")
        print(f"Peak Memory: {max_memory:.2f}MB")
        print(f"Memory Growth: {memory_growth:.2f}MB")
        
        assert memory_growth < 100  # Less than 100MB growth
        assert max_memory - initial_memory < 200  # Peak usage within 200MB

    def test_data_integrity(self, sample_tick_generator):
        """Test data integrity under load"""
        symbols = [f"TEST{i}" for i in range(10)]
        ticks_per_symbol = 1000
        verification_data = {}
        
        # Generate and store test data
        for symbol in symbols:
            verification_data[symbol] = []
            for _ in range(ticks_per_symbol):
                tick = sample_tick_generator(symbol)
                store_tick_data(tick)
                verification_data[symbol].append(tick)
        
        # Verify stored data
        for symbol in symbols:
            stored_data = fetch_history(symbol, pd.Timestamp.now() - pd.Timedelta(hours=1))
            original_data = pd.DataFrame(verification_data[symbol])
            
            # Compare key statistics
            assert len(stored_data) == len(original_data)
            assert np.allclose(
                stored_data['last_price'].mean(),
                original_data['last_price'].mean(),
                rtol=1e-5
            )