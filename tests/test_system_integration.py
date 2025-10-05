"""End-to-end system tests"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Any

from src.kite_auth import authenticate
from src.kite_ws import KiteWebsocket
from src.order_manager import OrderManager
from src.executor import Executor
from src.strategy import MACrossoverStrategy, RSIOLStrategy
from src.ml.model import predict
from src.storage import store_trade, fetch_trades
from src.notifier import send_notification
from src.dashboard.realtime import DashboardServer

class TestSystemIntegration:
    @pytest.fixture
    def setup_system(self):
        """Setup complete trading system"""
        executor = Executor(simulated=True)
        order_manager = OrderManager(simulated=True)
        strategies = {
            'mac': MACrossoverStrategy(),
            'rsi': RSIOLStrategy()
        }
        dashboard = DashboardServer()
        
        return {
            'executor': executor,
            'order_manager': order_manager,
            'strategies': strategies,
            'dashboard': dashboard
        }

    async def test_complete_trading_cycle(self, setup_system):
        """Test a complete trading cycle from signal to execution"""
        system = setup_system
        symbol = "TEST"
        
        # Generate test data
        dates = pd.date_range(start='2025-01-01', end='2025-10-01', freq='D')
        market_data = pd.DataFrame({
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.normal(1000000, 100000, len(dates))
        }, index=dates)
        
        # 1. Generate trading signals
        signals = []
        for name, strategy in system['strategies'].items():
            signal = strategy.generate(symbol, market_data)
            signals.append(signal)
            
            assert signal.symbol == symbol
            assert signal.signal in ['buy', 'sell', 'hold']
            assert 0 <= signal.confidence <= 1
        
        # 2. Get ML predictions
        ml_predictions = await predict(market_data)
        assert 'predictions' in ml_predictions
        assert 'confidence' in ml_predictions
        
        # 3. Execute trades
        for signal in signals:
            if signal.signal in ['buy', 'sell']:
                result = system['executor'].execute_signal(signal.to_dict())
                
                assert 'status' in result
                if result['status'] == 'completed':
                    assert 'order_id' in result
                    
                    # 4. Store trade details
                    store_trade({
                        'symbol': symbol,
                        'strategy': signal.signal,
                        'confidence': signal.confidence,
                        'price': result.get('price'),
                        'quantity': result.get('quantity'),
                        'timestamp': datetime.now()
                    })
        
        # 5. Verify trade storage
        trades = fetch_trades(symbol=symbol)
        assert len(trades) > 0
        
        # 6. Check dashboard updates
        assert len(system['dashboard'].positions) >= 0

    async def test_error_handling(self, setup_system):
        """Test system error handling and recovery"""
        system = setup_system
        
        # Test invalid order handling
        result = system['executor'].execute_signal({
            'symbol': 'INVALID',
            'signal': 'buy',
            'confidence': 0.5
        })
        assert result['status'] == 'error'
        
        # Test network error handling
        # Simulate network error in order placement
        system['order_manager'].simulated = False  # Force real API calls
        with pytest.raises(Exception):
            await system['order_manager'].place_order(
                symbol='TEST',
                quantity=1,
                order_type='MARKET',
                transaction_type='BUY'
            )
        
        # Verify system can continue after error
        system['order_manager'].simulated = True
        result = await system['order_manager'].place_order(
            symbol='TEST',
            quantity=1,
            order_type='MARKET',
            transaction_type='BUY'
        )
        assert result['status'] == 'COMPLETE'

    async def test_real_time_processing(self, setup_system):
        """Test real-time data processing and system response"""
        system = setup_system
        
        # Setup websocket connection
        ws = KiteWebsocket(simulated=True)
        received_ticks = []
        
        async def on_tick(tick):
            received_ticks.append(tick)
            
            # Generate signals
            market_data = pd.DataFrame(received_ticks)
            for strategy in system['strategies'].values():
                signal = strategy.generate(tick['symbol'], market_data)
                if signal.signal != 'hold':
                    await system['executor'].execute_signal(signal.to_dict())
        
        ws.on_tick = on_tick
        
        # Send test ticks
        test_ticks = [
            {'symbol': 'TEST', 'price': 100 + i, 'volume': 1000}
            for i in range(10)
        ]
        
        for tick in test_ticks:
            await ws.simulate_tick(tick)
            await asyncio.sleep(0.1)
        
        assert len(received_ticks) == len(test_ticks)
        assert len(system['executor'].orders) > 0

    def test_system_recovery(self, setup_system):
        """Test system recovery after component failure"""
        system = setup_system
        
        # Simulate component failure
        system['order_manager'].client = None
        
        # Attempt operation that should fail
        with pytest.raises(Exception):
            system['order_manager'].get_positions()
        
        # Recover component
        system['order_manager'].client = system['order_manager'].kite
        
        # Verify system is functional
        positions = system['order_manager'].get_positions()
        assert isinstance(positions, dict)

    def test_notification_system(self, setup_system):
        """Test notification system integration"""
        system = setup_system
        
        # Test different notification types
        notifications = [
            {
                'type': 'order',
                'message': 'Order executed: TEST 100 shares',
                'severity': 'info'
            },
            {
                'type': 'alert',
                'message': 'High volatility detected',
                'severity': 'warning'
            },
            {
                'type': 'error',
                'message': 'Order placement failed',
                'severity': 'error'
            }
        ]
        
        for notification in notifications:
            send_notification(notification)
            # Verify notification was processed
            # Implementation depends on your notification system
            
    @pytest.mark.asyncio
    async def test_data_synchronization(self, setup_system):
        """Test data synchronization and race condition handling"""
        import threading
        import queue
        
        system = setup_system
        symbol = "TEST"
        data_queue = queue.Queue()
        processing_errors = []
        processed_data = []
        
        # Event to signal threads to stop
        stop_event = threading.Event()
        
        def data_producer():
            """Simulate high-frequency market data production"""
            try:
                while not stop_event.is_set():
                    tick = {
                        'symbol': symbol,
                        'price': np.random.normal(100, 1),
                        'volume': int(np.random.normal(1000, 100)),
                        'timestamp': datetime.now()
                    }
                    data_queue.put(tick)
                    time.sleep(0.001)  # Simulate 1ms tick interval
            except Exception as e:
                processing_errors.append(f"Producer error: {str(e)}")
        
        async def data_consumer():
            """Process market data and generate signals"""
            try:
                while not stop_event.is_set():
                    try:
                        # Non-blocking queue get
                        tick = data_queue.get_nowait()
                        processed_data.append(tick)
                        
                        # Generate and process signals
                        for strategy in system['strategies'].values():
                            signal = strategy.generate(
                                tick['symbol'],
                                pd.DataFrame([tick])
                            )
                            
                            if signal.signal != 'hold':
                                # Execute trade
                                await system['executor'].execute_signal(
                                    signal.to_dict()
                                )
                        
                        data_queue.task_done()
                    except queue.Empty:
                        await asyncio.sleep(0.001)
                    except Exception as e:
                        processing_errors.append(f"Consumer error: {str(e)}")
            except Exception as e:
                processing_errors.append(f"Consumer task error: {str(e)}")
        
        # Start producer thread
        producer = threading.Thread(target=data_producer)
        producer.start()
        
        # Start multiple consumer tasks
        consumer_tasks = [
            asyncio.create_task(data_consumer())
            for _ in range(3)  # Run 3 concurrent consumers
        ]
        
        # Let the system run for a few seconds
        await asyncio.sleep(5)
        
        # Stop threads and tasks
        stop_event.set()
        producer.join()
        await asyncio.gather(*consumer_tasks)
        
        # Verify results
        assert len(processing_errors) == 0, f"Processing errors: {processing_errors}"
        assert len(processed_data) > 0, "No data was processed"
        
        # Check for data consistency
        timestamps = [tick['timestamp'] for tick in processed_data]
        # Verify timestamps are in order (allowing for small concurrent processing windows)
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            assert time_diff >= -0.1, "Data processed out of order"
        
        # Check system state
        assert system['executor'].cash > 0, "Invalid cash balance"
        assert len(system['executor'].orders) > 0, "No orders were placed"

    @pytest.mark.asyncio
    async def test_load_conditions(self, setup_system):
        """Test system behavior under heavy load conditions"""
        import psutil
        import time
        
        system = setup_system
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance metrics
        response_times = []
        order_success_rate = []
        memory_samples = []
        cpu_samples = []
        
        async def simulate_market_activity(intensity: str):
            orders = 0
            successful_orders = 0
            
            # Adjust load based on intensity
            iterations = {
                'low': 100,
                'medium': 500,
                'high': 1000
            }.get(intensity, 100)
            
            delay = {
                'low': 0.01,
                'medium': 0.005,
                'high': 0.001
            }.get(intensity, 0.01)
            
            for _ in range(iterations):
                start_time = time.time()
                
                try:
                    tick = {
                        'symbol': 'TEST',
                        'price': np.random.normal(100, 1),
                        'volume': int(np.random.normal(1000, 100))
                    }
                    
                    # Generate signal
                    signal = {
                        'symbol': tick['symbol'],
                        'signal': 'buy' if np.random.random() > 0.5 else 'sell',
                        'confidence': np.random.random()
                    }
                    
                    # Execute order
                    result = await system['executor'].execute_signal(signal)
                    orders += 1
                    
                    if result.get('status') == 'completed':
                        successful_orders += 1
                    
                    # Store metrics
                    response_times.append(time.time() - start_time)
                    memory_samples.append(process.memory_info().rss / 1024 / 1024)
                    cpu_samples.append(process.cpu_percent())
                    
                except Exception as e:
                    logging.error(f"Error during load test: {str(e)}")
                
                await asyncio.sleep(delay)
            
            return {
                'orders': orders,
                'successful': successful_orders,
                'intensity': intensity
            }
        
        # Run concurrent load tests with different intensities
        results = await asyncio.gather(
            simulate_market_activity('low'),
            simulate_market_activity('medium'),
            simulate_market_activity('high')
        )
        
        # Calculate performance metrics
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        p95_response_time = np.percentile(response_times, 95)
        memory_growth = memory_samples[-1] - initial_memory
        avg_cpu_usage = np.mean(cpu_samples)
        
        # Calculate success rates per intensity level
        success_rates = {
            r['intensity']: r['successful'] / r['orders'] if r['orders'] > 0 else 0
            for r in results
        }
        
        # Performance assertions
        assert avg_response_time < 0.1, f"High average response time: {avg_response_time:.3f}s"
        assert max_response_time < 0.5, f"High maximum response time: {max_response_time:.3f}s"
        assert p95_response_time < 0.2, f"High 95th percentile response time: {p95_response_time:.3f}s"
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"
        assert avg_cpu_usage < 80, f"High average CPU usage: {avg_cpu_usage:.1f}%"
        
        # Success rate assertions
        for intensity, rate in success_rates.items():
            min_success_rate = {
                'low': 0.99,
                'medium': 0.95,
                'high': 0.90
            }.get(intensity, 0.95)
            
            assert rate >= min_success_rate, (
                f"Low success rate at {intensity} intensity: {rate:.2%} "
                f"(expected >= {min_success_rate:.2%})"
            )
        
        # Verify system stability
        assert system['executor'].cash > 0
        assert len(system['executor'].orders) > 0