"""Backtesting module for fundamental analysis strategies"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from .fundamental import FundamentalAnalyzer
from .ml_fundamental import MLFundamentalAnalyzer
from src.strategy import Strategy

logger = logging.getLogger('atwz.backtest_fundamental')

@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Dict[str, Any]]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]

class FundamentalBacktester:
    """Backtester for fundamental analysis strategies"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.fundamental_analyzer = FundamentalAnalyzer(symbols)
        self.ml_analyzer = MLFundamentalAnalyzer()
        
    def prepare_historical_data(self, symbol: str,
                              start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """Prepare historical fundamental and price data"""
        try:
            # Get price data
            import yfinance as yf
            stock = yf.Ticker(symbol)
            price_data = stock.history(start=start_date, end=end_date)
            
            # Get fundamental data at different points in time
            fundamental_data = []
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    metrics = self.fundamental_analyzer.fetch_company_metrics(symbol)
                    health = self.fundamental_analyzer.calculate_financial_health(symbol)
                    
                    fundamental_data.append({
                        'date': current_date,
                        'pe_ratio': metrics.pe_ratio,
                        'pb_ratio': metrics.pb_ratio,
                        'roe': metrics.roe,
                        'health_score': health.overall_health_score,
                        # Add more metrics as needed
                    })
                except Exception as e:
                    logger.warning(f"Error fetching fundamentals for {current_date}: {e}")
                
                current_date += timedelta(days=30)  # Monthly fundamental updates
            
            # Convert to DataFrame and merge with price data
            fundamental_df = pd.DataFrame(fundamental_data)
            fundamental_df.set_index('date', inplace=True)
            
            # Forward fill fundamental data
            fundamental_df = fundamental_df.reindex(price_data.index, method='ffill')
            
            # Merge price and fundamental data
            combined_data = pd.concat([price_data, fundamental_df], axis=1)
            combined_data.fillna(method='ffill', inplace=True)
            
            return combined_data
        
        except Exception as e:
            logger.error(f"Error preparing historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_signals(self, data: pd.DataFrame,
                         strategy_type: str,
                         parameters: Dict[str, Any]) -> pd.Series:
        """Calculate trading signals based on fundamental data"""
        signals = pd.Series(index=data.index, data=0)
        
        if strategy_type == 'value':
            # Value investing strategy
            signals = self._value_strategy_signals(data, parameters)
        elif strategy_type == 'quality':
            # Quality investing strategy
            signals = self._quality_strategy_signals(data, parameters)
        elif strategy_type == 'growth':
            # Growth investing strategy
            signals = self._growth_strategy_signals(data, parameters)
        elif strategy_type == 'ml_enhanced':
            # ML-enhanced strategy
            signals = self._ml_strategy_signals(data, parameters)
        
        return signals
    
    def _value_strategy_signals(self, data: pd.DataFrame,
                              parameters: Dict[str, Any]) -> pd.Series:
        """Generate signals for value investing strategy"""
        signals = pd.Series(index=data.index, data=0)
        
        pe_threshold = parameters.get('pe_threshold', 15)
        pb_threshold = parameters.get('pb_threshold', 1.5)
        
        # Buy signals
        buy_condition = (
            (data['pe_ratio'] < pe_threshold) &
            (data['pb_ratio'] < pb_threshold)
        )
        
        # Sell signals
        sell_condition = (
            (data['pe_ratio'] > pe_threshold * 1.5) |
            (data['pb_ratio'] > pb_threshold * 1.5)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def _quality_strategy_signals(self, data: pd.DataFrame,
                                parameters: Dict[str, Any]) -> pd.Series:
        """Generate signals for quality investing strategy"""
        signals = pd.Series(index=data.index, data=0)
        
        health_threshold = parameters.get('health_threshold', 70)
        roe_threshold = parameters.get('roe_threshold', 0.15)
        
        # Buy signals
        buy_condition = (
            (data['health_score'] > health_threshold) &
            (data['roe'] > roe_threshold)
        )
        
        # Sell signals
        sell_condition = (
            (data['health_score'] < health_threshold * 0.8) |
            (data['roe'] < roe_threshold * 0.8)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def _growth_strategy_signals(self, data: pd.DataFrame,
                               parameters: Dict[str, Any]) -> pd.Series:
        """Generate signals for growth investing strategy"""
        # Implementation similar to value and quality strategies
        pass
    
    def _ml_strategy_signals(self, data: pd.DataFrame,
                           parameters: Dict[str, Any]) -> pd.Series:
        """Generate signals using ML predictions"""
        signals = pd.Series(index=data.index, data=0)
        
        for date in data.index:
            try:
                metrics = {
                    'symbol': data.name,
                    'pe_ratio': data.loc[date, 'pe_ratio'],
                    'pb_ratio': data.loc[date, 'pb_ratio'],
                    'roe': data.loc[date, 'roe'],
                    'health_score': data.loc[date, 'health_score'],
                    # Add more metrics as needed
                }
                
                # Get ML predictions
                valuation = self.ml_analyzer.predict_valuation(metrics)
                risk = self.ml_analyzer.predict_risk(metrics)
                
                # Generate signals based on ML predictions
                if (valuation.prediction > data.loc[date, 'Close'] * 1.2 and
                    risk.prediction < parameters.get('risk_threshold', 0.3)):
                    signals[date] = 1
                elif (valuation.prediction < data.loc[date, 'Close'] * 0.8 or
                      risk.prediction > parameters.get('risk_threshold', 0.3)):
                    signals[date] = -1
                
            except Exception as e:
                logger.warning(f"Error generating ML signals for {date}: {e}")
                continue
        
        return signals
    
    def calculate_returns(self, data: pd.DataFrame,
                         signals: pd.Series,
                         parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate strategy returns and metrics"""
        position = 0
        trades = []
        portfolio_value = pd.Series(index=data.index, data=1.0)
        
        for date in data.index:
            signal = signals[date]
            
            if signal == 1 and position <= 0:  # Buy signal
                position = 1
                trades.append({
                    'date': date,
                    'type': 'buy',
                    'price': data.loc[date, 'Close'],
                    'reason': 'signal'
                })
            elif signal == -1 and position >= 0:  # Sell signal
                position = -1
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': data.loc[date, 'Close'],
                    'reason': 'signal'
                })
            
            # Calculate daily returns
            if position != 0:
                daily_return = (
                    data.loc[date, 'Close'] / data.shift(1).loc[date, 'Close'] - 1
                ) * position
                portfolio_value[date] = portfolio_value[date - 1] * (1 + daily_return)
            else:
                portfolio_value[date] = portfolio_value[date - 1]
        
        # Calculate performance metrics
        total_return = portfolio_value[-1] / portfolio_value[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(data)) - 1
        
        # Calculate drawdown
        drawdown = (portfolio_value - portfolio_value.cummax()) / portfolio_value.cummax()
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio
        daily_returns = portfolio_value.pct_change()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades
        }
    
    def backtest_strategy(self, symbol: str,
                         strategy_type: str,
                         start_date: datetime,
                         end_date: datetime,
                         parameters: Dict[str, Any]) -> BacktestResult:
        """Run backtest for a fundamental strategy"""
        try:
            # Prepare data
            data = self.prepare_historical_data(symbol, start_date, end_date)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate signals
            signals = self.calculate_signals(data, strategy_type, parameters)
            
            # Calculate returns and metrics
            results = self.calculate_returns(data, signals, parameters)
            
            return BacktestResult(
                strategy_name=strategy_type,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                total_return=results['total_return'],
                annualized_return=results['annualized_return'],
                max_drawdown=results['max_drawdown'],
                sharpe_ratio=results['sharpe_ratio'],
                trades=results['trades'],
                metrics={
                    'win_rate': len([t for t in results['trades'] if t['type'] == 'sell' and t['price'] > t['entry_price']]) / len(results['trades']),
                    'avg_hold_period': np.mean([t['exit_date'] - t['entry_date'] for t in results['trades']]).days if results['trades'] else 0,
                    'profit_factor': sum([t['profit'] for t in results['trades'] if t['profit'] > 0]) / abs(sum([t['profit'] for t in results['trades'] if t['profit'] < 0])) if results['trades'] else 0
                },
                parameters=parameters
            )
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
            return BacktestResult(
                strategy_name=strategy_type,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                total_return=0,
                annualized_return=0,
                max_drawdown=0,
                sharpe_ratio=0,
                trades=[],
                metrics={},
                parameters=parameters
            )
    
    def optimize_parameters(self, symbol: str,
                          strategy_type: str,
                          start_date: datetime,
                          end_date: datetime,
                          parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search"""
        best_params = None
        best_return = float('-inf')
        
        # Generate parameter combinations
        from itertools import product
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        for params in product(*param_values):
            parameters = dict(zip(param_names, params))
            
            # Run backtest with these parameters
            result = self.backtest_strategy(
                symbol, strategy_type,
                start_date, end_date,
                parameters
            )
            
            # Update best parameters if needed
            if result.sharpe_ratio > best_return:
                best_return = result.sharpe_ratio
                best_params = parameters
        
        return best_params