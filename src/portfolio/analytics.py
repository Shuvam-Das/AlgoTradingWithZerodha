"""Advanced portfolio analytics and risk management.

This module provides:
1. Portfolio performance metrics
2. Risk analytics (VaR, volatility, etc.)
3. Allocation analysis
4. Correlation analysis
5. Performance attribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger('atwz.portfolio')

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    timestamp: datetime
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @property
    def return_pct(self) -> float:
        return (self.current_price / self.entry_price - 1) * 100

@dataclass
class Trade:
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    fees: float = 0.0
    
    @property
    def value(self) -> float:
        return self.quantity * self.price

@dataclass
class RiskMetrics:
    volatility: float
    var_95: float
    var_99: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    correlation: float

class PortfolioAnalytics:
    """Advanced portfolio analytics engine"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash: float = initial_capital
        self._price_history: Dict[str, pd.DataFrame] = {}
    
    def add_position(self, position: Position):
        """Add or update a position"""
        self.positions[position.symbol] = position
    
    def add_trade(self, trade: Trade):
        """Record a new trade"""
        self.trades.append(trade)
        # Update cash
        trade_value = trade.value + trade.fees
        if trade.side.lower() == 'buy':
            self.cash -= trade_value
        else:
            self.cash += trade_value
    
    def add_price_history(self, symbol: str, data: pd.DataFrame):
        """Add historical price data for a symbol"""
        self._price_history[symbol] = data
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return positions_value + self.cash
    
    def get_portfolio_return(self) -> float:
        """Calculate total return percentage"""
        current_value = self.get_portfolio_value()
        return (current_value / self.initial_capital - 1) * 100
    
    def calculate_risk_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        # Get portfolio returns series
        portfolio_values = self._get_portfolio_value_series()
        returns = portfolio_values.pct_change().dropna()
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Sharpe Ratio (assuming risk-free rate = 2%)
        rf_rate = 0.02
        excess_returns = returns - rf_rate/252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
        # Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = cum_returns / running_max - 1
        max_drawdown = drawdowns.min()
        
        # Beta and Correlation (if benchmark provided)
        beta = correlation = 0.0
        if benchmark_returns is not None:
            # Align dates
            common_dates = returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                aligned_returns = returns[common_dates]
                aligned_benchmark = benchmark_returns[common_dates]
                
                # Calculate beta
                covariance = aligned_returns.cov(aligned_benchmark)
                benchmark_variance = aligned_benchmark.var()
                beta = covariance / benchmark_variance
                
                # Calculate correlation
                correlation = aligned_returns.corr(aligned_benchmark)
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            beta=beta,
            correlation=correlation
        )
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """Calculate current asset allocation percentages"""
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return {}
        
        allocation = {
            'cash': (self.cash / total_value) * 100
        }
        
        for symbol, pos in self.positions.items():
            allocation[symbol] = (pos.market_value / total_value) * 100
            
        return allocation
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Calculate sector-wise allocation"""
        try:
            import yfinance as yf
            
            total_value = self.get_portfolio_value()
            if total_value == 0:
                return {}
            
            sector_values: Dict[str, float] = {}
            
            for symbol, pos in self.positions.items():
                try:
                    stock = yf.Ticker(symbol)
                    sector = stock.info.get('sector', 'Unknown')
                    if sector not in sector_values:
                        sector_values[sector] = 0
                    sector_values[sector] += pos.market_value
                except Exception:
                    logger.warning(f"Could not fetch sector for {symbol}")
                    
            # Convert to percentages
            return {sector: (value/total_value)*100 
                   for sector, value in sector_values.items()}
        except ImportError:
            logger.warning("yfinance not available for sector allocation")
            return {}
    
    def get_performance_attribution(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance attribution by position"""
        total_return = self.get_portfolio_return()
        if total_return == 0:
            return {}
        
        attribution = {}
        for symbol, pos in self.positions.items():
            position_return = pos.return_pct
            weight = pos.market_value / self.get_portfolio_value()
            contribution = position_return * weight
            
            attribution[symbol] = {
                'weight': weight * 100,
                'return': position_return,
                'contribution': contribution,
                'contribution_pct': (contribution / total_return) * 100 if total_return != 0 else 0
            }
            
        return attribution
    
    def get_trade_analytics(self) -> Dict[str, Any]:
        """Analyze trading performance"""
        if not self.trades:
            return {}
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'side': t.side,
            'quantity': t.quantity,
            'price': t.price,
            'value': t.value,
            'timestamp': t.timestamp
        } for t in self.trades])
        
        # Basic statistics
        stats = {
            'total_trades': len(trades_df),
            'buy_trades': len(trades_df[trades_df['side'] == 'buy']),
            'sell_trades': len(trades_df[trades_df['side'] == 'sell']),
            'total_volume': trades_df['value'].sum(),
            'avg_trade_size': trades_df['value'].mean(),
            'max_trade_size': trades_df['value'].max(),
        }
        
        # Trading activity by symbol
        by_symbol = trades_df.groupby('symbol').agg({
            'value': ['count', 'sum', 'mean'],
            'quantity': 'sum'
        }).round(2)
        
        # Add to stats
        stats['by_symbol'] = by_symbol.to_dict()
        
        return stats
    
    def _get_portfolio_value_series(self) -> pd.Series:
        """Generate historical portfolio value series"""
        if not self._price_history:
            raise ValueError("No price history available")
        
        # Get common date range
        all_dates = set()
        for df in self._price_history.values():
            all_dates.update(df.index)
        dates = sorted(list(all_dates))
        
        # Calculate portfolio value for each date
        values = []
        for date in dates:
            total = self.initial_capital  # Start with initial cash
            for symbol, pos in self.positions.items():
                if symbol in self._price_history:
                    df = self._price_history[symbol]
                    if date in df.index:
                        price = df.loc[date, 'close']
                        total += pos.quantity * price
            values.append(total)
            
        return pd.Series(values, index=dates)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive portfolio report"""
        report = {
            'timestamp': datetime.now(),
            'portfolio_value': self.get_portfolio_value(),
            'total_return': self.get_portfolio_return(),
            'cash': self.cash,
            'positions': len(self.positions),
            'risk_metrics': vars(self.calculate_risk_metrics()),
            'asset_allocation': self.get_asset_allocation(),
            'sector_allocation': self.get_sector_allocation(),
            'attribution': self.get_performance_attribution(),
            'trade_analytics': self.get_trade_analytics()
        }
        
        # Add position details
        report['position_details'] = [{
            'symbol': symbol,
            'quantity': pos.quantity,
            'entry_price': pos.entry_price,
            'current_price': pos.current_price,
            'market_value': pos.market_value,
            'unrealized_pnl': pos.unrealized_pnl,
            'return_pct': pos.return_pct
        } for symbol, pos in self.positions.items()]
        
        return report