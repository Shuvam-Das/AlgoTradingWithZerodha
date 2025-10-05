"""Helper module for calculating real-time risk metrics"""
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PortfolioRisk:
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    expected_shortfall: float  # Expected Shortfall/CVaR
    sharpe_ratio: float  # Sharpe Ratio
    volatility: float  # Portfolio Volatility
    max_drawdown: float  # Maximum Drawdown
    beta: float  # Portfolio Beta
    correlation_matrix: pd.DataFrame  # Asset Correlations


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk using historical simulation"""
    return abs(np.percentile(returns, (1 - confidence) * 100))


def calculate_expected_shortfall(returns: pd.Series, var_95: float) -> float:
    """Calculate Expected Shortfall (CVaR)"""
    return abs(returns[returns <= -var_95].mean())


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Calculate annualized Sharpe Ratio"""
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate Maximum Drawdown"""
    rolling_max = prices.expanding().max()
    drawdowns = prices/rolling_max - 1
    return abs(drawdowns.min())


def calculate_portfolio_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate Portfolio Beta against market benchmark"""
    covariance = portfolio_returns.cov(market_returns)
    market_variance = market_returns.var()
    return covariance / market_variance if market_variance != 0 else 1.0


def calculate_risk_metrics(portfolio_data: Dict[str, pd.DataFrame], 
                         market_data: pd.DataFrame) -> PortfolioRisk:
    """Calculate comprehensive risk metrics for the portfolio"""
    # Combine all position returns into portfolio returns
    portfolio_returns = pd.Series(0.0, index=market_data.index)
    weights = {}
    
    total_value = sum(data['value'].iloc[-1] for data in portfolio_data.values())
    
    for symbol, data in portfolio_data.items():
        weight = data['value'].iloc[-1] / total_value if total_value > 0 else 0
        weights[symbol] = weight
        portfolio_returns += data['returns'] * weight
    
    # Calculate risk metrics
    var_95 = calculate_var(portfolio_returns, 0.95)
    var_99 = calculate_var(portfolio_returns, 0.99)
    es = calculate_expected_shortfall(portfolio_returns, var_95)
    sharpe = calculate_sharpe_ratio(portfolio_returns)
    vol = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
    mdd = calculate_max_drawdown(portfolio_returns.cumsum())
    beta = calculate_portfolio_beta(portfolio_returns, market_data['returns'])
    
    # Calculate correlation matrix
    returns_df = pd.DataFrame({symbol: data['returns'] 
                             for symbol, data in portfolio_data.items()})
    corr_matrix = returns_df.corr()
    
    return PortfolioRisk(
        var_95=var_95,
        var_99=var_99,
        expected_shortfall=es,
        sharpe_ratio=sharpe,
        volatility=vol,
        max_drawdown=mdd,
        beta=beta,
        correlation_matrix=corr_matrix
    )


def format_risk_metrics(risk: PortfolioRisk) -> Dict[str, str]:
    """Format risk metrics for display"""
    return {
        'Value at Risk (95%)': f'{risk.var_95:.2%}',
        'Value at Risk (99%)': f'{risk.var_99:.2%}',
        'Expected Shortfall': f'{risk.expected_shortfall:.2%}',
        'Sharpe Ratio': f'{risk.sharpe_ratio:.2f}',
        'Volatility (Ann.)': f'{risk.volatility:.2%}',
        'Maximum Drawdown': f'{risk.max_drawdown:.2%}',
        'Portfolio Beta': f'{risk.beta:.2f}'
    }