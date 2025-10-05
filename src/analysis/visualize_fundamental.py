"""Visualization module for fundamental analysis"""

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from .fundamental import FundamentalAnalyzer, CompanyMetrics, ValuationModel
from .ml_fundamental import MLPrediction

logger = logging.getLogger('atwz.visualization')

class FundamentalVisualizer:
    """Create interactive visualizations for fundamental analysis"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'positive': '#2ca02c',
            'negative': '#d62728',
            'neutral': '#7f7f7f'
        }
    
    def create_valuation_dashboard(self,
                                 metrics: CompanyMetrics,
                                 valuation: ValuationModel,
                                 ml_prediction: MLPrediction) -> go.Figure:
        """Create comprehensive valuation dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Valuation Metrics',
                'Growth & Profitability',
                'Financial Health',
                'ML Insights'
            )
        )
        
        # Valuation metrics chart
        fig.add_trace(
            go.Bar(
                x=['P/E', 'P/B', 'Fair Value Ratio'],
                y=[
                    metrics.pe_ratio,
                    metrics.pb_ratio,
                    metrics.market_cap / valuation.fair_value
                ],
                name='Valuation Ratios',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=1
        )
        
        # Growth metrics
        fig.add_trace(
            go.Bar(
                x=['EPS Growth', 'Revenue Growth', 'ML Growth Pred'],
                y=[
                    metrics.eps_growth * 100,
                    metrics.revenue_growth * 100,
                    ml_prediction.prediction * 100
                ],
                name='Growth Metrics (%)',
                marker_color=self.color_scheme['secondary']
            ),
            row=1, col=2
        )
        
        # Financial health gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.health_score,
                title={'text': "Financial Health"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=2, col=1
        )
        
        # ML insights
        fig.add_trace(
            go.Scatter(
                x=list(ml_prediction.features_importance.keys()),
                y=list(ml_prediction.features_importance.values()),
                mode='markers+lines',
                name='Feature Importance',
                marker=dict(
                    size=10,
                    color=self.color_scheme['primary']
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        return fig
    
    def create_peer_comparison(self,
                             metrics: CompanyMetrics,
                             peer_metrics: Dict[str, CompanyMetrics]) -> go.Figure:
        """Create peer comparison visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'P/E Ratio Comparison',
                'ROE Comparison',
                'Growth Comparison',
                'Market Cap Comparison'
            )
        )
        
        # Prepare peer data
        peer_names = list(peer_metrics.keys())
        pe_ratios = [m.pe_ratio for m in peer_metrics.values()]
        roes = [m.roe for m in peer_metrics.values()]
        growth_rates = [m.revenue_growth for m in peer_metrics.values()]
        market_caps = [m.market_cap for m in peer_metrics.values()]
        
        # Add company's metrics
        peer_names.append(metrics.symbol)
        pe_ratios.append(metrics.pe_ratio)
        roes.append(metrics.roe)
        growth_rates.append(metrics.revenue_growth)
        market_caps.append(metrics.market_cap)
        
        # P/E comparison
        fig.add_trace(
            go.Bar(
                x=peer_names,
                y=pe_ratios,
                name='P/E Ratio',
                marker_color=[
                    self.color_scheme['primary'] if name != metrics.symbol
                    else self.color_scheme['secondary']
                    for name in peer_names
                ]
            ),
            row=1, col=1
        )
        
        # ROE comparison
        fig.add_trace(
            go.Bar(
                x=peer_names,
                y=[r * 100 for r in roes],
                name='ROE (%)',
                marker_color=[
                    self.color_scheme['primary'] if name != metrics.symbol
                    else self.color_scheme['secondary']
                    for name in peer_names
                ]
            ),
            row=1, col=2
        )
        
        # Growth comparison
        fig.add_trace(
            go.Bar(
                x=peer_names,
                y=[g * 100 for g in growth_rates],
                name='Revenue Growth (%)',
                marker_color=[
                    self.color_scheme['primary'] if name != metrics.symbol
                    else self.color_scheme['secondary']
                    for name in peer_names
                ]
            ),
            row=2, col=1
        )
        
        # Market cap comparison
        fig.add_trace(
            go.Bar(
                x=peer_names,
                y=market_caps,
                name='Market Cap',
                marker_color=[
                    self.color_scheme['primary'] if name != metrics.symbol
                    else self.color_scheme['secondary']
                    for name in peer_names
                ]
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        return fig
    
    def create_historical_analysis(self,
                                 historical_data: pd.DataFrame,
                                 signals: Optional[pd.Series] = None) -> go.Figure:
        """Create historical fundamental analysis visualization"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Price and Signals',
                'Fundamental Metrics',
                'Trading Volume'
            ),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Price chart with signals
        fig.add_trace(
            go.Candlestick(
                x=historical_data.index,
                open=historical_data['Open'],
                high=historical_data['High'],
                low=historical_data['Low'],
                close=historical_data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        if signals is not None:
            # Add buy signals
            buy_signals = signals[signals == 1]
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=historical_data.loc[buy_signals.index, 'Low'] * 0.99,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color=self.color_scheme['positive']
                    ),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
            
            # Add sell signals
            sell_signals = signals[signals == -1]
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=historical_data.loc[sell_signals.index, 'High'] * 1.01,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color=self.color_scheme['negative']
                    ),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
        
        # Fundamental metrics
        for metric in ['pe_ratio', 'pb_ratio', 'roe']:
            if metric in historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=historical_data[metric],
                        name=metric.upper(),
                        mode='lines'
                    ),
                    row=2, col=1
                )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=historical_data.index,
                y=historical_data['Volume'],
                name='Volume',
                marker_color=self.color_scheme['neutral']
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=1000,
            xaxis3_title='Date',
            yaxis_title='Price',
            yaxis2_title='Value',
            yaxis3_title='Volume'
        )
        
        return fig
    
    def create_backtest_results(self, results: Dict[str, Any]) -> go.Figure:
        """Create backtest results visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Portfolio Value',
                'Drawdown',
                'Monthly Returns',
                'Trade Distribution'
            )
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=results['portfolio_value'].index,
                y=results['portfolio_value'].values,
                name='Portfolio Value',
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=results['drawdown'].index,
                y=results['drawdown'].values * 100,
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color=self.color_scheme['negative'])
            ),
            row=1, col=2
        )
        
        # Monthly returns
        monthly_returns = results['portfolio_value'].resample('M').last().pct_change()
        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.values * 100,
                name='Monthly Returns %',
                marker_color=[
                    self.color_scheme['positive'] if x > 0
                    else self.color_scheme['negative']
                    for x in monthly_returns.values
                ]
            ),
            row=2, col=1
        )
        
        # Trade distribution
        trade_returns = pd.Series([t['return'] for t in results['trades']])
        fig.add_trace(
            go.Histogram(
                x=trade_returns * 100,
                name='Trade Returns Distribution',
                marker_color=self.color_scheme['primary']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            xaxis4_title='Return %',
            yaxis_title='Value',
            yaxis2_title='Drawdown %',
            yaxis3_title='Return %',
            yaxis4_title='Frequency'
        )
        
        return fig
    
    def save_dashboard(self, fig: go.Figure, filename: str):
        """Save dashboard to HTML file"""
        try:
            fig.write_html(filename)
            logger.info(f"Dashboard saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")
    
    def display_dashboard(self, fig: go.Figure):
        """Display dashboard in notebook or browser"""
        try:
            fig.show()
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")
            
    def create_summary_report(self,
                            metrics: CompanyMetrics,
                            valuation: ValuationModel,
                            ml_prediction: MLPrediction) -> str:
        """Create text-based summary report"""
        report = f"""
        Fundamental Analysis Summary Report
        =================================
        Symbol: {metrics.symbol}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Valuation Metrics
        ----------------
        Market Cap: ${metrics.market_cap:,.2f}
        P/E Ratio: {metrics.pe_ratio:.2f}
        P/B Ratio: {metrics.pb_ratio:.2f}
        Fair Value: ${valuation.fair_value:,.2f}
        Valuation Confidence: {valuation.confidence:.1%}
        
        Growth & Profitability
        --------------------
        EPS Growth: {metrics.eps_growth:.1%}
        Revenue Growth: {metrics.revenue_growth:.1%}
        ROE: {metrics.roe:.1%}
        ROA: {metrics.roa:.1%}
        
        Financial Health
        --------------
        Health Score: {metrics.health_score}/100
        Debt/Equity: {metrics.debt_to_equity:.2f}
        Current Ratio: {metrics.current_ratio:.2f}
        
        ML Insights
        ----------
        Predicted Growth: {ml_prediction.prediction:.1%}
        ML Confidence: {ml_prediction.confidence:.1%}
        Top Features: {', '.join(f"{k}: {v:.3f}" for k, v in list(ml_prediction.features_importance.items())[:3])}
        
        Investment Recommendation
        ----------------------
        {self._generate_recommendation(metrics, valuation, ml_prediction)}
        """
        
        return report
    
    def _generate_recommendation(self,
                               metrics: CompanyMetrics,
                               valuation: ValuationModel,
                               ml_prediction: MLPrediction) -> str:
        """Generate investment recommendation based on analysis"""
        current_price = metrics.market_cap / metrics.shares_outstanding
        upside = (valuation.fair_value / current_price - 1)
        
        if upside > 0.2 and metrics.health_score > 70 and ml_prediction.confidence > 0.7:
            return f"Strong Buy - Potential upside of {upside:.1%} with high confidence"
        elif upside > 0.1 and metrics.health_score > 60:
            return f"Buy - Moderate upside of {upside:.1%} with good fundamentals"
        elif upside < -0.1 or metrics.health_score < 50:
            return "Sell - Overvalued or poor fundamentals"
        else:
            return "Hold - Fair valuation and average fundamentals"