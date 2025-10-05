"""Real-time dashboard for visualization and monitoring.

This module provides:
1. TradingView-style charts with ML insights
2. Advanced portfolio analytics and risk metrics
3. Real-time performance and execution metrics
4. System health monitoring with predictive alerts
5. ML model performance visualization
"""

import dash
from dash import html, dcc, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import queue
import logging
import asyncio

from .risk_metrics import calculate_risk_metrics, format_risk_metrics, PortfolioRisk
from .ml_insights import MLDashboardIntegrator
from .metrics import calculate_market_metrics, calculate_execution_quality
from .chart_manager import ChartManager

logger = logging.getLogger('atwz.dashboard')

class DashboardServer:
    """Real-time trading dashboard"""
    
    def __init__(self, update_interval: int = 1000):
        self.app = dash.Dash(__name__)
        self.update_interval = update_interval
        self.data_queue = queue.Queue()
        self.last_update = datetime.now()
        
        # ML integration
        self.ml_integrator = MLDashboardIntegrator()
        
        # State storage
        self.current_prices: Dict[str, float] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.market_data: pd.DataFrame = pd.DataFrame()
        self.risk_metrics: Optional[PortfolioRisk] = None
        self.execution_metrics: Dict[str, float] = {}
        self.ml_insights: List[Dict[str, Any]] = []
        
        # System monitoring
        self.system_health: Dict[str, Any] = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'api_latency': 0,
            'error_count': 0,
            'ml_model_health': {},
            'database_latency': 0,
            'websocket_status': 'connected'
        }
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Create the dashboard layout with advanced features"""
        self._add_helper_methods()
        self.app.layout = html.Div([
            # Header with enhanced status
            html.Div([
                html.H1('AlgoTradingWithZerodha Dashboard'),
                html.Div([
                    html.Span('Last Update: ', style={'fontWeight': 'bold'}),
                    html.Span(id='last-update'),
                    html.Span(' | System Status: ', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                    html.Span(id='system-status'),
                ], style={'marginTop': '10px'})
            ], style={'padding': '10px', 'backgroundColor': '#f8f9fa'}),
            
            # Main content
            html.Div([
                # Left column - Advanced Charts
                html.Div([
                    html.H3('Market Overview with ML Insights'),
                    dcc.Graph(id='advanced-chart'),
                    html.Div([
                        dcc.Graph(id='volume-analysis', style={'display': 'inline-block', 'width': '50%'}),
                        dcc.Graph(id='ml-confidence', style={'display': 'inline-block', 'width': '50%'})
                    ]),
                    dcc.Interval(
                        id='interval-component',
                        interval=self.update_interval,
                        n_intervals=0
                    )
                ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'}),
                
                # Right column - Enhanced Analytics
                html.Div([
                    # Portfolio Risk Metrics
                    html.H3('Risk Analytics'),
                    dash_table.DataTable(
                        id='risk-metrics-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px'}
                    ),
                    
                    # ML Insights
                    html.H3('ML Trading Signals'),
                    html.Div(id='ml-insights'),
                    
                    # Position Management
                    html.H3('Position Management'),
                    dash_table.DataTable(
                        id='positions-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px'}
                    ),
                    
                    # Execution Analytics
                    html.H3('Execution Quality'),
                    html.Div(id='execution-metrics'),
                    
                    # System Health
                    html.H3('System Health'),
                    html.Div(id='system-health'),
                    
                    # Alerts Panel
                    html.H3('Critical Alerts'),
                    html.Div(id='alerts-panel', style={'maxHeight': '200px', 'overflow': 'auto'})
                ], style={'width': '29%', 'display': 'inline-block', 'vertical-align': 'top'})
            ], style={'padding': '20px'})
        ])
    
    def _add_helper_methods(self):
        """Add helper methods for data formatting and display"""
        
        def _update_state(self, data: Dict[str, Any]):
            """Update internal state with new data"""
            if 'prices' in data:
                self.current_prices.update(data['prices'])
            if 'positions' in data:
                self.positions.update(data['positions'])
            if 'alerts' in data:
                self.alerts.extend(data['alerts'])
            if 'market_data' in data:
                self.market_data = data['market_data']
            if 'system' in data:
                self.system_health.update(data['system'])
        
        def _get_risk_levels(self) -> Dict[str, float]:
            """Get current risk levels for visualization"""
            if not self.risk_metrics:
                return {}
            return {
                'Stop Loss': self.risk_metrics.var_95,
                'Value at Risk (99%)': self.risk_metrics.var_99
            }
        
        def _format_risk_metrics(self) -> List[Dict[str, str]]:
            """Format risk metrics for table display"""
            if not self.risk_metrics:
                return []
            
            metrics = format_risk_metrics(self.risk_metrics)
            return [{'metric': k, 'value': v} for k, v in metrics.items()]
        
        def _format_positions(self) -> List[Dict[str, Any]]:
            """Format positions for table display"""
            return [
                {
                    'symbol': symbol,
                    'quantity': pos['qty'],
                    'entry_price': f"{pos['entry_price']:.2f}",
                    'current_price': f"{self.current_prices.get(symbol, 0):.2f}",
                    'pnl': f"{pos.get('pnl', 0):.2f}%"
                }
                for symbol, pos in self.positions.items()
            ]
        
        def _format_execution_metrics(self) -> html.Div:
            """Format execution metrics for display"""
            return html.Div([
                html.P(f"Average Slippage: {self.execution_metrics['slippage']:.4f}%"),
                html.P(f"Fill Time: {self.execution_metrics['fill_time']:.2f}s"),
                html.P(f"Rejection Rate: {self.execution_metrics['rejection_rate']:.2%}"),
                html.P(f"Partial Fills: {self.execution_metrics['partial_fills']:.2%}")
            ])
        
        def _format_system_health(self) -> html.Div:
            """Format system health for display"""
            style = {'padding': '10px', 'margin': '5px', 'border-radius': '5px'}
            return html.Div([
                html.Div([
                    html.Strong('CPU: '),
                    html.Span(f"{self.system_health['cpu_usage']}%")
                ], style={**style, 'background-color': self._get_health_color(self.system_health['cpu_usage'])}),
                html.Div([
                    html.Strong('Memory: '),
                    html.Span(f"{self.system_health['memory_usage']}%")
                ], style={**style, 'background-color': self._get_health_color(self.system_health['memory_usage'])}),
                html.Div([
                    html.Strong('API Latency: '),
                    html.Span(f"{self.system_health['api_latency']}ms")
                ], style={**style, 'background-color': self._get_health_color(self.system_health['api_latency'], is_latency=True)})
            ])
        
        def _format_alerts(self) -> html.Div:
            """Format alerts for display"""
            return html.Div([
                html.Div([
                    html.Strong(alert['type']),
                    html.Span(f": {alert['message']}"),
                    html.Span(f" ({alert['timestamp'].strftime('%H:%M:%S')})")
                ], style={
                    'padding': '5px',
                    'margin': '2px',
                    'background-color': self._get_alert_color(alert['severity'])
                })
                for alert in reversed(self.alerts[-5:])
            ])
        
        def _create_ml_insights_html(self) -> html.Div:
            """Create HTML for ML insights display"""
            return html.Div([
                html.Div([
                    html.Strong(f"{insight['symbol']}: "),
                    html.Span(
                        f"{insight['direction'].upper()} ({insight['confidence']:.1%})",
                        style={'color': 'green' if insight['direction'] == 'buy' else 'red'}
                    )
                ], style={'padding': '5px'})
                for insight in self.ml_insights
            ])
        
        def _get_system_status(self) -> html.Span:
            """Get overall system status indicator"""
            status = 'Healthy'
            color = 'green'
            
            if self.system_health['error_count'] > 0:
                status = 'Warning'
                color = 'orange'
            if self.system_health['cpu_usage'] > 90 or self.system_health['memory_usage'] > 90:
                status = 'Critical'
                color = 'red'
            
            return html.Span(status, style={'color': color, 'fontWeight': 'bold'})
        
        def _get_health_color(self, value: float, is_latency: bool = False) -> str:
            """Get color for health metrics"""
            if is_latency:
                if value < 100:
                    return 'rgba(0, 255, 0, 0.2)'
                elif value < 300:
                    return 'rgba(255, 165, 0, 0.2)'
                return 'rgba(255, 0, 0, 0.2)'
            
            if value < 70:
                return 'rgba(0, 255, 0, 0.2)'
            elif value < 90:
                return 'rgba(255, 165, 0, 0.2)'
            return 'rgba(255, 0, 0, 0.2)'
        
        def _get_alert_color(self, severity: str) -> str:
            """Get color for alert severity"""
            colors = {
                'low': 'rgba(0, 255, 0, 0.2)',
                'medium': 'rgba(255, 165, 0, 0.2)',
                'high': 'rgba(255, 0, 0, 0.2)'
            }
            return colors.get(severity.lower(), 'rgba(128, 128, 128, 0.2)')
        
        # Add methods to instance
        for method in [
            _update_state, _get_risk_levels, _format_risk_metrics,
            _format_positions, _format_execution_metrics, _format_system_health,
            _format_alerts, _create_ml_insights_html, _get_system_status,
            _get_health_color, _get_alert_color
        ]:
            setattr(self.__class__, method.__name__, method)
    
    def _setup_callbacks(self):
        """Set up all dashboard callbacks with enhanced features"""
        @self.app.callback(
            [
                dash.Output('advanced-chart', 'figure'),
                dash.Output('volume-analysis', 'figure'),
                dash.Output('ml-confidence', 'figure'),
                dash.Output('risk-metrics-table', 'data'),
                dash.Output('ml-insights', 'children'),
                dash.Output('positions-table', 'data'),
                dash.Output('execution-metrics', 'children'),
                dash.Output('system-health', 'children'),
                dash.Output('alerts-panel', 'children'),
                dash.Output('system-status', 'children'),
                dash.Output('last-update', 'children')
            ],
            [dash.Input('interval-component', 'n_intervals')]
        )
        async def update_dashboard(n):
            # Process any new data from queue
            try:
                while not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    self._update_state(data)
            except queue.Empty:
                pass
            
            # Get ML predictions and insights
            ml_data = await self.ml_integrator.update_predictions(self.market_data)
            self.ml_insights = self.ml_integrator.get_trading_insights()
            
            # Calculate risk metrics
            self.risk_metrics = calculate_risk_metrics(
                self.positions,
                self.market_data
            )
            
            # Update execution metrics
            self.execution_metrics = calculate_execution_quality(
                list(self.positions.values())
            )
            
            # Create chart figures
            advanced_chart = ChartManager.create_advanced_chart(
                self.market_data,
                ml_data['predictions'],
                self._get_risk_levels()
            )
            
            volume_chart = ChartManager.create_volume_profile(
                self.market_data
            )
            
            ml_confidence = ChartManager.create_ml_confidence_chart(
                ml_data['confidence']
            )
            
            # Format data for tables
            risk_table = self._format_risk_metrics()
            positions_table = self._format_positions()
            execution_metrics = self._format_execution_metrics()
            system_health = self._format_system_health()
            alerts = self._format_alerts()
            
            # Update status indicators
            system_status = self._get_system_status()
            last_update = f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return (
                advanced_chart,
                volume_chart,
                ml_confidence,
                risk_table,
                self._create_ml_insights_html(),
                positions_table,
                execution_metrics,
                system_health,
                alerts,
                system_status,
                last_update
            )
                    update = self.data_queue.get_nowait()
                    self._process_update(update)
                except queue.Empty:
                    break
            
            # Generate all components
            price_chart = self._create_price_chart()
            volume_chart = self._create_volume_chart()
            portfolio = self._create_portfolio_summary()
            positions = self._create_positions_table()
            alerts = self._create_alerts_panel()
            health = self._create_system_health()
            
            last_update = f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            
            return price_chart, volume_chart, portfolio, positions, alerts, health, last_update
    
    def _create_price_chart(self) -> go.Figure:
        """Create main price chart with indicators"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Add price candlesticks
        fig.add_trace(go.Candlestick(
            x=self.price_data.index,
            open=self.price_data['open'],
            high=self.price_data['high'],
            low=self.price_data['low'],
            close=self.price_data['close'],
            name='OHLC'
        ), row=1, col=1)
        
        # Add moving averages
        for window in [20, 50, 200]:
            ma = self.price_data['close'].rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=self.price_data.index,
                y=ma,
                name=f'MA{window}',
                line=dict(width=1)
            ), row=1, col=1)
        
        # Add MACD
        macd = self.price_data['close'].ewm(span=12).mean() - \
               self.price_data['close'].ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        
        fig.add_trace(go.Scatter(
            x=self.price_data.index,
            y=macd,
            name='MACD',
            line=dict(color='blue')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.price_data.index,
            y=signal,
            name='Signal',
            line=dict(color='orange')
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='Price Chart with Indicators',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_volume_chart(self) -> go.Figure:
        """Create volume chart with analysis"""
        fig = go.Figure()
        
        # Add volume bars
        colors = ['red' if row['close'] < row['open'] else 'green' 
                 for _, row in self.price_data.iterrows()]
        
        fig.add_trace(go.Bar(
            x=self.price_data.index,
            y=self.price_data['volume'],
            name='Volume',
            marker_color=colors
        ))
        
        # Add volume MA
        vol_ma = self.price_data['volume'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=self.price_data.index,
            y=vol_ma,
            name='Volume MA20',
            line=dict(color='blue', width=1)
        ))
        
        # Update layout
        fig.update_layout(
            title='Volume Analysis',
            xaxis_title='Date',
            yaxis_title='Volume',
            height=300
        )
        
        return fig
    
    def _create_portfolio_summary(self) -> html.Div:
        """Create portfolio summary panel"""
        total_value = sum(pos['value'] for pos in self.positions.values())
        daily_pnl = sum(pos.get('daily_pnl', 0) for pos in self.positions.values())
        total_pnl = sum(pos.get('total_pnl', 0) for pos in self.positions.values())
        
        return html.Div([
            html.P(f'Total Portfolio Value: ${total_value:,.2f}'),
            html.P(f'Daily P&L: ${daily_pnl:,.2f}',
                  style={'color': 'green' if daily_pnl >= 0 else 'red'}),
            html.P(f'Total P&L: ${total_pnl:,.2f}',
                  style={'color': 'green' if total_pnl >= 0 else 'red'})
        ])
    
    def _create_positions_table(self) -> html.Table:
        """Create positions table"""
        headers = ['Symbol', 'Qty', 'Avg Price', 'Current', 'P&L']
        
        rows = []
        for symbol, pos in self.positions.items():
            rows.append(html.Tr([
                html.Td(symbol),
                html.Td(pos['quantity']),
                html.Td(f"${pos['avg_price']:.2f}"),
                html.Td(f"${pos['current_price']:.2f}"),
                html.Td(f"${pos['total_pnl']:.2f}",
                       style={'color': 'green' if pos['total_pnl'] >= 0 else 'red'})
            ]))
        
        return html.Table(
            [html.Tr([html.Th(col) for col in headers])] + rows,
            style={'width': '100%', 'textAlign': 'left'}
        )
    
    def _create_alerts_panel(self) -> html.Div:
        """Create alerts panel"""
        alerts = []
        for alert in self.alerts[-5:]:  # Show last 5 alerts
            alerts.append(html.Div(
                f"{alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}",
                style={'color': alert.get('color', 'black'),
                       'padding': '5px',
                       'border-bottom': '1px solid #eee'}
            ))
        return html.Div(alerts)
    
    def _create_system_health(self) -> html.Div:
        """Create system health panel"""
        return html.Div([
            html.P(f"CPU Usage: {self.system_health['cpu_usage']}%"),
            html.P(f"Memory Usage: {self.system_health['memory_usage']}%"),
            html.P(f"API Latency: {self.system_health['api_latency']}ms"),
            html.P(f"Error Count: {self.system_health['error_count']}")
        ])
    
    def _process_update(self, update: Dict[str, Any]):
        """Process incoming data updates"""
        update_type = update.get('type')
        
        if update_type == 'price':
            self.current_prices[update['symbol']] = update['price']
            # Update price history
            if hasattr(self, 'price_data'):
                self.price_data.loc[update['timestamp']] = {
                    'open': update['open'],
                    'high': update['high'],
                    'low': update['low'],
                    'close': update['close'],
                    'volume': update['volume']
                }
        
        elif update_type == 'position':
            self.positions[update['symbol']] = update['position']
        
        elif update_type == 'alert':
            self.alerts.append({
                'timestamp': datetime.now(),
                'message': update['message'],
                'color': update.get('color', 'black')
            })
        
        elif update_type == 'health':
            self.system_health.update(update['data'])
    
    def add_data_update(self, update: Dict[str, Any]):
        """Add update to the queue for processing"""
        try:
            self.data_queue.put_nowait(update)
        except queue.Full:
            logger.warning("Update queue is full, dropping update")
    
    def run(self, host: str = 'localhost', port: int = 8050, debug: bool = False):
        """Run the dashboard server"""
        self.app.run_server(host=host, port=port, debug=debug)

# Helper class for external data updates
class DashboardUpdater:
    """Helper to send updates to dashboard"""
    
    def __init__(self, host: str = 'localhost', port: int = 8050):
        self.base_url = f"http://{host}:{port}"
    
    def update_price(self, symbol: str, price_data: Dict[str, Any]):
        """Send price update"""
        self._send_update({
            'type': 'price',
            'symbol': symbol,
            'timestamp': datetime.now(),
            **price_data
        })
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Send position update"""
        self._send_update({
            'type': 'position',
            'symbol': symbol,
            'position': position_data
        })
    
    def add_alert(self, message: str, color: str = 'black'):
        """Send new alert"""
        self._send_update({
            'type': 'alert',
            'message': message,
            'color': color
        })
    
    def update_health(self, health_data: Dict[str, Any]):
        """Send health metrics update"""
        self._send_update({
            'type': 'health',
            'data': health_data
        })
    
    def _send_update(self, data: Dict[str, Any]):
        """Send update to dashboard server"""
        try:
            import requests
            requests.post(f"{self.base_url}/update", json=data)
        except Exception as e:
            logger.error(f"Failed to send dashboard update: {e}")

# Example usage:
if __name__ == '__main__':
    # Start dashboard server
    dashboard = DashboardServer()
    
    # In a separate thread/process:
    updater = DashboardUpdater()
    
    # Send updates
    updater.update_price('RELIANCE', {
        'open': 2500,
        'high': 2550,
        'low': 2480,
        'close': 2520,
        'volume': 1000000
    })
    
    updater.add_alert("New trading signal: Buy RELIANCE", color='green')
    
    # Run the server
    dashboard.run()