"""Advanced notification system for alerts and updates.

This module provides:
1. Multi-channel notifications (email, Telegram, desktop)
2. Customizable alert conditions
3. Pre-market analysis reports
4. Error monitoring and alerts
5. Performance notifications
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import telegram
import os
import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, time
import pandas as pd
import asyncio
import threading
from queue import Queue
import jinja2
from dataclasses import dataclass

logger = logging.getLogger('atwz.notifications')

@dataclass
class AlertCondition:
    type: str  # 'price', 'technical', 'volume', 'news', 'system'
    symbol: Optional[str]
    condition: str
    threshold: Union[float, str]
    message_template: str
    severity: str = 'info'  # 'info', 'warning', 'error', 'critical'
    channels: List[str] = None  # List of notification channels to use

class NotificationManager:
    """Advanced notification system with multiple channels"""
    
    def __init__(self):
        self.email_config: Dict[str, Any] = {}
        self.telegram_config: Dict[str, Any] = {}
        self.alert_conditions: List[AlertCondition] = []
        self.notification_queue: Queue = Queue()
        self._templates = self._load_templates()
        self._setup_background_worker()
    
    def configure_email(self, smtp_server: str, smtp_port: int,
                       username: str, password: str, from_address: str,
                       to_addresses: List[str]):
        """Configure email settings"""
        self.email_config = {
            'server': smtp_server,
            'port': smtp_port,
            'username': username,
            'password': password,
            'from': from_address,
            'to': to_addresses
        }
    
    def configure_telegram(self, bot_token: str, chat_ids: List[str]):
        """Configure Telegram settings"""
        self.telegram_config = {
            'bot_token': bot_token,
            'chat_ids': chat_ids,
            'bot': telegram.Bot(token=bot_token)
        }
    
    def add_alert_condition(self, condition: AlertCondition):
        """Add a new alert condition to monitor"""
        self.alert_conditions.append(condition)
    
    async def send_notification(self, message: str, severity: str = 'info',
                              channels: Optional[List[str]] = None,
                              data: Optional[Dict[str, Any]] = None):
        """Send notification through specified channels"""
        if channels is None:
            channels = ['email', 'telegram']
        
        tasks = []
        
        if 'email' in channels and self.email_config:
            tasks.append(self._send_email(message, severity, data))
        
        if 'telegram' in channels and self.telegram_config:
            tasks.append(self._send_telegram(message, severity, data))
        
        await asyncio.gather(*tasks)
    
    def check_alert_conditions(self, market_data: Dict[str, Any]):
        """Check all alert conditions against current market data"""
        for condition in self.alert_conditions:
            try:
                if self._evaluate_condition(condition, market_data):
                    message = self._format_alert_message(condition, market_data)
                    self.notification_queue.put({
                        'message': message,
                        'severity': condition.severity,
                        'channels': condition.channels,
                        'data': market_data
                    })
            except Exception as e:
                logger.error(f"Error checking alert condition: {e}")
    
    async def send_daily_analysis(self, portfolio_data: Dict[str, Any],
                                market_analysis: Dict[str, Any]):
        """Send pre-market daily analysis report"""
        template = self._templates.get_template('daily_analysis.html')
        
        # Prepare report data
        report_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio': portfolio_data,
            'market': market_analysis,
            'recommendations': self._generate_recommendations(portfolio_data, market_analysis)
        }
        
        # Render HTML report
        html_content = template.render(**report_data)
        
        # Send through all configured channels
        await self.send_notification(
            message="Daily Market Analysis",
            data={'html': html_content},
            channels=['email']
        )
    
    def monitor_system_health(self, metrics: Dict[str, Any]):
        """Monitor system health and send alerts if needed"""
        # Check CPU usage
        if metrics.get('cpu_usage', 0) > 80:
            self.notification_queue.put({
                'message': f"High CPU usage: {metrics['cpu_usage']}%",
                'severity': 'warning',
                'channels': ['telegram'],
                'data': metrics
            })
        
        # Check memory usage
        if metrics.get('memory_usage', 0) > 80:
            self.notification_queue.put({
                'message': f"High memory usage: {metrics['memory_usage']}%",
                'severity': 'warning',
                'channels': ['telegram'],
                'data': metrics
            })
        
        # Check error rate
        error_threshold = 10
        if metrics.get('error_count', 0) > error_threshold:
            self.notification_queue.put({
                'message': f"High error rate detected: {metrics['error_count']} errors",
                'severity': 'error',
                'channels': ['email', 'telegram'],
                'data': metrics
            })
    
    async def send_performance_update(self, performance_data: Dict[str, Any]):
        """Send trading performance update"""
        template = self._templates.get_template('performance_update.html')
        
        # Render HTML report
        html_content = template.render(**performance_data)
        
        # Send through email
        await self.send_notification(
            message="Trading Performance Update",
            data={'html': html_content},
            channels=['email']
        )
    
    def _setup_background_worker(self):
        """Start background worker for processing notification queue"""
        def worker():
            while True:
                try:
                    notification = self.notification_queue.get()
                    asyncio.run(self.send_notification(
                        message=notification['message'],
                        severity=notification['severity'],
                        channels=notification['channels'],
                        data=notification.get('data')
                    ))
                except Exception as e:
                    logger.error(f"Error in notification worker: {e}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    async def _send_email(self, message: str, severity: str,
                         data: Optional[Dict[str, Any]] = None):
        """Send email notification"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{severity.upper()}] Trading Alert"
            msg['From'] = self.email_config['from']
            msg['To'] = ', '.join(self.email_config['to'])
            
            # Plain text content
            text_content = message
            msg.attach(MIMEText(text_content, 'plain'))
            
            # HTML content if provided
            if data and 'html' in data:
                msg.attach(MIMEText(data['html'], 'html'))
            
            # Send email
            with smtplib.SMTP(self.email_config['server'],
                            self.email_config['port']) as server:
                server.starttls()
                server.login(self.email_config['username'],
                           self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email notification sent: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    async def _send_telegram(self, message: str, severity: str,
                           data: Optional[Dict[str, Any]] = None):
        """Send Telegram notification"""
        try:
            # Format message with severity emoji
            emoji_map = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '🚨',
                'critical': '🔥'
            }
            emoji = emoji_map.get(severity, 'ℹ️')
            formatted_message = f"{emoji} {message}"
            
            # Send to all configured chat IDs
            for chat_id in self.telegram_config['chat_ids']:
                await self.telegram_config['bot'].send_message(
                    chat_id=chat_id,
                    text=formatted_message,
                    parse_mode='HTML'
                )
            
            logger.info(f"Telegram notification sent: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def _evaluate_condition(self, condition: AlertCondition,
                          market_data: Dict[str, Any]) -> bool:
        """Evaluate if an alert condition is met"""
        try:
            if condition.type == 'price':
                current_price = market_data.get(condition.symbol, {}).get('price')
                if current_price is None:
                    return False
                
                if condition.condition == 'above':
                    return current_price > condition.threshold
                elif condition.condition == 'below':
                    return current_price < condition.threshold
            
            elif condition.type == 'technical':
                if condition.condition == 'ma_crossover':
                    data = market_data.get(condition.symbol, {})
                    return self._check_ma_crossover(data, condition.threshold)
            
            elif condition.type == 'volume':
                volume = market_data.get(condition.symbol, {}).get('volume')
                avg_volume = market_data.get(condition.symbol, {}).get('avg_volume')
                if volume is None or avg_volume is None:
                    return False
                
                if condition.condition == 'spike':
                    return volume > avg_volume * condition.threshold
            
            elif condition.type == 'system':
                if condition.condition in market_data:
                    return market_data[condition.condition] > condition.threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _format_alert_message(self, condition: AlertCondition,
                            market_data: Dict[str, Any]) -> str:
        """Format alert message using condition template"""
        try:
            return condition.message_template.format(
                symbol=condition.symbol,
                price=market_data.get(condition.symbol, {}).get('price'),
                threshold=condition.threshold,
                **market_data
            )
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return f"Alert triggered for {condition.symbol}"
    
    def _check_ma_crossover(self, data: Dict[str, Any], ma_periods: str) -> bool:
        """Check for moving average crossover"""
        try:
            short_period, long_period = map(int, ma_periods.split(','))
            
            prices = data.get('prices', [])
            if len(prices) < long_period:
                return False
            
            df = pd.Series(prices)
            short_ma = df.rolling(short_period).mean()
            long_ma = df.rolling(long_period).mean()
            
            # Check if short MA crossed above long MA
            return (short_ma.iloc[-2] <= long_ma.iloc[-2] and 
                   short_ma.iloc[-1] > long_ma.iloc[-1])
            
        except Exception as e:
            logger.error(f"Error checking MA crossover: {e}")
            return False
    
    def _load_templates(self) -> jinja2.Environment:
        """Load Jinja2 templates"""
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(template_dir, exist_ok=True)
        
        # Create default templates if they don't exist
        self._create_default_templates(template_dir)
        
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=True
        )
    
    def _create_default_templates(self, template_dir: str):
        """Create default notification templates"""
        templates = {
            'daily_analysis.html': '''
                <h1>Daily Market Analysis - {{ date }}</h1>
                <h2>Portfolio Summary</h2>
                <ul>
                    <li>Total Value: ${{ portfolio.total_value }}</li>
                    <li>Daily P&L: ${{ portfolio.daily_pnl }}</li>
                    <li>Open Positions: {{ portfolio.positions }}</li>
                </ul>
                
                <h2>Market Overview</h2>
                {{ market.summary }}
                
                <h2>Recommendations</h2>
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            ''',
            
            'performance_update.html': '''
                <h1>Trading Performance Update</h1>
                <h2>Summary</h2>
                <ul>
                    <li>Total Return: {{ total_return }}%</li>
                    <li>Win Rate: {{ win_rate }}%</li>
                    <li>Sharpe Ratio: {{ sharpe_ratio }}</li>
                </ul>
                
                <h2>Recent Trades</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Return</th>
                    </tr>
                    {% for trade in recent_trades %}
                    <tr>
                        <td>{{ trade.symbol }}</td>
                        <td>{{ trade.type }}</td>
                        <td>{{ trade.return }}%</td>
                    </tr>
                    {% endfor %}
                </table>
            '''
        }
        
        for name, content in templates.items():
            path = os.path.join(template_dir, name)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    f.write(content.strip())
    
    def _generate_recommendations(self, portfolio_data: Dict[str, Any],
                                market_analysis: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        # Portfolio-based recommendations
        for position in portfolio_data.get('positions', []):
            # Check stop loss
            if position['unrealized_pnl_pct'] < -5:
                recommendations.append(
                    f"Consider closing {position['symbol']} position - "
                    f"Stop loss threshold reached ({position['unrealized_pnl_pct']}%)"
                )
            
            # Check profit taking
            elif position['unrealized_pnl_pct'] > 20:
                recommendations.append(
                    f"Consider taking profits on {position['symbol']} - "
                    f"Target reached ({position['unrealized_pnl_pct']}%)"
                )
        
        # Market-based recommendations
        for signal in market_analysis.get('signals', []):
            if signal['strength'] > 0.7:  # Strong signals only
                recommendations.append(
                    f"{signal['type'].title()} signal for {signal['symbol']}: "
                    f"{signal['description']}"
                )
        
        return recommendations

# Example usage
if __name__ == '__main__':
    # Initialize notification manager
    nm = NotificationManager()
    
    # Configure channels
    nm.configure_email(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username=os.getenv("SMTP_USER"),
        password=os.getenv("SMTP_PASS"),
        from_address="trading@example.com",
        to_addresses=["trader@example.com"]
    )
    
    nm.configure_telegram(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_ids=[os.getenv("TELEGRAM_CHAT_ID")]
    )
    
    # Add some alert conditions
    nm.add_alert_condition(AlertCondition(
        type='price',
        symbol='RELIANCE',
        condition='above',
        threshold=2500,
        message_template="{symbol} price crossed above {threshold}",
        severity='info',
        channels=['telegram']
    ))
    
    # Monitor market data
    market_data = {
        'RELIANCE': {
            'price': 2550,
            'volume': 1000000
        }
    }
    
    nm.check_alert_conditions(market_data)