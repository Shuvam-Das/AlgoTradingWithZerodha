"""Fundamental analysis module for stock valuation and company metrics.

This module provides tools for:
1. Financial ratio analysis
2. Company fundamentals fetching
3. Valuation models:
   - Discounted Cash Flow (DCF)
   - Comparable Company Analysis
   - Dividend Discount Model (DDM)
4. Industry/sector analysis with peer comparison
5. News sentiment integration for sentiment-adjusted valuations
"""

from .sentiment import NewsSentimentAnalyzer, SentimentAnalysis
from src.ml.model import predict  # For ML-enhanced forecasts

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta

logger = logging.getLogger('atwz.fundamental')

@dataclass
class CompanyMetrics:
    symbol: str
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    debt_to_equity: float
    current_ratio: float
    roe: float
    roa: float
    eps_growth: float
    revenue_growth: float
    timestamp: datetime

@dataclass
class DividendAnalysis:
    symbol: str
    dividend_yield: float
    payout_ratio: float
    dividend_growth_rate: float
    years_of_growth: int
    dividend_coverage: float
    next_dividend_date: Optional[datetime]
    dividend_sustainability: float  # 0-1 score
    timestamp: datetime

@dataclass
class FinancialHealth:
    symbol: str
    altman_z_score: float
    piotroski_score: int  # 0-9 score
    beneish_m_score: float
    leverage_ratio: float
    interest_coverage: float
    cash_ratio: float
    overall_health_score: float  # 0-100 score
    timestamp: datetime

@dataclass
class ESGMetrics:
    symbol: str
    environmental_score: float
    social_score: float
    governance_score: float
    controversy_score: float
    overall_esg_score: float
    peer_comparison: Dict[str, float]
    timestamp: datetime

@dataclass
class IndustryMetrics:
    sector: str
    industry: str
    peer_symbols: List[str]
    avg_pe: float
    avg_pb: float
    avg_roe: float
    market_leader: str  # Symbol of market leader
    total_market_cap: float
    growth_rate: float
    esg_ranking: Dict[str, float]  # ESG scores by peer
    dividend_metrics: Dict[str, float]  # Dividend yields by peer
    timestamp: datetime

@dataclass
class ValuationModel:
    symbol: str
    model_type: str  # DCF, Comparable, DDM
    fair_value: float
    confidence: float
    assumptions: Dict[str, Any]
    timestamp: datetime
    peer_metrics: Optional[IndustryMetrics] = None
    sentiment_adjusted: bool = False

class FundamentalAnalyzer:
    """Comprehensive fundamental analysis engine"""
    
    def __init__(self, symbols: List[str]):
        """Initialize with list of symbols to analyze"""
        self.symbols = symbols
        self.metrics: Dict[str, CompanyMetrics] = {}
        self.industry_metrics: Dict[str, IndustryMetrics] = {}
        self.valuations: Dict[str, List[ValuationModel]] = {}
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.dividend_metrics: Dict[str, DividendAnalysis] = {}
        self.health_scores: Dict[str, FinancialHealth] = {}
        self.esg_metrics: Dict[str, ESGMetrics] = {}
        self.cache_time = datetime.now()
        self.cache_duration = timedelta(hours=24)
        # Initialize technical analysis for correlation
        from src.analysis.technical import TechnicalAnalyzer
        self.technical_analyzer = TechnicalAnalyzer()
        # Real-time data feed integration
        self.realtime_callbacks: List[Any] = []
        self.last_realtime_update: Optional[datetime] = None
        # Strategy auto-adjustment
        self.strategy_parameters: Dict[str, Any] = {}

    def subscribe_realtime(self, callback):
        """Subscribe a callback for real-time data updates"""
        self.realtime_callbacks.append(callback)

    def on_realtime_update(self, symbol: str, tick: Dict[str, Any]):
        """Handle real-time market data update"""
        self.last_realtime_update = datetime.now()
        # Update metrics with latest tick
        if symbol in self.symbols:
            try:
                metrics = self.fetch_company_metrics(symbol)
                self.metrics[symbol] = metrics
                # Auto-adjust strategy if needed
                self.auto_adjust_strategy(symbol, metrics)
            except Exception as e:
                logger.warning(f"Error updating real-time metrics for {symbol}: {e}")
        # Notify subscribers
        for cb in self.realtime_callbacks:
            try:
                cb(symbol, tick)
            except Exception as e:
                logger.error(f"Error in realtime callback: {e}")

    def auto_adjust_strategy(self, symbol: str, metrics: CompanyMetrics):
        """Automatically adjust strategy parameters using ML predictions and live metrics"""
        try:
            from .ml_fundamental import MLFundamentalAnalyzer
            ml_analyzer = MLFundamentalAnalyzer()
            # Prepare features for ML prediction
            features = {
                'symbol': symbol,
                'market_cap': metrics.market_cap,
                'pe_ratio': metrics.pe_ratio,
                'pb_ratio': metrics.pb_ratio,
                'debt_to_equity': metrics.debt_to_equity,
                'roe': metrics.roe,
                'roa': metrics.roa,
                'eps_growth': metrics.eps_growth,
                'revenue_growth': metrics.revenue_growth
            }
            # Get ML prediction for optimal PE threshold
            ml_pred = ml_analyzer.predict_valuation(features)
            optimal_pe = max(8, min(30, ml_pred.prediction))
            # Adjust other parameters based on ML confidence and growth
            growth_pred = ml_analyzer.predict_growth(features)
            health_score = getattr(metrics, 'health_score', 70)
            if growth_pred.confidence > 0.7 and health_score > 60:
                pb_threshold = 1.2
            else:
                pb_threshold = 1.5
            self.strategy_parameters[symbol] = {
                'pe_threshold': optimal_pe,
                'pb_threshold': pb_threshold,
                'growth_target': growth_pred.prediction
            }
        except Exception as e:
            logger.warning(f"ML auto-adjust failed for {symbol}: {e}")
            # Fallback to static adjustment
            market_pe = metrics.pe_ratio
            if market_pe > 30:
                self.strategy_parameters[symbol] = {'pe_threshold': 20}
            elif market_pe < 10:
                self.strategy_parameters[symbol] = {'pe_threshold': 8}
            else:
                self.strategy_parameters[symbol] = {'pe_threshold': 15}
        
    def get_sector_peers(self, symbol: str) -> List[str]:
        """Get list of peer companies in same sector"""
        try:
            stock = yf.Ticker(symbol)
            sector = stock.info.get('sector')
            industry = stock.info.get('industry')
            
            # Use yfinance sector/industry screening
            peers = stock.info.get('recommendedSymbols', [])
            if not peers:
                # Fallback to basic peer list
                peers = [p for p in self.symbols if p != symbol]
            
            return peers[:10]  # Limit to top 10 peers
        except Exception as e:
            logger.warning(f"Error getting peers for {symbol}: {e}")
            return []
            
    def analyze_industry(self, symbol: str) -> IndustryMetrics:
        """Analyze industry metrics and peer comparison"""
        try:
            stock = yf.Ticker(symbol)
            peers = self.get_sector_peers(symbol)
            peer_metrics = []
            
            for peer in peers:
                metrics = self.fetch_company_metrics(peer)
                peer_metrics.append(metrics)
            
            # Calculate industry averages
            avg_pe = np.mean([m.pe_ratio for m in peer_metrics])
            avg_pb = np.mean([m.pb_ratio for m in peer_metrics])
            avg_roe = np.mean([m.roe for m in peer_metrics])
            
            # Find market leader (highest market cap)
            market_leader = max(
                peer_metrics,
                key=lambda x: x.market_cap
            ).symbol
            
            # Calculate total market cap
            total_market_cap = sum(m.market_cap for m in peer_metrics)
            
            # Calculate industry growth rate
            growth_rates = [m.revenue_growth for m in peer_metrics]
            avg_growth = np.mean(growth_rates)
            
            return IndustryMetrics(
                sector=stock.info.get('sector', ''),
                industry=stock.info.get('industry', ''),
                peer_symbols=peers,
                avg_pe=avg_pe,
                avg_pb=avg_pb,
                avg_roe=avg_roe,
                market_leader=market_leader,
                total_market_cap=total_market_cap,
                growth_rate=avg_growth,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing industry for {symbol}: {e}")
            return IndustryMetrics(
                sector='', industry='', peer_symbols=[],
                avg_pe=0, avg_pb=0, avg_roe=0,
                market_leader='', total_market_cap=0,
                growth_rate=0, timestamp=datetime.now()
            )
    
    def calculate_dcf_valuation(self, symbol: str, years: int = 5) -> ValuationModel:
        """Calculate Discounted Cash Flow valuation"""
        try:
            stock = yf.Ticker(symbol)
            financials = stock.financials
            cashflows = stock.cashflow
            
            # Get historical free cash flows
            fcf = cashflows.loc['Free Cash Flow']
            
            # Calculate growth rate using ML prediction
            growth_prediction = predict({
                'symbol': symbol,
                'target': 'growth_rate',
                'horizon': years
            })
            growth_rate = growth_prediction.get('predictions', [0.1])[0]
            
            # Project future cash flows
            current_fcf = fcf.iloc[0]
            future_fcfs = [
                current_fcf * (1 + growth_rate) ** i
                for i in range(1, years + 1)
            ]
            
            # Calculate terminal value
            terminal_growth = 0.02  # Long-term growth assumption
            terminal_value = (
                future_fcfs[-1] * (1 + terminal_growth) /
                (0.1 - terminal_growth)  # 10% discount rate
            )
            
            # Discount future cash flows
            discount_rate = 0.1
            present_values = [
                fcf / (1 + discount_rate) ** i
                for i, fcf in enumerate(future_fcfs, 1)
            ]
            
            # Add discounted terminal value
            present_values.append(
                terminal_value / (1 + discount_rate) ** years
            )
            
            # Calculate fair value
            fair_value = sum(present_values)
            shares_outstanding = stock.info.get('sharesOutstanding', 1)
            fair_value_per_share = fair_value / shares_outstanding
            
            # Get industry metrics for context
            industry = self.analyze_industry(symbol)
            
            # Get sentiment adjustment
            sentiment = self.sentiment_analyzer.analyze_sentiment(symbol)
            sentiment_factor = 1 + (sentiment.overall_sentiment * sentiment.confidence)
            
            return ValuationModel(
                symbol=symbol,
                model_type='DCF',
                fair_value=fair_value_per_share * sentiment_factor,
                confidence=min(0.7, 1 - abs(growth_rate)),
                assumptions={
                    'growth_rate': growth_rate,
                    'terminal_growth': terminal_growth,
                    'discount_rate': discount_rate,
                    'projection_years': years,
                    'sentiment_factor': sentiment_factor
                },
                timestamp=datetime.now(),
                peer_metrics=industry,
                sentiment_adjusted=True
            )
            
        except Exception as e:
            logger.error(f"Error calculating DCF for {symbol}: {e}")
            return ValuationModel(
                symbol=symbol,
                model_type='DCF',
                fair_value=0,
                confidence=0,
                assumptions={},
                timestamp=datetime.now()
            )
    
    def analyze_dividends(self, symbol: str) -> DividendAnalysis:
        """Analyze dividend metrics and sustainability"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            dividends = stock.dividends
            
            # Calculate basic metrics
            div_yield = info.get('dividendYield', 0) * 100
            payout = info.get('payoutRatio', 0) * 100
            
            # Calculate dividend growth
            if len(dividends) > 1:
                growth_rates = []
                for i in range(1, len(dividends)):
                    growth = (dividends.iloc[i] / dividends.iloc[i-1]) - 1
                    growth_rates.append(growth)
                avg_growth = np.mean(growth_rates)
                years_growth = len([r for r in growth_rates if r > 0])
            else:
                avg_growth = 0
                years_growth = 0
            
            # Calculate dividend coverage
            try:
                eps = info.get('trailingEPS', 0)
                last_div = dividends.iloc[-1] if len(dividends) > 0 else 0
                coverage = eps / last_div if last_div > 0 else 0
            except Exception:
                coverage = 0
            
            # Predict next dividend date
            if len(dividends) > 0:
                last_date = dividends.index[-1]
                freq = pd.infer_freq(dividends.index)
                if freq:
                    next_date = last_date + pd.Timedelta(freq)
                else:
                    next_date = None
            else:
                next_date = None
            
            # Calculate sustainability score
            sustainability = min(1.0, (
                (coverage * 0.3) +  # Weight factors
                (min(years_growth, 10) / 10 * 0.3) +
                (min(div_yield, 10) / 10 * 0.2) +
                (max(0, 1 - payout/100) * 0.2)
            ))
            
            return DividendAnalysis(
                symbol=symbol,
                dividend_yield=div_yield,
                payout_ratio=payout,
                dividend_growth_rate=avg_growth,
                years_of_growth=years_growth,
                dividend_coverage=coverage,
                next_dividend_date=next_date,
                dividend_sustainability=sustainability,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing dividends for {symbol}: {e}")
            return DividendAnalysis(
                symbol=symbol,
                dividend_yield=0,
                payout_ratio=0,
                dividend_growth_rate=0,
                years_of_growth=0,
                dividend_coverage=0,
                next_dividend_date=None,
                dividend_sustainability=0,
                timestamp=datetime.now()
            )
    
    def calculate_financial_health(self, symbol: str) -> FinancialHealth:
        """Calculate comprehensive financial health metrics"""
        try:
            stock = yf.Ticker(symbol)
            bs = stock.balance_sheet
            is_stmt = stock.income_stmt
            cf = stock.cashflow
            
            # Calculate Altman Z-Score
            try:
                working_capital = (
                    bs.loc['Total Current Assets'].iloc[0] -
                    bs.loc['Total Current Liabilities'].iloc[0]
                )
                retained_earnings = bs.loc['Retained Earnings'].iloc[0]
                ebit = is_stmt.loc['EBIT'].iloc[0]
                market_cap = stock.info.get('marketCap', 0)
                total_assets = bs.loc['Total Assets'].iloc[0]
                total_liabilities = bs.loc['Total Liabilities'].iloc[0]
                sales = is_stmt.loc['Total Revenue'].iloc[0]
                
                z_score = (
                    1.2 * (working_capital / total_assets) +
                    1.4 * (retained_earnings / total_assets) +
                    3.3 * (ebit / total_assets) +
                    0.6 * (market_cap / total_liabilities) +
                    1.0 * (sales / total_assets)
                )
            except Exception:
                z_score = 0
            
            # Calculate Piotroski F-Score
            f_score = 0
            # ROA and Operating Cash Flow positive
            if is_stmt.loc['Net Income'].iloc[0] > 0:
                f_score += 1
            if cf.loc['Operating Cash Flow'].iloc[0] > 0:
                f_score += 1
            # ROA improving
            if len(is_stmt.loc['Net Income']) > 1:
                if is_stmt.loc['Net Income'].iloc[0] > is_stmt.loc['Net Income'].iloc[1]:
                    f_score += 1
            # Leverage decreasing
            if len(bs.loc['Total Liabilities']) > 1:
                if bs.loc['Total Liabilities'].iloc[0] < bs.loc['Total Liabilities'].iloc[1]:
                    f_score += 1
            
            # Calculate Beneish M-Score for earnings manipulation detection
            try:
                if len(is_stmt) > 1 and len(bs) > 1:
                    dsri = (
                        (bs.loc['Accounts Receivable'].iloc[0] / is_stmt.loc['Total Revenue'].iloc[0]) /
                        (bs.loc['Accounts Receivable'].iloc[1] / is_stmt.loc['Total Revenue'].iloc[1])
                    )
                    aqi = (
                        (1 - (bs.loc['Current Assets'].iloc[0] + bs.loc['Property Plant Equipment'].iloc[0]) /
                         bs.loc['Total Assets'].iloc[0]) /
                        (1 - (bs.loc['Current Assets'].iloc[1] + bs.loc['Property Plant Equipment'].iloc[1]) /
                         bs.loc['Total Assets'].iloc[1])
                    )
                    m_score = -4.84 + (0.92 * dsri) + (0.528 * aqi)
                else:
                    m_score = 0
            except Exception:
                m_score = 0
            
            # Calculate additional ratios
            try:
                leverage = total_liabilities / total_assets
                interest_coverage = ebit / is_stmt.loc['Interest Expense'].iloc[0]
                cash_ratio = bs.loc['Cash'].iloc[0] / bs.loc['Total Current Liabilities'].iloc[0]
            except Exception:
                leverage = interest_coverage = cash_ratio = 0
            
            # Calculate overall health score (0-100)
            health_score = min(100, max(0, (
                (min(z_score, 5) / 5 * 30) +  # 30% weight to Z-score
                (f_score / 9 * 30) +          # 30% weight to F-score
                (max(0, 2-leverage) / 2 * 20) +  # 20% weight to leverage
                (min(interest_coverage, 10) / 10 * 20)  # 20% weight to interest coverage
            )))
            
            return FinancialHealth(
                symbol=symbol,
                altman_z_score=z_score,
                piotroski_score=f_score,
                beneish_m_score=m_score,
                leverage_ratio=leverage,
                interest_coverage=interest_coverage,
                cash_ratio=cash_ratio,
                overall_health_score=health_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating financial health for {symbol}: {e}")
            return FinancialHealth(
                symbol=symbol,
                altman_z_score=0,
                piotroski_score=0,
                beneish_m_score=0,
                leverage_ratio=0,
                interest_coverage=0,
                cash_ratio=0,
                overall_health_score=0,
                timestamp=datetime.now()
            )
    
    def analyze_esg(self, symbol: str) -> ESGMetrics:
        """Analyze Environmental, Social, and Governance metrics"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get ESG scores from Yahoo Finance
            esg_data = info.get('esgScores', {})
            
            env_score = esg_data.get('environmentScore', 0)
            social_score = esg_data.get('socialScore', 0)
            gov_score = esg_data.get('governanceScore', 0)
            controversy = esg_data.get('controversyLevel', 0)
            
            # Get peer comparison
            peers = self.get_sector_peers(symbol)
            peer_scores = {}
            
            for peer in peers:
                try:
                    peer_stock = yf.Ticker(peer)
                    peer_esg = peer_stock.info.get('esgScores', {})
                    peer_scores[peer] = peer_esg.get('totalEsg', 0)
                except Exception:
                    continue
            
            # Calculate overall score
            overall_score = (
                env_score * 0.4 +      # 40% weight to environmental
                social_score * 0.3 +   # 30% weight to social
                gov_score * 0.3        # 30% weight to governance
            ) * (1 - controversy/10)   # Penalize for controversies
            
            return ESGMetrics(
                symbol=symbol,
                environmental_score=env_score,
                social_score=social_score,
                governance_score=gov_score,
                controversy_score=controversy,
                overall_esg_score=overall_score,
                peer_comparison=peer_scores,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing ESG metrics for {symbol}: {e}")
            return ESGMetrics(
                symbol=symbol,
                environmental_score=0,
                social_score=0,
                governance_score=0,
                controversy_score=0,
                overall_esg_score=0,
                peer_comparison={},
                timestamp=datetime.now()
            )
    
    def analyze_technical_fundamental_correlation(self, symbol: str) -> Dict[str, Any]:
        """Analyze correlation between technical and fundamental indicators"""
        try:
            # Get technical indicators
            technical = self.technical_analyzer.calculate_all_indicators(symbol)
            
            # Get fundamental metrics
            fundamentals = self.fetch_company_metrics(symbol)
            health = self.calculate_financial_health(symbol)
            dividends = self.analyze_dividends(symbol)
            esg = self.analyze_esg(symbol)
            
            # Calculate correlations
            price_data = pd.DataFrame({
                'price': technical['prices'],
                'volume': technical['volume'],
                'rsi': technical['rsi'],
                'macd': technical['macd'],
                'health_score': [health.overall_health_score] * len(technical['prices']),
                'dividend_score': [dividends.dividend_sustainability] * len(technical['prices']),
                'esg_score': [esg.overall_esg_score] * len(technical['prices'])
            })
            
            correlations = price_data.corr()
            
            # Generate insights
            insights = {
                'price_health_correlation': correlations.loc['price', 'health_score'],
                'volume_health_correlation': correlations.loc['volume', 'health_score'],
                'rsi_fundamental_alignment': abs(correlations.loc['rsi', 'health_score']) > 0.6,
                'macd_fundamental_alignment': abs(correlations.loc['macd', 'health_score']) > 0.6,
                'esg_price_impact': correlations.loc['price', 'esg_score'],
                'dividend_price_correlation': correlations.loc['price', 'dividend_score']
            }
            
            return {
                'correlations': correlations.to_dict(),
                'insights': insights,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing technical-fundamental correlation for {symbol}: {e}")
            return {
                'correlations': {},
                'insights': {},
                'timestamp': datetime.now()
            }
    
    def calculate_comparable_valuation(self, symbol: str) -> ValuationModel:
        """Calculate valuation using comparable company analysis"""
        try:
            stock = yf.Ticker(symbol)
            industry = self.analyze_industry(symbol)
            metrics = self.fetch_company_metrics(symbol)
            
            # Calculate fair value using industry average multiples
            pe_value = metrics.eps_growth * industry.avg_pe
            pb_value = metrics.pb_ratio * industry.avg_pb
            
            # Weight the different methods
            fair_value = (pe_value * 0.6 + pb_value * 0.4)
            
            # Adjust based on company's relative strength
            if metrics.roe > industry.avg_roe:
                fair_value *= 1.1  # 10% premium
            
            # Get sentiment adjustment
            sentiment = self.sentiment_analyzer.analyze_sentiment(symbol)
            sentiment_factor = 1 + (sentiment.overall_sentiment * sentiment.confidence)
            
            return ValuationModel(
                symbol=symbol,
                model_type='Comparable',
                fair_value=fair_value * sentiment_factor,
                confidence=0.6,  # Lower confidence than DCF
                assumptions={
                    'industry_pe': industry.avg_pe,
                    'industry_pb': industry.avg_pb,
                    'industry_roe': industry.avg_roe,
                    'pe_weight': 0.6,
                    'pb_weight': 0.4,
                    'sentiment_factor': sentiment_factor
                },
                timestamp=datetime.now(),
                peer_metrics=industry,
                sentiment_adjusted=True
            )
            
        except Exception as e:
            logger.error(f"Error calculating comparable valuation for {symbol}: {e}")
            return ValuationModel(
                symbol=symbol,
                model_type='Comparable',
                fair_value=0,
                confidence=0,
                assumptions={},
                timestamp=datetime.now()
            )
    
    def fetch_company_metrics(self, symbol: str) -> CompanyMetrics:
        """Fetch key financial metrics for a company"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Calculate metrics
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            
            # Get financial ratios from statements
            try:
                bs = stock.balance_sheet
                is_stmt = stock.income_stmt
                
                total_debt = bs.loc['Total Debt'].iloc[0]
                total_equity = bs.loc['Total Stockholder Equity'].iloc[0]
                debt_to_equity = total_debt / total_equity if total_equity != 0 else 0
                
                current_assets = bs.loc['Total Current Assets'].iloc[0]
                current_liab = bs.loc['Total Current Liabilities'].iloc[0]
                current_ratio = current_assets / current_liab if current_liab != 0 else 0
                
                net_income = is_stmt.loc['Net Income'].iloc[0]
                roe = net_income / total_equity if total_equity != 0 else 0
                total_assets = bs.loc['Total Assets'].iloc[0]
                roa = net_income / total_assets if total_assets != 0 else 0
                
                # Growth metrics
                eps_series = pd.Series(stock.earnings_per_share)
                eps_growth = (eps_series.iloc[-1] / eps_series.iloc[0] - 1) if len(eps_series) > 1 else 0
                
                revenue = is_stmt.loc['Total Revenue']
                revenue_growth = (revenue.iloc[0] / revenue.iloc[-1] - 1) if len(revenue) > 1 else 0
                
            except Exception as e:
                logger.warning(f"Error calculating detailed metrics for {symbol}: {e}")
                debt_to_equity = current_ratio = roe = roa = eps_growth = revenue_growth = 0
            
            metrics = CompanyMetrics(
                symbol=symbol,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                debt_to_equity=debt_to_equity,
                current_ratio=current_ratio,
                roe=roe,
                roa=roa,
                eps_growth=eps_growth,
                revenue_growth=revenue_growth,
                timestamp=datetime.now()
            )
            
            self.metrics[symbol] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to fetch metrics for {symbol}: {e}")
            raise
    
    def run_dcf_valuation(self, symbol: str, 
                         growth_rate: float,
                         discount_rate: float = 0.1,
                         periods: int = 5) -> ValuationModel:
        """Run a Discounted Cash Flow valuation"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get free cash flow history
            try:
                cf = stock.cashflow
                fcf = cf.loc['Free Cash Flow']
                base_fcf = fcf.iloc[0]
            except:
                logger.warning(f"FCF not found for {symbol}, using net income")
                is_stmt = stock.income_stmt
                base_fcf = is_stmt.loc['Net Income'].iloc[0]
            
            # Project future cash flows
            future_fcf = []
            for i in range(periods):
                fcf_t = base_fcf * (1 + growth_rate)**(i+1)
                future_fcf.append(fcf_t)
            
            # Calculate terminal value
            terminal_growth = min(growth_rate, 0.03)  # Cap at 3%
            terminal_value = future_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            future_fcf.append(terminal_value)
            
            # Discount cash flows
            present_values = []
            for i, fcf in enumerate(future_fcf):
                pv = fcf / (1 + discount_rate)**(i+1)
                present_values.append(pv)
            
            enterprise_value = sum(present_values)
            
            # Adjust for cash and debt
            try:
                bs = stock.balance_sheet
                cash = bs.loc['Cash'].iloc[0]
                debt = bs.loc['Total Debt'].iloc[0]
                equity_value = enterprise_value + cash - debt
            except:
                equity_value = enterprise_value
            
            shares_outstanding = stock.info.get('sharesOutstanding', 1)
            fair_value = equity_value / shares_outstanding
            
            assumptions = {
                'growth_rate': growth_rate,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'periods': periods
            }
            
            return ValuationModel(
                symbol=symbol,
                model_type='DCF',
                fair_value=fair_value,
                confidence=0.7,  # Could be adjusted based on inputs quality
                assumptions=assumptions,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"DCF valuation failed for {symbol}: {e}")
            raise
    
    def get_relative_valuation(self, symbol: str, peers: List[str]) -> Dict[str, Any]:
        """Compare company valuation metrics to peers"""
        metrics = []
        
        # Fetch metrics for symbol and peers
        for sym in [symbol] + peers:
            try:
                if sym not in self.metrics or \
                   datetime.now() - self.metrics[sym].timestamp > self.cache_duration:
                    self.fetch_company_metrics(sym)
                metrics.append(self.metrics[sym])
            except Exception:
                logger.warning(f"Could not fetch metrics for {sym}")
                
        if not metrics:
            raise ValueError("No metrics available for comparison")
        
        # Calculate peer averages
        peer_metrics = metrics[1:]  # Exclude the main symbol
        if not peer_metrics:
            raise ValueError("No peer metrics available")
        
        peer_averages = {
            'pe_ratio': np.mean([m.pe_ratio for m in peer_metrics if m.pe_ratio > 0]),
            'pb_ratio': np.mean([m.pb_ratio for m in peer_metrics if m.pb_ratio > 0]),
            'debt_to_equity': np.mean([m.debt_to_equity for m in peer_metrics]),
            'roe': np.mean([m.roe for m in peer_metrics]),
        }
        
        # Compare to symbol's metrics
        symbol_metrics = metrics[0]
        comparison = {}
        for metric, peer_avg in peer_averages.items():
            symbol_value = getattr(symbol_metrics, metric)
            if peer_avg != 0:
                comparison[metric] = {
                    'symbol_value': symbol_value,
                    'peer_average': peer_avg,
                    'difference_pct': (symbol_value / peer_avg - 1) * 100 if peer_avg != 0 else 0
                }
        
        return comparison
    
    def get_industry_analysis(self, symbol: str) -> Dict[str, Any]:
        """Analyze industry trends and market position"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            industry = info.get('industry')
            sector = info.get('sector')
            
            # Get industry peers
            peers = stock.info.get('recommendationKey', [])
            
            # Fetch sector ETF if available
            sector_etf = None
            sector_mapping = {
                'Technology': 'XLK',
                'Financial': 'XLF',
                'Healthcare': 'XLV',
                'Consumer Cyclical': 'XLY',
                'Consumer Defensive': 'XLP',
                'Energy': 'XLE',
                'Basic Materials': 'XLB',
                'Industrials': 'XLI',
                'Real Estate': 'XLRE',
                'Utilities': 'XLU',
            }
            
            if sector in sector_mapping:
                try:
                    sector_etf = yf.Ticker(sector_mapping[sector])
                except:
                    pass
            
            analysis = {
                'industry': industry,
                'sector': sector,
                'market_cap_rank': None,
                'revenue_rank': None,
                'growth_rank': None,
                'sector_performance': None
            }
            
            # Add sector performance if ETF available
            if sector_etf:
                try:
                    sector_hist = sector_etf.history(period='1y')
                    sector_return = (sector_hist['Close'][-1] / sector_hist['Close'][0] - 1) * 100
                    analysis['sector_performance'] = sector_return
                except:
                    pass
            
            return analysis
            
        except Exception as e:
            logger.error(f"Industry analysis failed for {symbol}: {e}")
            raise
    
    def get_news_sentiment(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Analyze recent news sentiment"""
        try:
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                return {'error': 'No news available'}
            
            # Use sentiment analyzer if available
            try:
                from transformers import pipeline
                sentiment = pipeline('sentiment-analysis')
                
                analyzed = []
                for article in news:
                    title = article.get('title', '')
                    if title:
                        result = sentiment(title)[0]
                        analyzed.append({
                            'title': title,
                            'date': article.get('date'),
                            'sentiment': result['label'],
                            'score': result['score']
                        })
                
                # Aggregate sentiment
                positive = sum(1 for a in analyzed if a['sentiment'] == 'POSITIVE')
                negative = sum(1 for a in analyzed if a['sentiment'] == 'NEGATIVE')
                
                return {
                    'total_articles': len(analyzed),
                    'positive_ratio': positive / len(analyzed) if analyzed else 0,
                    'negative_ratio': negative / len(analyzed) if analyzed else 0,
                    'articles': analyzed
                }
                
            except ImportError:
                # Fallback to simple keyword analysis
                keywords = {
                    'positive': ['growth', 'profit', 'beat', 'up', 'gain', 'positive'],
                    'negative': ['loss', 'down', 'fall', 'cut', 'negative', 'risk']
                }
                
                analyzed = []
                for article in news:
                    title = article.get('title', '').lower()
                    pos_count = sum(1 for w in keywords['positive'] if w in title)
                    neg_count = sum(1 for w in keywords['negative'] if w in title)
                    sentiment = 'POSITIVE' if pos_count > neg_count else 'NEGATIVE' if neg_count > pos_count else 'NEUTRAL'
                    
                    analyzed.append({
                        'title': article.get('title'),
                        'date': article.get('date'),
                        'sentiment': sentiment
                    })
                
                positive = sum(1 for a in analyzed if a['sentiment'] == 'POSITIVE')
                negative = sum(1 for a in analyzed if a['sentiment'] == 'NEGATIVE')
                
                return {
                    'total_articles': len(analyzed),
                    'positive_ratio': positive / len(analyzed) if analyzed else 0,
                    'negative_ratio': negative / len(analyzed) if analyzed else 0,
                    'articles': analyzed[:10]  # Limit to recent 10
                }
        
        except Exception as e:
            logger.error(f"News sentiment analysis failed for {symbol}: {e}")
            raise