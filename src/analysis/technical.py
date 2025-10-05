"""Advanced technical analysis module with comprehensive indicators and patterns.

This module provides advanced technical analysis capabilities:
1. Complex technical indicators
2. Chart pattern recognition
3. Volume profile analysis
4. Support/resistance detection
5. Correlation analysis
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger('atwz.analysis')

@dataclass
class ChartPattern:
    pattern: str
    start_idx: int
    end_idx: int
    confidence: float
    direction: str  # 'bullish' or 'bearish'

@dataclass
class SupportResistance:
    level: float
    strength: float
    type: str  # 'support' or 'resistance'
    start_idx: int
    end_idx: Optional[int] = None

class TechnicalAnalyzer:
    """Comprehensive technical analysis engine"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with OHLCV DataFrame"""
        self.df = df.copy()
        self._validate_data()
        self.results: Dict[str, pd.Series] = {}
        
    def _validate_data(self):
        """Ensure required columns exist"""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def add_all_indicators(self) -> pd.DataFrame:
        """Calculate and add all technical indicators"""
        df = self.df.copy()
        
        # Trend Indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # Momentum Indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # Volatility Indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        upper, middle, lower = talib.BBANDS(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Volume Indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Additional Custom Indicators
        df['price_volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def detect_patterns(self) -> List[ChartPattern]:
        """Detect chart patterns using TA-Lib"""
        patterns = []
        
        # Candlestick Patterns
        pattern_functions = {
            'CDLDOJI': 'Doji',
            'CDLENGULFING': 'Engulfing',
            'CDLHARAMI': 'Harami',
            'CDLMORNINGSTAR': 'Morning Star',
            'CDLEVENINGSTAR': 'Evening Star'
        }
        
        for func_name, pattern_name in pattern_functions.items():
            try:
                pattern_func = getattr(talib, func_name)
                result = pattern_func(self.df['open'], self.df['high'], 
                                   self.df['low'], self.df['close'])
                
                for i in range(len(result)):
                    if result[i] != 0:
                        direction = 'bullish' if result[i] > 0 else 'bearish'
                        confidence = abs(result[i]) / 100.0
                        patterns.append(ChartPattern(
                            pattern=pattern_name,
                            start_idx=max(0, i-1),
                            end_idx=i,
                            confidence=confidence,
                            direction=direction
                        ))
            except Exception as e:
                logger.warning(f"Pattern detection failed for {func_name}: {e}")
        
        return patterns
    
    def find_support_resistance(self, 
                              window: int = 20, 
                              threshold: float = 0.02) -> List[SupportResistance]:
        """Identify support and resistance levels"""
        levels = []
        price_range = self.df['high'].max() - self.df['low'].min()
        
        for i in range(window, len(self.df)):
            window_data = self.df.iloc[i-window:i]
            
            # Look for price levels with multiple touches
            highs = window_data['high'].value_counts()
            lows = window_data['low'].value_counts()
            
            # Find resistance levels
            for price, count in highs.items():
                if count >= 3:  # At least 3 touches
                    strength = count / window
                    price_diff = abs(price - self.df['close'].iloc[i]) / price_range
                    if price_diff < threshold:
                        levels.append(SupportResistance(
                            level=price,
                            strength=strength,
                            type='resistance',
                            start_idx=i-window,
                            end_idx=i
                        ))
            
            # Find support levels
            for price, count in lows.items():
                if count >= 3:
                    strength = count / window
                    price_diff = abs(price - self.df['close'].iloc[i]) / price_range
                    if price_diff < threshold:
                        levels.append(SupportResistance(
                            level=price,
                            strength=strength,
                            type='support',
                            start_idx=i-window,
                            end_idx=i
                        ))
        
        return levels
    
    def volume_profile(self, price_bins: int = 50) -> pd.DataFrame:
        """Calculate volume profile"""
        price_range = pd.interval_range(
            start=self.df['low'].min(),
            end=self.df['high'].max(),
            periods=price_bins
        )
        
        volume_profile = pd.DataFrame(index=price_range)
        volume_profile['volume'] = 0
        
        for idx, row in self.df.iterrows():
            # Find which price bins this candle spans
            for interval in price_range:
                if (interval.left <= row['high'] and interval.right >= row['low']):
                    # Roughly distribute volume across the candle's range
                    volume_profile.loc[interval, 'volume'] += row['volume'] / price_bins
        
        return volume_profile.sort_values('volume', ascending=False)
    
    def correlation_analysis(self, symbols: List[str], 
                           data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze correlations between multiple symbols"""
        returns = pd.DataFrame()
        
        # Calculate returns for each symbol
        for symbol in symbols:
            if symbol in data:
                returns[symbol] = data[symbol]['close'].pct_change()
        
        # Correlation matrix
        corr_matrix = returns.corr()
        
        # Add correlation metadata
        corr_meta = pd.DataFrame(index=corr_matrix.index)
        corr_meta['avg_correlation'] = corr_matrix.mean()
        corr_meta['volatility'] = returns.std()
        corr_meta['beta'] = returns.cov()[symbols[0]] / returns[symbols[0]].var()
        
        return corr_meta
    
    def divergence_analysis(self) -> List[Dict[str, Any]]:
        """Detect RSI and MACD divergences"""
        divergences = []
        
        # Calculate indicators if not already present
        if 'rsi' not in self.df.columns:
            self.df['rsi'] = talib.RSI(self.df['close'])
        if 'macd' not in self.df.columns:
            self.df['macd'], _, _ = talib.MACD(self.df['close'])
        
        # Look for divergences in windows
        window = 20
        for i in range(window, len(self.df)):
            window_data = self.df.iloc[i-window:i]
            
            # Price making higher highs but RSI making lower highs (bearish)
            if (window_data['close'].iloc[-1] > window_data['close'].max() and 
                window_data['rsi'].iloc[-1] < window_data['rsi'].max()):
                divergences.append({
                    'type': 'bearish',
                    'indicator': 'RSI',
                    'index': i,
                    'strength': (window_data['rsi'].max() - window_data['rsi'].iloc[-1]) / 
                               window_data['rsi'].std()
                })
            
            # Price making lower lows but RSI making higher lows (bullish)
            if (window_data['close'].iloc[-1] < window_data['close'].min() and
                window_data['rsi'].iloc[-1] > window_data['rsi'].min()):
                divergences.append({
                    'type': 'bullish',
                    'indicator': 'RSI',
                    'index': i,
                    'strength': (window_data['rsi'].iloc[-1] - window_data['rsi'].min()) /
                               window_data['rsi'].std()
                })
        
        return divergences