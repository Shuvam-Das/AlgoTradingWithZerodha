from typing import Dict, List, Any
import pandas as pd


def calculate_market_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key market metrics"""
    metrics = {}
    
    # Price metrics
    metrics['price_change'] = data['close'].pct_change().iloc[-1]
    metrics['price_volatility'] = data['close'].pct_change().std()
    
    # Volume metrics
    metrics['volume_change'] = data['volume'].pct_change().iloc[-1]
    metrics['relative_volume'] = (
        data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
    )
    
    # Range metrics
    metrics['daily_range'] = (data['high'] - data['low']).iloc[-1]
    metrics['avg_range'] = (data['high'] - data['low']).rolling(20).mean().iloc[-1]
    
    return metrics


def calculate_execution_quality(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate execution quality metrics"""
    if not trades:
        return {
            'slippage': 0.0,
            'fill_time': 0.0,
            'rejection_rate': 0.0,
            'partial_fills': 0.0
        }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(trades)
    
    metrics = {
        'slippage': df['slippage'].mean() if 'slippage' in df else 0.0,
        'fill_time': df['fill_time'].mean() if 'fill_time' in df else 0.0,
        'rejection_rate': (
            len(df[df['status'] == 'rejected']) / len(df) 
            if 'status' in df else 0.0
        ),
        'partial_fills': (
            len(df[df['fill_type'] == 'partial']) / len(df)
            if 'fill_type' in df else 0.0
        )
    }
    
    return metrics