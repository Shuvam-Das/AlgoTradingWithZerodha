"""Real-time chart manager for advanced visualizations"""
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ChartManager:
    """Manages advanced chart creation and updates"""
    
    @staticmethod
    def create_advanced_chart(
        market_data: pd.DataFrame,
        ml_predictions: Dict[str, Any],
        risk_levels: Optional[Dict[str, float]] = None
    ) -> go.Figure:
        """Create an advanced chart with ML insights and risk levels"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price with ML Predictions', 'Trading Volume & Signals')
        )
        
        # Main price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=market_data.index,
                open=market_data['open'],
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add ML prediction overlay
        if ml_predictions:
            pred_df = pd.DataFrame(ml_predictions)
            fig.add_trace(
                go.Scatter(
                    x=pred_df.index,
                    y=pred_df['prediction'],
                    mode='lines',
                    line=dict(color='rgba(255, 165, 0, 0.5)', width=2),
                    name='ML Prediction'
                ),
                row=1, col=1
            )
        
        # Add risk levels if provided
        if risk_levels:
            for level_name, level_value in risk_levels.items():
                fig.add_hline(
                    y=level_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=level_name,
                    row=1, col=1
                )
        
        # Volume bars in subplot
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(market_data['close'], market_data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=market_data.index,
                y=market_data['volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text="Market Analysis with ML Insights",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_ml_confidence_chart(
        confidence_data: Dict[str, float],
        threshold: float = 0.8
    ) -> go.Figure:
        """Create ML confidence visualization"""
        fig = go.Figure()
        
        symbols = list(confidence_data.keys())
        confidence_values = list(confidence_data.values())
        
        # Add confidence bars
        fig.add_trace(go.Bar(
            x=symbols,
            y=confidence_values,
            marker_color=['rgba(0, 255, 0, 0.6)' if v >= threshold else 'rgba(255, 165, 0, 0.6)'
                         for v in confidence_values],
            name='ML Confidence'
        ))
        
        # Add threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Confidence Threshold ({threshold})"
        )
        
        fig.update_layout(
            title="ML Model Confidence by Symbol",
            yaxis_title="Confidence Score",
            showlegend=False,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_volume_profile(market_data: pd.DataFrame) -> go.Figure:
        """Create volume profile analysis"""
        fig = go.Figure()
        
        # Calculate volume profile
        price_bins = pd.qcut(market_data['close'], q=20)
        volume_profile = market_data.groupby(price_bins)['volume'].sum()
        
        # Create horizontal volume profile
        fig.add_trace(go.Bar(
            x=volume_profile.values,
            y=[p.left for p in volume_profile.index],
            orientation='h',
            name='Volume Profile',
            marker_color='rgba(0, 255, 0, 0.3)'
        ))
        
        # Add VWAP line
        vwap = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()
        fig.add_vline(
            x=vwap.mean(),
            line_dash="dash",
            line_color="white",
            annotation_text="VWAP"
        )
        
        fig.update_layout(
            title="Volume Profile Analysis",
            xaxis_title="Volume",
            yaxis_title="Price Levels",
            showlegend=False,
            template='plotly_dark'
        )
        
        return fig