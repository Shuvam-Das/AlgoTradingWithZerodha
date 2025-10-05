"""News sentiment analysis module for fundamental analysis"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from textblob import TextBlob
import logging
from dataclasses import dataclass

logger = logging.getLogger('atwz.sentiment')

@dataclass
class NewsArticle:
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    sentiment_subjectivity: float

@dataclass
class SentimentAnalysis:
    symbol: str
    overall_sentiment: float
    confidence: float
    article_count: int
    timestamp: datetime
    top_positive: List[NewsArticle]
    top_negative: List[NewsArticle]
    sources: List[str]

class NewsSentimentAnalyzer:
    """Analyze news sentiment for stocks"""
    
    def __init__(self):
        self.cache: Dict[str, SentimentAnalysis] = {}
        self.cache_duration = timedelta(hours=1)
    
    def analyze_text(self, text: str) -> tuple[float, float]:
        """Analyze sentiment of text using TextBlob"""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    
    async def fetch_news(self, symbol: str, days: int = 7) -> List[NewsArticle]:
        """Fetch news articles for a symbol"""
        articles = []
        
        # Sources configuration
        sources = [
            {
                'name': 'Yahoo Finance',
                'url': f'https://finance.yahoo.com/quote/{symbol}/news',
                'selector': 'h3.Mb\\(5px\\)'
            },
            # Add more sources as needed
        ]
        
        for source in sources:
            try:
                response = requests.get(source['url'])
                soup = BeautifulSoup(response.text, 'html.parser')
                headlines = soup.select(source['selector'])
                
                for headline in headlines:
                    title = headline.text.strip()
                    url = headline.find('a')['href'] if headline.find('a') else ''
                    
                    # Fetch full article content if URL available
                    content = ''
                    if url:
                        try:
                            article_response = requests.get(url)
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            content = ' '.join([p.text for p in article_soup.find_all('p')])
                        except Exception as e:
                            logger.warning(f"Error fetching article content: {e}")
                    
                    # Analyze sentiment
                    sentiment_score, subjectivity = self.analyze_text(title + ' ' + content)
                    
                    articles.append(NewsArticle(
                        title=title,
                        content=content[:500],  # Truncate content
                        source=source['name'],
                        url=url,
                        published_at=datetime.now(),  # Should parse from article
                        sentiment_score=sentiment_score,
                        sentiment_subjectivity=subjectivity
                    ))
                
            except Exception as e:
                logger.error(f"Error fetching news from {source['name']}: {e}")
        
        return articles
    
    def analyze_sentiment(self, symbol: str, days: int = 7) -> SentimentAnalysis:
        """Analyze news sentiment for a symbol"""
        # Check cache
        if (
            symbol in self.cache and 
            datetime.now() - self.cache[symbol].timestamp < self.cache_duration
        ):
            return self.cache[symbol]
        
        # Fetch and analyze news
        articles = self.fetch_news(symbol, days)
        
        if not articles:
            logger.warning(f"No news articles found for {symbol}")
            return SentimentAnalysis(
                symbol=symbol,
                overall_sentiment=0,
                confidence=0,
                article_count=0,
                timestamp=datetime.now(),
                top_positive=[],
                top_negative=[],
                sources=[]
            )
        
        # Calculate overall sentiment
        sentiments = [a.sentiment_score for a in articles]
        subjectivities = [a.sentiment_subjectivity for a in articles]
        
        overall_sentiment = np.mean(sentiments)
        confidence = 1 - np.mean(subjectivities)  # Lower subjectivity = higher confidence
        
        # Sort articles by sentiment
        sorted_articles = sorted(articles, key=lambda x: x.sentiment_score)
        top_negative = sorted_articles[:3]  # 3 most negative
        top_positive = sorted_articles[-3:]  # 3 most positive
        
        # Create analysis result
        analysis = SentimentAnalysis(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            article_count=len(articles),
            timestamp=datetime.now(),
            top_positive=top_positive,
            top_negative=top_negative,
            sources=list(set(a.source for a in articles))
        )
        
        # Cache result
        self.cache[symbol] = analysis
        
        return analysis