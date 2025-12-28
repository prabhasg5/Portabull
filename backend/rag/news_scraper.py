"""
Portabull - Streaming News Scraper
Live RSS/web scraping for financial news with streaming ingestion

This module provides real-time news ingestion from:
1. RSS feeds (Economic Times, Moneycontrol, etc.)
2. Yahoo Finance news API
3. Custom web scraping for specific sources
"""

import asyncio
import aiohttp
import feedparser
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
import hashlib
import json
import re
from bs4 import BeautifulSoup


@dataclass
class NewsArticle:
    """Represents a news article"""
    article_id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str] = field(default_factory=list)  # Related stock symbols
    categories: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None  # positive, negative, neutral
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "symbols": self.symbols,
            "categories": self.categories,
            "sentiment": self.sentiment
        }
    
    def to_rag_content(self) -> str:
        """Format article for RAG ingestion"""
        return f"""
=== NEWS ARTICLE ===
Title: {self.title}
Source: {self.source}
Published: {self.published_at.strftime('%Y-%m-%d %H:%M')}
Related Stocks: {', '.join(self.symbols) if self.symbols else 'General Market News'}
Categories: {', '.join(self.categories) if self.categories else 'Uncategorized'}

Summary:
{self.summary}

Sentiment: {self.sentiment or 'Not analyzed'}
URL: {self.url}
"""


# Stock symbol patterns for Indian markets
INDIAN_STOCK_PATTERNS = {
    # Banking
    "HDFC": ["HDFC", "HDFCBANK", "hdfc bank", "hdfc limited"],
    "ICICI": ["ICICI", "ICICIBANK", "icici bank"],
    "SBI": ["SBI", "SBIN", "state bank", "state bank of india"],
    "KOTAK": ["KOTAK", "KOTAKBANK", "kotak mahindra"],
    "AXIS": ["AXIS", "AXISBANK", "axis bank"],
    
    # IT
    "TCS": ["TCS", "tata consultancy", "tata consulting"],
    "INFY": ["INFY", "INFOSYS", "infosys"],
    "WIPRO": ["WIPRO", "wipro"],
    "HCLTECH": ["HCL", "HCLTECH", "hcl technologies"],
    "TECHM": ["TECHM", "tech mahindra"],
    
    # Others
    "RELIANCE": ["RELIANCE", "reliance industries", "ril", "mukesh ambani"],
    "TATAMOTORS": ["TATA MOTORS", "TATAMOTORS", "tata motors"],
    "MARUTI": ["MARUTI", "maruti suzuki"],
    "BHARTIARTL": ["BHARTI", "BHARTIARTL", "airtel", "bharti airtel"],
    "ITC": ["ITC", "itc limited"],
    "LT": ["L&T", "LT", "larsen", "larsen & toubro"],
    "SUNPHARMA": ["SUN PHARMA", "SUNPHARMA", "sun pharmaceutical"],
    "BAJFINANCE": ["BAJAJ FINANCE", "BAJFINANCE", "bajaj finserv"],
    "TITAN": ["TITAN", "titan company"],
    "ADANI": ["ADANI", "adani enterprises", "adani green", "adani ports"]
}


class RSSFeedSource:
    """Configuration for an RSS feed source"""
    
    def __init__(
        self,
        name: str,
        url: str,
        category: str = "general",
        refresh_interval: int = 300  # 5 minutes default
    ):
        self.name = name
        self.url = url
        self.category = category
        self.refresh_interval = refresh_interval
        self.last_fetched: Optional[datetime] = None


# Default RSS feed sources for Indian financial news
DEFAULT_RSS_SOURCES = [
    RSSFeedSource(
        name="Economic Times - Markets",
        url="https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        category="markets"
    ),
    RSSFeedSource(
        name="Economic Times - Stocks",
        url="https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        category="stocks"
    ),
    RSSFeedSource(
        name="Moneycontrol - News",
        url="https://www.moneycontrol.com/rss/latestnews.xml",
        category="general"
    ),
    RSSFeedSource(
        name="Moneycontrol - Markets",
        url="https://www.moneycontrol.com/rss/marketreports.xml",
        category="markets"
    ),
    RSSFeedSource(
        name="LiveMint - Markets",
        url="https://www.livemint.com/rss/markets",
        category="markets"
    ),
    RSSFeedSource(
        name="Business Standard",
        url="https://www.business-standard.com/rss/markets-106.rss",
        category="markets"
    )
]


class StreamingNewsScraper:
    """
    Streaming news scraper that continuously fetches and processes financial news
    
    Features:
    - Multiple RSS feed sources
    - Automatic symbol extraction from news
    - Deduplication of articles
    - Sentiment analysis (basic)
    - Integration with Pathway streaming pipeline
    """
    
    def __init__(self, sources: List[RSSFeedSource] = None):
        self.sources = sources or DEFAULT_RSS_SOURCES
        self._seen_articles: Set[str] = set()  # Track seen article IDs
        self._articles: Dict[str, NewsArticle] = {}
        self._running = False
        self._subscribers: List[callable] = []
        self._max_articles = 500  # Keep last 500 articles
    
    def subscribe(self, callback: callable):
        """Subscribe to new article events"""
        self._subscribers.append(callback)
    
    async def _notify_subscribers(self, article: NewsArticle):
        """Notify all subscribers of new article"""
        for subscriber in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(article)
                else:
                    subscriber(article)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    async def start_streaming(self, interval_seconds: int = 60):
        """
        Start streaming news from all sources
        
        This runs continuously, fetching news from all sources
        and pushing new articles to the streaming pipeline
        """
        
        self._running = True
        logger.info(f"Starting news streaming with {len(self.sources)} sources")
        
        while self._running:
            try:
                # Fetch from all sources in parallel
                tasks = [self._fetch_from_source(source) for source in self.sources]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count new articles
                new_count = sum(r if isinstance(r, int) else 0 for r in results)
                if new_count > 0:
                    logger.info(f"Fetched {new_count} new articles")
                
            except Exception as e:
                logger.error(f"Error in news streaming loop: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def stop_streaming(self):
        """Stop the streaming loop"""
        self._running = False
        logger.info("News streaming stopped")
    
    async def _fetch_from_source(self, source: RSSFeedSource) -> int:
        """Fetch news from a single RSS source"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source.url, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch {source.name}: HTTP {response.status}")
                        return 0
                    
                    content = await response.text()
            
            # Parse RSS feed
            feed = feedparser.parse(content)
            new_count = 0
            
            for entry in feed.entries[:20]:  # Limit to 20 most recent
                article = self._parse_feed_entry(entry, source)
                
                if article and article.article_id not in self._seen_articles:
                    self._seen_articles.add(article.article_id)
                    self._articles[article.article_id] = article
                    
                    # Extract related symbols
                    article.symbols = self._extract_symbols(article.title + " " + article.summary)
                    
                    # Basic sentiment analysis
                    article.sentiment = self._analyze_sentiment(article.title + " " + article.summary)
                    
                    # Notify subscribers
                    await self._notify_subscribers(article)
                    
                    new_count += 1
            
            # Cleanup old articles
            self._cleanup_old_articles()
            
            source.last_fetched = datetime.now()
            return new_count
            
        except Exception as e:
            logger.error(f"Error fetching from {source.name}: {e}")
            return 0
    
    def _parse_feed_entry(self, entry: Dict, source: RSSFeedSource) -> Optional[NewsArticle]:
        """Parse a feed entry into a NewsArticle"""
        
        try:
            # Generate unique ID
            title = entry.get("title", "")
            link = entry.get("link", "")
            article_id = hashlib.md5(f"{title}{link}".encode()).hexdigest()[:16]
            
            # Parse published date
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                published_at = datetime(*published[:6])
            else:
                published_at = datetime.now()
            
            # Get summary
            summary = entry.get("summary", entry.get("description", ""))
            # Clean HTML from summary
            if summary:
                summary = BeautifulSoup(summary, "html.parser").get_text()
                summary = re.sub(r'\s+', ' ', summary).strip()[:500]
            
            return NewsArticle(
                article_id=article_id,
                title=title,
                summary=summary,
                source=source.name,
                url=link,
                published_at=published_at,
                categories=[source.category]
            )
            
        except Exception as e:
            logger.error(f"Error parsing feed entry: {e}")
            return None
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text"""
        
        text_upper = text.upper()
        text_lower = text.lower()
        found_symbols = set()
        
        for symbol, patterns in INDIAN_STOCK_PATTERNS.items():
            for pattern in patterns:
                if pattern.upper() in text_upper or pattern.lower() in text_lower:
                    found_symbols.add(symbol)
                    break
        
        return list(found_symbols)
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis based on keywords"""
        
        text_lower = text.lower()
        
        positive_words = [
            "surge", "jump", "rally", "gain", "rise", "bullish", "profit",
            "growth", "beat", "exceed", "strong", "positive", "up", "high",
            "record", "best", "boom", "soar", "optimistic", "buy"
        ]
        
        negative_words = [
            "fall", "drop", "plunge", "crash", "loss", "bearish", "decline",
            "weak", "miss", "below", "negative", "down", "low", "worst",
            "slump", "tank", "pessimistic", "sell", "concern", "risk"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count + 1:
            return "positive"
        elif negative_count > positive_count + 1:
            return "negative"
        else:
            return "neutral"
    
    def _cleanup_old_articles(self):
        """Remove old articles to prevent memory bloat"""
        
        if len(self._articles) > self._max_articles:
            # Sort by published date and keep most recent
            sorted_articles = sorted(
                self._articles.items(),
                key=lambda x: x[1].published_at,
                reverse=True
            )
            
            self._articles = dict(sorted_articles[:self._max_articles])
            self._seen_articles = set(self._articles.keys())
    
    def get_recent_articles(
        self,
        limit: int = 20,
        symbol: str = None,
        sentiment: str = None
    ) -> List[NewsArticle]:
        """Get recent articles with optional filtering"""
        
        articles = list(self._articles.values())
        
        # Filter by symbol
        if symbol:
            articles = [a for a in articles if symbol.upper() in a.symbols]
        
        # Filter by sentiment
        if sentiment:
            articles = [a for a in articles if a.sentiment == sentiment]
        
        # Sort by date and limit
        articles.sort(key=lambda x: x.published_at, reverse=True)
        return articles[:limit]
    
    def get_articles_for_symbols(self, symbols: List[str]) -> Dict[str, List[NewsArticle]]:
        """Get articles grouped by symbol"""
        
        result = {s: [] for s in symbols}
        
        for article in self._articles.values():
            for symbol in symbols:
                if symbol.upper() in article.symbols:
                    result[symbol].append(article)
        
        # Sort each list by date
        for symbol in result:
            result[symbol].sort(key=lambda x: x.published_at, reverse=True)
            result[symbol] = result[symbol][:5]  # Limit to 5 per symbol
        
        return result
    
    async def fetch_once(self) -> List[NewsArticle]:
        """Fetch news once from all sources (non-streaming)"""
        
        tasks = [self._fetch_from_source(source) for source in self.sources]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return self.get_recent_articles(limit=50)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics"""
        
        articles = list(self._articles.values())
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for a in articles:
            if a.sentiment:
                sentiment_counts[a.sentiment] = sentiment_counts.get(a.sentiment, 0) + 1
        
        source_counts = {}
        for a in articles:
            source_counts[a.source] = source_counts.get(a.source, 0) + 1
        
        return {
            "total_articles": len(articles),
            "sources_count": len(self.sources),
            "sentiment_distribution": sentiment_counts,
            "articles_by_source": source_counts,
            "streaming": self._running,
            "last_fetch": max((s.last_fetched for s in self.sources if s.last_fetched), default=None)
        }


# Singleton
_news_scraper: Optional[StreamingNewsScraper] = None


def get_news_scraper() -> StreamingNewsScraper:
    """Get or create news scraper singleton"""
    global _news_scraper
    if _news_scraper is None:
        _news_scraper = StreamingNewsScraper()
    return _news_scraper


async def integrate_with_pipeline(pipeline):
    """
    Integrate news scraper with Pathway streaming pipeline
    
    This creates a bridge between the news scraper and the RAG pipeline
    """
    
    scraper = get_news_scraper()
    
    async def on_new_article(article: NewsArticle):
        """Callback when new article is fetched"""
        await pipeline.add_document(
            doc_id=f"news_{article.article_id}",
            content=article.to_rag_content(),
            doc_type="news",
            symbol=article.symbols[0] if article.symbols else "",
            source=article.source,
            metadata={
                "title": article.title,
                "sentiment": article.sentiment,
                "symbols": article.symbols,
                "url": article.url
            }
        )
    
    scraper.subscribe(on_new_article)
    logger.info("News scraper integrated with streaming pipeline")
