"""
Portabull - Market Data Fetcher for Dynamic RAG
Fetches real-time market data, news, and fundamentals for portfolio stocks
"""

import asyncio
import yfinance as yf
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import json


class MarketDataFetcher:
    """
    Fetches market data for Dynamic RAG
    
    Data sources:
    - yfinance: Stock prices, fundamentals, news
    - Company info, sector data, analyst recommendations
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minutes
        
    def _get_yf_symbol(self, symbol: str, exchange: str = "NSE") -> str:
        """Convert symbol to yfinance format"""
        symbol = symbol.upper().strip()
        exchange = exchange.upper().strip()
        
        if exchange in ('NSE', 'NS'):
            return f"{symbol}.NS"
        elif exchange in ('BSE', 'BO'):
            return f"{symbol}.BO"
        elif exchange in ('NASDAQ', 'NYSE', 'US'):
            return symbol
        else:
            return f"{symbol}.NS"
    
    async def fetch_stock_data(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Fetch comprehensive stock data for RAG context
        """
        yf_symbol = self._get_yf_symbol(symbol, exchange)
        
        # Check cache
        cache_key = f"{yf_symbol}_data"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now().timestamp() - cached['timestamp'] < self._cache_ttl:
                return cached['data']
        
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(yf_symbol))
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            # Get historical data for context
            hist = await loop.run_in_executor(
                None, 
                lambda: ticker.history(period="1mo")
            )
            
            data = {
                "symbol": symbol,
                "exchange": exchange,
                "company_name": info.get("longName") or info.get("shortName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "previous_close": info.get("previousClose", 0),
                "day_high": info.get("dayHigh", 0),
                "day_low": info.get("dayLow", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "eps": info.get("trailingEps", 0),
                "beta": info.get("beta", 0),
                "volume": info.get("volume", 0),
                "avg_volume": info.get("averageVolume", 0),
                "recommendation": info.get("recommendationKey", "N/A"),
                "target_price": info.get("targetMeanPrice", 0),
                "analyst_count": info.get("numberOfAnalystOpinions", 0),
                "business_summary": info.get("longBusinessSummary", "")[:500] if info.get("longBusinessSummary") else "",
                # Price history stats
                "month_high": hist["High"].max() if not hist.empty else 0,
                "month_low": hist["Low"].min() if not hist.empty else 0,
                "month_avg": hist["Close"].mean() if not hist.empty else 0,
                "month_volatility": hist["Close"].std() if not hist.empty else 0,
                "fetched_at": datetime.now().isoformat()
            }
            
            # Cache it
            self._cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now().timestamp()
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {yf_symbol}: {e}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "error": str(e)
            }
    
    async def fetch_stock_news(self, symbol: str, exchange: str = "NSE") -> List[Dict[str, Any]]:
        """
        Fetch recent news for a stock
        """
        yf_symbol = self._get_yf_symbol(symbol, exchange)
        
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(yf_symbol))
            news = await loop.run_in_executor(None, lambda: ticker.news)
            
            if not news:
                return []
            
            formatted_news = []
            for item in news[:5]:  # Limit to 5 news items
                formatted_news.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat() if item.get("providerPublishTime") else "",
                    "type": item.get("type", "news"),
                    "symbol": symbol
                })
            
            return formatted_news
            
        except Exception as e:
            logger.error(f"Error fetching news for {yf_symbol}: {e}")
            return []
    
    async def fetch_sector_overview(self, sector: str) -> Dict[str, Any]:
        """
        Fetch sector overview data
        """
        # Define sector ETFs/indices for context
        sector_etfs = {
            "Technology": "XLK",
            "Financial Services": "XLF",
            "Healthcare": "XLV",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Communication Services": "XLC"
        }
        
        etf_symbol = sector_etfs.get(sector, "SPY")  # Default to S&P 500
        
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(etf_symbol))
            hist = await loop.run_in_executor(None, lambda: ticker.history(period="1mo"))
            
            if hist.empty:
                return {"sector": sector, "error": "No data available"}
            
            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            month_change = ((end_price - start_price) / start_price) * 100
            
            return {
                "sector": sector,
                "etf_symbol": etf_symbol,
                "current_price": end_price,
                "month_change_percent": month_change,
                "month_high": hist["High"].max(),
                "month_low": hist["Low"].min(),
                "trend": "bullish" if month_change > 2 else "bearish" if month_change < -2 else "neutral"
            }
            
        except Exception as e:
            logger.error(f"Error fetching sector data for {sector}: {e}")
            return {"sector": sector, "error": str(e)}
    
    async def fetch_market_overview(self) -> Dict[str, Any]:
        """
        Fetch overall market overview
        """
        indices = {
            "NIFTY50": "^NSEI",
            "SENSEX": "^BSESN",
            "BANKNIFTY": "^NSEBANK",
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC"
        }
        
        market_data = {}
        
        for name, symbol in indices.items():
            try:
                loop = asyncio.get_event_loop()
                ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
                info = await loop.run_in_executor(None, lambda: ticker.info)
                
                market_data[name] = {
                    "current": info.get("regularMarketPrice", 0),
                    "previous_close": info.get("previousClose", 0),
                    "change": info.get("regularMarketPrice", 0) - info.get("previousClose", 0),
                    "change_percent": ((info.get("regularMarketPrice", 0) - info.get("previousClose", 0)) / info.get("previousClose", 1)) * 100 if info.get("previousClose") else 0
                }
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
                market_data[name] = {"error": str(e)}
        
        return {
            "indices": market_data,
            "fetched_at": datetime.now().isoformat()
        }
    
    def format_for_rag(self, data: Dict[str, Any], data_type: str) -> str:
        """
        Format data as text for RAG ingestion
        """
        if data_type == "stock":
            return self._format_stock_for_rag(data)
        elif data_type == "news":
            return self._format_news_for_rag(data)
        elif data_type == "market":
            return self._format_market_for_rag(data)
        elif data_type == "sector":
            return self._format_sector_for_rag(data)
        else:
            return json.dumps(data, indent=2)
    
    def _format_stock_for_rag(self, data: Dict) -> str:
        """Format stock data as readable text for RAG"""
        if "error" in data:
            return f"Error fetching data for {data.get('symbol', 'Unknown')}: {data['error']}"
        
        lines = [
            f"=== Stock Analysis: {data.get('company_name', data.get('symbol', 'N/A'))} ({data.get('symbol', '')}) ===",
            f"Sector: {data.get('sector', 'Unknown')} | Industry: {data.get('industry', 'Unknown')}",
            "",
            "📊 Current Price Information:",
            f"  • Current Price: ₹{data.get('current_price', 0):,.2f}",
            f"  • Previous Close: ₹{data.get('previous_close', 0):,.2f}",
            f"  • Day Range: ₹{data.get('day_low', 0):,.2f} - ₹{data.get('day_high', 0):,.2f}",
            f"  • 52-Week Range: ₹{data.get('52_week_low', 0):,.2f} - ₹{data.get('52_week_high', 0):,.2f}",
            "",
            "📈 Valuation Metrics:",
            f"  • Market Cap: ₹{data.get('market_cap', 0):,.0f}",
            f"  • P/E Ratio: {data.get('pe_ratio', 0):.2f}" if data.get('pe_ratio') else "  • P/E Ratio: N/A",
            f"  • Forward P/E: {data.get('forward_pe', 0):.2f}" if data.get('forward_pe') else "  • Forward P/E: N/A",
            f"  • P/B Ratio: {data.get('pb_ratio', 0):.2f}" if data.get('pb_ratio') else "  • P/B Ratio: N/A",
            f"  • EPS: ₹{data.get('eps', 0):.2f}" if data.get('eps') else "  • EPS: N/A",
            f"  • Dividend Yield: {data.get('dividend_yield', 0)*100:.2f}%" if data.get('dividend_yield') else "  • Dividend Yield: N/A",
            "",
            "📉 Risk & Volatility:",
            f"  • Beta: {data.get('beta', 0):.2f}" if data.get('beta') else "  • Beta: N/A",
            f"  • 1-Month Volatility: {data.get('month_volatility', 0):.2f}",
            f"  • 1-Month Range: ₹{data.get('month_low', 0):,.2f} - ₹{data.get('month_high', 0):,.2f}",
            "",
            "🎯 Analyst Opinion:",
            f"  • Recommendation: {data.get('recommendation', 'N/A').upper()}",
            f"  • Target Price: ₹{data.get('target_price', 0):,.2f}" if data.get('target_price') else "  • Target Price: N/A",
            f"  • Number of Analysts: {data.get('analyst_count', 0)}",
            "",
            "📝 Business Summary:",
            f"  {data.get('business_summary', 'No summary available.')[:300]}...",
            "",
            f"Data as of: {data.get('fetched_at', 'Unknown')}"
        ]
        
        return "\n".join(lines)
    
    def _format_news_for_rag(self, news_list: List[Dict]) -> str:
        """Format news as readable text for RAG"""
        if not news_list:
            return "No recent news available."
        
        lines = ["=== Recent News ===", ""]
        
        for item in news_list:
            lines.append(f"📰 {item.get('title', 'No title')}")
            lines.append(f"   Publisher: {item.get('publisher', 'Unknown')} | {item.get('published', 'Unknown date')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_market_for_rag(self, data: Dict) -> str:
        """Format market overview as readable text for RAG"""
        lines = [
            "=== Market Overview ===",
            f"As of: {data.get('fetched_at', 'Unknown')}",
            ""
        ]
        
        indices = data.get("indices", {})
        for name, info in indices.items():
            if "error" not in info:
                change_symbol = "🟢" if info.get("change_percent", 0) >= 0 else "🔴"
                lines.append(
                    f"{change_symbol} {name}: {info.get('current', 0):,.2f} "
                    f"({info.get('change_percent', 0):+.2f}%)"
                )
        
        return "\n".join(lines)
    
    def _format_sector_for_rag(self, data: Dict) -> str:
        """Format sector data as readable text for RAG"""
        if "error" in data:
            return f"Sector data unavailable for {data.get('sector', 'Unknown')}"
        
        trend_emoji = "📈" if data.get("trend") == "bullish" else "📉" if data.get("trend") == "bearish" else "➡️"
        
        return f"""=== Sector Analysis: {data.get('sector', 'Unknown')} ===
{trend_emoji} Trend: {data.get('trend', 'Unknown').upper()}
Monthly Change: {data.get('month_change_percent', 0):+.2f}%
Month Range: ${data.get('month_low', 0):,.2f} - ${data.get('month_high', 0):,.2f}
"""


# Singleton instance
_fetcher_instance = None

def get_market_data_fetcher() -> MarketDataFetcher:
    """Get or create the market data fetcher instance"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = MarketDataFetcher()
    return _fetcher_instance
