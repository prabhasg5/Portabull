"""
Portabull - Agent Tools System
Explicit tools for agents to use during reasoning

This module provides a structured toolset for agents:
1. Price Lookup - Get real-time and historical prices
2. News Fetch - Get latest news for stocks
3. Technical Analysis - Calculate technical indicators
4. Fundamental Analysis - Get company financials
5. Market Sentiment - Analyze market sentiment
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
import json


class ToolType(Enum):
    """Types of tools available to agents"""
    PRICE_LOOKUP = "price_lookup"
    NEWS_FETCH = "news_fetch"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    MARKET_SENTIMENT = "market_sentiment"
    SECTOR_ANALYSIS = "sector_analysis"
    PORTFOLIO_METRICS = "portfolio_metrics"
    HYPOTHESIS_CHECK = "hypothesis_check"


@dataclass
class ToolCall:
    """Represents a tool call made by an agent"""
    tool_type: ToolType
    parameters: Dict[str, Any]
    called_at: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None
    execution_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_type.value,
            "parameters": self.parameters,
            "called_at": self.called_at.isoformat(),
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms
        }


@dataclass
class Tool:
    """Definition of a tool"""
    name: str
    tool_type: ToolType
    description: str
    parameters_schema: Dict[str, Any]
    handler: Callable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.tool_type.value,
            "description": self.description,
            "parameters": self.parameters_schema
        }


class AgentToolkit:
    """
    Toolkit providing tools for agent reasoning
    
    Each tool has:
    - Name and description
    - Parameter schema
    - Handler function
    - Result formatting
    """
    
    def __init__(self):
        self._tools: Dict[ToolType, Tool] = {}
        self._call_history: List[ToolCall] = []
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools"""
        
        # 1. Price Lookup Tool
        self.register_tool(Tool(
            name="lookup_price",
            tool_type=ToolType.PRICE_LOOKUP,
            description="Get current or historical price for a stock symbol",
            parameters_schema={
                "symbol": {"type": "string", "required": True, "description": "Stock symbol (e.g., RELIANCE, TCS)"},
                "exchange": {"type": "string", "required": False, "default": "NSE"},
                "period": {"type": "string", "required": False, "default": "1d", "options": ["1d", "5d", "1mo", "3mo"]}
            },
            handler=self._price_lookup_handler
        ))
        
        # 2. News Fetch Tool
        self.register_tool(Tool(
            name="fetch_news",
            tool_type=ToolType.NEWS_FETCH,
            description="Fetch latest news articles for a stock or sector",
            parameters_schema={
                "symbol": {"type": "string", "required": False, "description": "Stock symbol"},
                "sector": {"type": "string", "required": False, "description": "Sector name"},
                "limit": {"type": "integer", "required": False, "default": 5}
            },
            handler=self._news_fetch_handler
        ))
        
        # 3. Technical Analysis Tool
        self.register_tool(Tool(
            name="technical_analysis",
            tool_type=ToolType.TECHNICAL_ANALYSIS,
            description="Calculate technical indicators (RSI, MACD, Moving Averages, etc.)",
            parameters_schema={
                "symbol": {"type": "string", "required": True, "description": "Stock symbol"},
                "indicators": {"type": "array", "required": False, "default": ["RSI", "MACD", "SMA_50", "SMA_200"]}
            },
            handler=self._technical_analysis_handler
        ))
        
        # 4. Fundamental Analysis Tool
        self.register_tool(Tool(
            name="fundamental_analysis",
            tool_type=ToolType.FUNDAMENTAL_ANALYSIS,
            description="Get fundamental data (P/E, P/B, Market Cap, Revenue, etc.)",
            parameters_schema={
                "symbol": {"type": "string", "required": True, "description": "Stock symbol"},
                "metrics": {"type": "array", "required": False, "default": ["pe_ratio", "pb_ratio", "market_cap", "revenue"]}
            },
            handler=self._fundamental_analysis_handler
        ))
        
        # 5. Market Sentiment Tool
        self.register_tool(Tool(
            name="market_sentiment",
            tool_type=ToolType.MARKET_SENTIMENT,
            description="Analyze market sentiment for a stock or overall market",
            parameters_schema={
                "symbol": {"type": "string", "required": False, "description": "Stock symbol (optional for market-wide)"},
                "source": {"type": "string", "required": False, "default": "all", "options": ["news", "social", "analyst", "all"]}
            },
            handler=self._market_sentiment_handler
        ))
        
        # 6. Sector Analysis Tool
        self.register_tool(Tool(
            name="sector_analysis",
            tool_type=ToolType.SECTOR_ANALYSIS,
            description="Analyze sector performance and trends",
            parameters_schema={
                "sector": {"type": "string", "required": True, "description": "Sector name (IT, Banking, Pharma, etc.)"},
                "period": {"type": "string", "required": False, "default": "1mo"}
            },
            handler=self._sector_analysis_handler
        ))
        
        # 7. Portfolio Metrics Tool
        self.register_tool(Tool(
            name="portfolio_metrics",
            tool_type=ToolType.PORTFOLIO_METRICS,
            description="Calculate portfolio metrics (Sharpe ratio, Beta, Volatility, etc.)",
            parameters_schema={
                "holdings": {"type": "array", "required": True, "description": "List of holdings"},
                "metrics": {"type": "array", "required": False, "default": ["returns", "volatility", "sharpe", "beta"]}
            },
            handler=self._portfolio_metrics_handler
        ))
        
        # 8. Hypothesis Check Tool
        self.register_tool(Tool(
            name="check_hypothesis",
            tool_type=ToolType.HYPOTHESIS_CHECK,
            description="Validate or check an investment hypothesis with current data",
            parameters_schema={
                "hypothesis": {"type": "string", "required": True, "description": "The hypothesis to check"},
                "symbols": {"type": "array", "required": False, "description": "Related symbols"}
            },
            handler=self._hypothesis_check_handler
        ))
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self._tools[tool.tool_type] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        return [tool.to_dict() for tool in self._tools.values()]
    
    async def call_tool(
        self,
        tool_type: ToolType,
        parameters: Dict[str, Any]
    ) -> ToolCall:
        """
        Call a tool and return the result
        
        This is the main interface for agents to use tools
        """
        
        if tool_type not in self._tools:
            return ToolCall(
                tool_type=tool_type,
                parameters=parameters,
                success=False,
                error=f"Tool {tool_type.value} not found"
            )
        
        tool = self._tools[tool_type]
        start_time = datetime.now()
        
        try:
            result = await tool.handler(parameters)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            tool_call = ToolCall(
                tool_type=tool_type,
                parameters=parameters,
                result=result,
                success=True,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            tool_call = ToolCall(
                tool_type=tool_type,
                parameters=parameters,
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
            logger.error(f"Tool {tool_type.value} failed: {e}")
        
        self._call_history.append(tool_call)
        return tool_call
    
    # ============================================
    # Tool Handlers
    # ============================================
    
    async def _price_lookup_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for price lookup tool"""
        import yfinance as yf
        
        symbol = params.get("symbol", "").upper()
        exchange = params.get("exchange", "NSE")
        period = params.get("period", "1d")
        
        # Format symbol for yfinance
        if exchange in ["NSE", "BSE"]:
            yf_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
        else:
            yf_symbol = symbol
        
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for {symbol}", "symbol": symbol}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "current_price": round(current_price, 2),
            "previous_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "high": round(hist['High'].iloc[-1], 2),
            "low": round(hist['Low'].iloc[-1], 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _news_fetch_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for news fetch tool"""
        import yfinance as yf
        
        symbol = params.get("symbol", "")
        limit = params.get("limit", 5)
        
        if symbol:
            yf_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(yf_symbol)
            news = ticker.news[:limit] if hasattr(ticker, 'news') else []
            
            formatted_news = []
            for item in news:
                formatted_news.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat()
                })
            
            return {
                "symbol": symbol,
                "news_count": len(formatted_news),
                "articles": formatted_news
            }
        
        return {"error": "Symbol required for news fetch", "articles": []}
    
    async def _technical_analysis_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for technical analysis tool"""
        import yfinance as yf
        import numpy as np
        
        symbol = params.get("symbol", "").upper()
        indicators = params.get("indicators", ["RSI", "MACD", "SMA_50", "SMA_200"])
        
        yf_symbol = f"{symbol}.NS"
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="6mo")
        
        if hist.empty:
            return {"error": f"No data for {symbol}", "indicators": {}}
        
        close_prices = hist['Close'].values
        result = {"symbol": symbol, "indicators": {}}
        
        # Calculate requested indicators
        if "RSI" in indicators:
            result["indicators"]["RSI"] = self._calculate_rsi(close_prices)
        
        if "SMA_50" in indicators:
            result["indicators"]["SMA_50"] = round(np.mean(close_prices[-50:]), 2) if len(close_prices) >= 50 else None
        
        if "SMA_200" in indicators:
            result["indicators"]["SMA_200"] = round(np.mean(close_prices[-200:]), 2) if len(close_prices) >= 200 else None
        
        if "MACD" in indicators:
            macd = self._calculate_macd(close_prices)
            result["indicators"]["MACD"] = macd
        
        # Add signals
        result["signals"] = self._generate_signals(result["indicators"], close_prices[-1])
        
        return result
    
    def _calculate_rsi(self, prices, period=14) -> float:
        """Calculate RSI"""
        import numpy as np
        
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def _calculate_macd(self, prices) -> Dict[str, float]:
        """Calculate MACD"""
        import numpy as np
        
        if len(prices) < 26:
            return {"line": 0, "signal": 0, "histogram": 0}
        
        # EMA calculations
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        signal_line = self._ema([macd_line], 9) if macd_line else 0
        histogram = macd_line - signal_line
        
        return {
            "line": round(macd_line, 2),
            "signal": round(signal_line, 2),
            "histogram": round(histogram, 2)
        }
    
    def _ema(self, prices, period) -> float:
        """Calculate EMA"""
        import numpy as np
        
        if len(prices) < period:
            return float(np.mean(prices))
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _generate_signals(self, indicators: Dict, current_price: float) -> List[str]:
        """Generate trading signals from indicators"""
        signals = []
        
        rsi = indicators.get("RSI")
        if rsi:
            if rsi > 70:
                signals.append("RSI indicates OVERBOUGHT conditions")
            elif rsi < 30:
                signals.append("RSI indicates OVERSOLD conditions")
            else:
                signals.append("RSI is NEUTRAL")
        
        sma_50 = indicators.get("SMA_50")
        sma_200 = indicators.get("SMA_200")
        
        if sma_50 and sma_200:
            if sma_50 > sma_200:
                signals.append("GOLDEN CROSS: SMA50 above SMA200 (Bullish)")
            else:
                signals.append("DEATH CROSS: SMA50 below SMA200 (Bearish)")
        
        if sma_50 and current_price:
            if current_price > sma_50:
                signals.append("Price above 50-day MA (Bullish)")
            else:
                signals.append("Price below 50-day MA (Bearish)")
        
        macd = indicators.get("MACD", {})
        if macd.get("histogram", 0) > 0:
            signals.append("MACD histogram positive (Bullish momentum)")
        elif macd.get("histogram", 0) < 0:
            signals.append("MACD histogram negative (Bearish momentum)")
        
        return signals
    
    async def _fundamental_analysis_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for fundamental analysis tool"""
        import yfinance as yf
        
        symbol = params.get("symbol", "").upper()
        
        yf_symbol = f"{symbol}.NS"
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        
        return {
            "symbol": symbol,
            "company_name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "fundamentals": {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "dividend_yield": info.get("dividendYield"),
                "eps": info.get("trailingEps"),
                "revenue": info.get("totalRevenue"),
                "profit_margin": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow")
            },
            "recommendation": info.get("recommendationKey", "N/A"),
            "target_price": info.get("targetMeanPrice")
        }
    
    async def _market_sentiment_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for market sentiment tool"""
        import yfinance as yf
        
        symbol = params.get("symbol", "")
        
        # Market indices for overall sentiment
        indices = {
            "NIFTY50": "^NSEI",
            "SENSEX": "^BSESN",
            "BANKNIFTY": "^NSEBANK"
        }
        
        sentiment_data = {
            "overall_sentiment": "neutral",
            "sentiment_score": 50,
            "market_indices": {},
            "analysis": []
        }
        
        # Get index data
        bullish_count = 0
        bearish_count = 0
        
        for name, yf_symbol in indices.items():
            try:
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                    sentiment_data["market_indices"][name] = {
                        "current": round(hist['Close'].iloc[-1], 2),
                        "5d_change": round(change, 2)
                    }
                    if change > 1:
                        bullish_count += 1
                    elif change < -1:
                        bearish_count += 1
            except:
                pass
        
        # Calculate overall sentiment
        if bullish_count > bearish_count:
            sentiment_data["overall_sentiment"] = "bullish"
            sentiment_data["sentiment_score"] = 60 + (bullish_count * 10)
            sentiment_data["analysis"].append("Market indices show positive momentum")
        elif bearish_count > bullish_count:
            sentiment_data["overall_sentiment"] = "bearish"
            sentiment_data["sentiment_score"] = 40 - (bearish_count * 10)
            sentiment_data["analysis"].append("Market indices show negative momentum")
        else:
            sentiment_data["analysis"].append("Market showing mixed signals")
        
        # Stock-specific sentiment if symbol provided
        if symbol:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info
                
                rec = info.get("recommendationKey", "").lower()
                if rec in ["buy", "strong_buy"]:
                    sentiment_data["stock_sentiment"] = "bullish"
                    sentiment_data["analysis"].append(f"Analyst recommendation: {rec.upper()}")
                elif rec in ["sell", "strong_sell"]:
                    sentiment_data["stock_sentiment"] = "bearish"
                    sentiment_data["analysis"].append(f"Analyst recommendation: {rec.upper()}")
                else:
                    sentiment_data["stock_sentiment"] = "neutral"
            except:
                pass
        
        return sentiment_data
    
    async def _sector_analysis_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for sector analysis tool"""
        
        sector = params.get("sector", "").upper()
        
        # Sector ETFs/proxies for NSE
        sector_proxies = {
            "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
            "BANKING": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN"],
            "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON"],
            "AUTO": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
            "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
            "ENERGY": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIGREEN"]
        }
        
        stocks = sector_proxies.get(sector, [])
        if not stocks:
            return {"error": f"Unknown sector: {sector}", "available_sectors": list(sector_proxies.keys())}
        
        import yfinance as yf
        
        sector_data = {
            "sector": sector,
            "stocks_analyzed": len(stocks),
            "performance": {},
            "top_performer": None,
            "worst_performer": None,
            "sector_trend": "neutral"
        }
        
        performances = []
        
        for symbol in stocks[:5]:  # Limit to 5 stocks
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period="1mo")
                if not hist.empty:
                    change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                    sector_data["performance"][symbol] = round(change, 2)
                    performances.append((symbol, change))
            except:
                pass
        
        if performances:
            performances.sort(key=lambda x: x[1], reverse=True)
            sector_data["top_performer"] = {"symbol": performances[0][0], "return": round(performances[0][1], 2)}
            sector_data["worst_performer"] = {"symbol": performances[-1][0], "return": round(performances[-1][1], 2)}
            
            avg_return = sum(p[1] for p in performances) / len(performances)
            sector_data["average_return"] = round(avg_return, 2)
            
            if avg_return > 3:
                sector_data["sector_trend"] = "bullish"
            elif avg_return < -3:
                sector_data["sector_trend"] = "bearish"
        
        return sector_data
    
    async def _portfolio_metrics_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for portfolio metrics tool"""
        
        holdings = params.get("holdings", [])
        
        if not holdings:
            return {"error": "Holdings required for portfolio metrics"}
        
        total_value = sum(h.get("current_value", h.get("quantity", 0) * h.get("last_price", 0)) for h in holdings)
        total_invested = sum(h.get("quantity", 0) * h.get("average_price", 0) for h in holdings)
        total_pnl = sum(h.get("pnl", 0) for h in holdings)
        
        return {
            "total_holdings": len(holdings),
            "total_value": round(total_value, 2),
            "total_invested": round(total_invested, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((total_pnl / total_invested) * 100, 2) if total_invested > 0 else 0,
            "metrics": {
                "diversification_score": min(len(holdings) / 15 * 100, 100),  # 15 stocks = 100%
                "concentration": self._calculate_concentration(holdings, total_value),
                "top_holdings": self._get_top_holdings(holdings, total_value, 3)
            }
        }
    
    def _calculate_concentration(self, holdings: List[Dict], total_value: float) -> Dict[str, Any]:
        """Calculate portfolio concentration metrics"""
        if total_value == 0:
            return {}
        
        weights = []
        for h in holdings:
            value = h.get("current_value", h.get("quantity", 0) * h.get("last_price", 0))
            weights.append(value / total_value)
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum(w ** 2 for w in weights) * 10000
        
        return {
            "hhi": round(hhi, 2),
            "top_5_weight": round(sum(sorted(weights, reverse=True)[:5]) * 100, 2),
            "interpretation": "Highly concentrated" if hhi > 2500 else "Moderately concentrated" if hhi > 1500 else "Well diversified"
        }
    
    def _get_top_holdings(self, holdings: List[Dict], total_value: float, n: int) -> List[Dict]:
        """Get top N holdings by value"""
        if total_value == 0:
            return []
        
        sorted_holdings = sorted(
            holdings,
            key=lambda h: h.get("current_value", h.get("quantity", 0) * h.get("last_price", 0)),
            reverse=True
        )
        
        return [
            {
                "symbol": h.get("tradingsymbol", h.get("symbol", "Unknown")),
                "value": h.get("current_value", h.get("quantity", 0) * h.get("last_price", 0)),
                "weight": round((h.get("current_value", h.get("quantity", 0) * h.get("last_price", 0)) / total_value) * 100, 2)
            }
            for h in sorted_holdings[:n]
        ]
    
    async def _hypothesis_check_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for hypothesis checking tool"""
        
        hypothesis = params.get("hypothesis", "")
        symbols = params.get("symbols", [])
        
        # Use other tools to validate hypothesis
        validation_results = {
            "hypothesis": hypothesis,
            "validation_status": "pending",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "confidence": 0.5
        }
        
        # If symbols provided, gather data
        for symbol in symbols[:3]:  # Limit checks
            try:
                # Get technical signals
                tech_result = await self._technical_analysis_handler({"symbol": symbol})
                if tech_result.get("signals"):
                    for signal in tech_result["signals"]:
                        if "Bullish" in signal:
                            validation_results["supporting_evidence"].append(f"{symbol}: {signal}")
                        elif "Bearish" in signal:
                            validation_results["contradicting_evidence"].append(f"{symbol}: {signal}")
            except:
                pass
        
        # Calculate confidence based on evidence
        support = len(validation_results["supporting_evidence"])
        contradict = len(validation_results["contradicting_evidence"])
        
        if support + contradict > 0:
            validation_results["confidence"] = round(support / (support + contradict), 2)
        
        if validation_results["confidence"] > 0.6:
            validation_results["validation_status"] = "supported"
        elif validation_results["confidence"] < 0.4:
            validation_results["validation_status"] = "contradicted"
        else:
            validation_results["validation_status"] = "inconclusive"
        
        return validation_results
    
    def get_call_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tool call history"""
        return [call.to_dict() for call in self._call_history[-limit:]]


# Singleton
_toolkit: Optional[AgentToolkit] = None


def get_agent_toolkit() -> AgentToolkit:
    """Get or create agent toolkit singleton"""
    global _toolkit
    if _toolkit is None:
        _toolkit = AgentToolkit()
    return _toolkit
