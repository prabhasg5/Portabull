"""
Portabull - Zerodha Kite Connect Integration
Read-only portfolio access for analysis

This module handles:
- OAuth2 authentication with Zerodha
- Portfolio data retrieval (holdings, positions)
- Real-time market data subscriptions
- Order and trade history (read-only)
"""

from kiteconnect import KiteConnect, KiteTicker
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
from loguru import logger
import aiohttp
from functools import wraps


@dataclass
class ZerodhaCredentials:
    """User's Zerodha credentials and tokens"""
    user_id: str
    access_token: str
    refresh_token: Optional[str] = None
    token_expiry: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if the access token is still valid"""
        if not self.token_expiry:
            return True
        return datetime.now() < self.token_expiry


@dataclass
class Holding:
    """Represents a stock holding in the portfolio"""
    tradingsymbol: str
    exchange: str
    isin: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_percent: float
    value: float
    day_change: float
    day_change_percent: float
    
    @classmethod
    def from_kite_response(cls, data: Dict[str, Any]) -> 'Holding':
        """Create Holding from Kite API response"""
        avg_price = data.get('average_price', 0)
        last_price = data.get('last_price', 0)
        quantity = data.get('quantity', 0)
        
        value = last_price * quantity
        pnl = (last_price - avg_price) * quantity
        pnl_percent = ((last_price - avg_price) / avg_price * 100) if avg_price else 0
        
        return cls(
            tradingsymbol=data.get('tradingsymbol', ''),
            exchange=data.get('exchange', 'NSE'),
            isin=data.get('isin', ''),
            quantity=quantity,
            average_price=avg_price,
            last_price=last_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            value=value,
            day_change=data.get('day_change', 0),
            day_change_percent=data.get('day_change_percentage', 0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tradingsymbol': self.tradingsymbol,
            'exchange': self.exchange,
            'isin': self.isin,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'last_price': self.last_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'value': self.value,
            'day_change': self.day_change,
            'day_change_percent': self.day_change_percent
        }


@dataclass
class Position:
    """Represents a trading position"""
    tradingsymbol: str
    exchange: str
    product: str  # CNC, MIS, NRML
    quantity: int
    buy_price: float
    sell_price: float
    last_price: float
    pnl: float
    multiplier: float
    
    @classmethod
    def from_kite_response(cls, data: Dict[str, Any]) -> 'Position':
        """Create Position from Kite API response"""
        return cls(
            tradingsymbol=data.get('tradingsymbol', ''),
            exchange=data.get('exchange', 'NSE'),
            product=data.get('product', 'CNC'),
            quantity=data.get('quantity', 0),
            buy_price=data.get('average_price', 0),
            sell_price=data.get('sell_price', 0),
            last_price=data.get('last_price', 0),
            pnl=data.get('pnl', 0),
            multiplier=data.get('multiplier', 1)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tradingsymbol': self.tradingsymbol,
            'exchange': self.exchange,
            'product': self.product,
            'quantity': self.quantity,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'last_price': self.last_price,
            'pnl': self.pnl,
            'multiplier': self.multiplier
        }


class ZerodhaClient:
    """
    Zerodha Kite Connect Client
    
    Provides read-only access to:
    - User profile
    - Holdings and positions
    - Market quotes
    - Historical data
    - Trade history
    
    Note: This client does NOT support placing orders.
    It's designed for portfolio analysis only.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        redirect_url: str = "http://localhost:8000/auth/callback"
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.redirect_url = redirect_url
        
        # Initialize Kite Connect
        self.kite = KiteConnect(api_key=api_key)
        
        # Ticker for real-time data
        self.ticker: Optional[KiteTicker] = None
        
        # Callbacks for real-time updates
        self._tick_callbacks: List[Callable] = []
        
        # Cache
        self._holdings_cache: List[Holding] = []
        self._positions_cache: List[Position] = []
        self._last_refresh: Optional[datetime] = None
        
        logger.info("ZerodhaClient initialized")
    
    def get_login_url(self) -> str:
        """Get the Zerodha login URL for OAuth"""
        return self.kite.login_url()
    
    async def authenticate(self, request_token: str) -> ZerodhaCredentials:
        """
        Complete OAuth flow with request token
        
        Args:
            request_token: Token received from Zerodha callback
            
        Returns:
            ZerodhaCredentials with access token
        """
        try:
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.api_secret
            )
            
            access_token = data["access_token"]
            user_id = data["user_id"]
            
            # Set access token for future requests
            self.kite.set_access_token(access_token)
            
            credentials = ZerodhaCredentials(
                user_id=user_id,
                access_token=access_token,
                token_expiry=datetime.now() + timedelta(hours=8)  # Kite tokens expire at 6 AM next day
            )
            
            logger.info(f"Successfully authenticated user: {user_id}")
            return credentials
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def set_access_token(self, access_token: str):
        """Set access token for authenticated requests"""
        self.kite.set_access_token(access_token)
    
    async def get_profile(self) -> Dict[str, Any]:
        """Get user profile information"""
        try:
            profile = self.kite.profile()
            return {
                "user_id": profile.get("user_id"),
                "user_name": profile.get("user_name"),
                "email": profile.get("email"),
                "broker": profile.get("broker", "ZERODHA"),
                "exchanges": profile.get("exchanges", [])
            }
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            raise
    
    async def get_holdings(self, force_refresh: bool = False) -> List[Holding]:
        """
        Get user's stock holdings
        
        Args:
            force_refresh: Force refresh from API even if cached
            
        Returns:
            List of Holding objects
        """
        try:
            # Check cache
            if not force_refresh and self._holdings_cache:
                if self._last_refresh and (datetime.now() - self._last_refresh).seconds < 60:
                    return self._holdings_cache
            
            holdings_data = self.kite.holdings()
            
            holdings = [
                Holding.from_kite_response(h)
                for h in holdings_data
            ]
            
            self._holdings_cache = holdings
            self._last_refresh = datetime.now()
            
            logger.debug(f"Retrieved {len(holdings)} holdings")
            return holdings
            
        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            raise
    
    async def get_positions(self) -> Dict[str, List[Position]]:
        """
        Get user's trading positions
        
        Returns:
            Dict with 'net' and 'day' positions
        """
        try:
            positions_data = self.kite.positions()
            
            net_positions = [
                Position.from_kite_response(p)
                for p in positions_data.get('net', [])
            ]
            
            day_positions = [
                Position.from_kite_response(p)
                for p in positions_data.get('day', [])
            ]
            
            self._positions_cache = net_positions
            
            return {
                'net': net_positions,
                'day': day_positions
            }
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get complete portfolio summary
        
        Returns comprehensive portfolio data for analysis
        """
        holdings = await self.get_holdings()
        positions = await self.get_positions()
        
        # Calculate totals
        total_invested = sum(h.average_price * h.quantity for h in holdings)
        total_current = sum(h.value for h in holdings)
        total_pnl = sum(h.pnl for h in holdings)
        
        # Day's change
        day_pnl = sum(h.day_change * h.quantity for h in holdings)
        
        return {
            "holdings": [h.to_dict() for h in holdings],
            "positions": {
                "net": [p.to_dict() for p in positions['net']],
                "day": [p.to_dict() for p in positions['day']]
            },
            "summary": {
                "total_holdings": len(holdings),
                "total_invested": total_invested,
                "total_current_value": total_current,
                "total_pnl": total_pnl,
                "total_pnl_percent": (total_pnl / total_invested * 100) if total_invested else 0,
                "day_pnl": day_pnl,
                "day_pnl_percent": (day_pnl / total_current * 100) if total_current else 0
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time quotes for symbols
        
        Args:
            symbols: List of symbols in format "EXCHANGE:SYMBOL"
            
        Returns:
            Dict of quotes keyed by symbol
        """
        try:
            quotes = self.kite.quote(symbols)
            
            result = {}
            for symbol, data in quotes.items():
                result[symbol] = {
                    "last_price": data.get("last_price"),
                    "open": data.get("ohlc", {}).get("open"),
                    "high": data.get("ohlc", {}).get("high"),
                    "low": data.get("ohlc", {}).get("low"),
                    "close": data.get("ohlc", {}).get("close"),
                    "change": data.get("net_change"),
                    "change_percent": (
                        (data.get("last_price", 0) - data.get("ohlc", {}).get("close", 0)) /
                        data.get("ohlc", {}).get("close", 1) * 100
                    ),
                    "volume": data.get("volume"),
                    "timestamp": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            raise
    
    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Get historical OHLC data
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, NFO, etc.)
            from_date: Start date
            to_date: End date
            interval: Candle interval (minute, day, etc.)
            
        Returns:
            List of OHLC candles
        """
        try:
            # Get instrument token
            instruments = self.kite.instruments(exchange)
            instrument = next(
                (i for i in instruments if i['tradingsymbol'] == symbol),
                None
            )
            
            if not instrument:
                raise ValueError(f"Instrument not found: {symbol}")
            
            instrument_token = instrument['instrument_token']
            
            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            return [
                {
                    "date": candle["date"].isoformat(),
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle["volume"]
                }
                for candle in data
            ]
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            raise
    
    async def get_trades(self) -> List[Dict[str, Any]]:
        """Get today's executed trades"""
        try:
            trades = self.kite.trades()
            return [
                {
                    "trade_id": t.get("trade_id"),
                    "order_id": t.get("order_id"),
                    "tradingsymbol": t.get("tradingsymbol"),
                    "exchange": t.get("exchange"),
                    "transaction_type": t.get("transaction_type"),
                    "quantity": t.get("quantity"),
                    "price": t.get("price"),
                    "product": t.get("product"),
                    "fill_timestamp": t.get("fill_timestamp")
                }
                for t in trades
            ]
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            raise
    
    def start_ticker(
        self,
        symbols: List[str],
        on_tick: Callable[[Dict], None]
    ):
        """
        Start real-time data ticker
        
        Args:
            symbols: List of symbols to subscribe
            on_tick: Callback for tick updates
        """
        if not self.ticker:
            self.ticker = KiteTicker(
                self.api_key,
                self.kite.access_token
            )
        
        def on_ticks(ws, ticks):
            for tick in ticks:
                on_tick(tick)
        
        def on_connect(ws, response):
            # Get instrument tokens for symbols
            ws.subscribe(symbols)
            ws.set_mode(ws.MODE_FULL, symbols)
            logger.info(f"Ticker connected, subscribed to {len(symbols)} symbols")
        
        def on_close(ws, code, reason):
            logger.warning(f"Ticker closed: {code} - {reason}")
        
        self.ticker.on_ticks = on_ticks
        self.ticker.on_connect = on_connect
        self.ticker.on_close = on_close
        
        # Start in a separate thread
        self.ticker.connect(threaded=True)
    
    def stop_ticker(self):
        """Stop the real-time ticker"""
        if self.ticker:
            self.ticker.close()
            self.ticker = None
            logger.info("Ticker stopped")


class MockZerodhaClient:
    """
    Mock Zerodha client for development and testing
    
    Simulates Zerodha API responses with realistic mock data
    """
    
    def __init__(self):
        self._mock_holdings = [
            {
                "tradingsymbol": "RELIANCE",
                "exchange": "NSE",
                "isin": "INE002A01018",
                "quantity": 10,
                "average_price": 2450.00,
                "last_price": 2520.50,
                "day_change": 15.30,
                "day_change_percentage": 0.61
            },
            {
                "tradingsymbol": "TCS",
                "exchange": "NSE",
                "isin": "INE467B01029",
                "quantity": 5,
                "average_price": 3800.00,
                "last_price": 3950.25,
                "day_change": -25.50,
                "day_change_percentage": -0.64
            },
            {
                "tradingsymbol": "HDFCBANK",
                "exchange": "NSE",
                "isin": "INE040A01034",
                "quantity": 20,
                "average_price": 1650.00,
                "last_price": 1720.80,
                "day_change": 12.40,
                "day_change_percentage": 0.73
            },
            {
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "isin": "INE009A01021",
                "quantity": 15,
                "average_price": 1480.00,
                "last_price": 1510.60,
                "day_change": 8.90,
                "day_change_percentage": 0.59
            },
            {
                "tradingsymbol": "ICICIBANK",
                "exchange": "NSE",
                "isin": "INE090A01021",
                "quantity": 25,
                "average_price": 980.00,
                "last_price": 1025.40,
                "day_change": -5.20,
                "day_change_percentage": -0.50
            }
        ]
        
        logger.info("MockZerodhaClient initialized with sample portfolio")
    
    def get_login_url(self) -> str:
        return "http://localhost:8000/auth/mock-login"
    
    async def authenticate(self, request_token: str) -> ZerodhaCredentials:
        return ZerodhaCredentials(
            user_id="MOCK123",
            access_token="mock_access_token_12345",
            token_expiry=datetime.now() + timedelta(hours=8)
        )
    
    async def get_profile(self) -> Dict[str, Any]:
        return {
            "user_id": "MOCK123",
            "user_name": "Demo User",
            "email": "demo@portabull.ai",
            "broker": "ZERODHA",
            "exchanges": ["NSE", "BSE", "NFO"]
        }
    
    async def get_holdings(self, force_refresh: bool = False) -> List[Holding]:
        return [Holding.from_kite_response(h) for h in self._mock_holdings]
    
    async def get_positions(self) -> Dict[str, List[Position]]:
        return {"net": [], "day": []}
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        holdings = await self.get_holdings()
        
        total_invested = sum(h.average_price * h.quantity for h in holdings)
        total_current = sum(h.value for h in holdings)
        total_pnl = sum(h.pnl for h in holdings)
        day_pnl = sum(h.day_change * h.quantity for h in holdings)
        
        return {
            "holdings": [h.to_dict() for h in holdings],
            "positions": {"net": [], "day": []},
            "summary": {
                "total_holdings": len(holdings),
                "total_invested": total_invested,
                "total_current_value": total_current,
                "total_pnl": total_pnl,
                "total_pnl_percent": (total_pnl / total_invested * 100) if total_invested else 0,
                "day_pnl": day_pnl,
                "day_pnl_percent": (day_pnl / total_current * 100) if total_current else 0
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        # Return mock quotes
        return {
            symbol: {
                "last_price": 1000 + hash(symbol) % 1000,
                "change_percent": (hash(symbol) % 10) - 5
            }
            for symbol in symbols
        }


def create_broker_client(
    provider: str = "paper",
    api_key: str = None,
    api_secret: str = None,
    user_id: str = "default"
):
    """
    Factory function to create broker client
    
    Args:
        provider: Broker provider ("paper", "zerodha", "mock")
        api_key: API key (for Zerodha)
        api_secret: API secret (for Zerodha)
        user_id: User ID for paper trading
        
    Returns:
        Broker client instance
    """
    provider = provider.lower().strip()
    
    if provider == "paper":
        from integrations.paper_broker import PaperBrokerClient
        logger.info("Using PaperBrokerClient")
        return PaperBrokerClient(user_id=user_id)
    
    elif provider == "zerodha":
        if not api_key or not api_secret:
            logger.warning("Zerodha credentials missing, falling back to paper trading")
            from integrations.paper_broker import PaperBrokerClient
            return PaperBrokerClient(user_id=user_id)
        logger.info("Using ZerodhaClient")
        return ZerodhaClient(api_key=api_key, api_secret=api_secret)
    
    else:  # mock
        logger.info("Using MockZerodhaClient")
        return MockZerodhaClient()


# Backward compatibility alias
def create_zerodha_client(
    api_key: str = None,
    api_secret: str = None,
    use_mock: bool = False
) -> ZerodhaClient:
    """
    Factory function to create Zerodha client (legacy)
    
    Args:
        api_key: Kite API key
        api_secret: Kite API secret
        use_mock: Use mock client for development
        
    Returns:
        ZerodhaClient or MockZerodhaClient
    """
    if use_mock or not api_key:
        logger.info("Using MockZerodhaClient")
        return MockZerodhaClient()
    
    return ZerodhaClient(
        api_key=api_key,
        api_secret=api_secret
    )

