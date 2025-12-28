"""
Portabull - Paper Trading Broker Client
Allows users to build and manage a simulated portfolio with real-time quotes
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio

import yfinance as yf
from loguru import logger


@dataclass
class PaperHolding:
    """Represents a paper trading holding"""
    symbol: str
    exchange: str  # NSE, BSE, NASDAQ, etc.
    quantity: int
    average_price: float
    bought_at: str  # ISO timestamp
    
    # Computed fields (updated on quote fetch)
    last_price: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    value: float = 0.0
    day_change: float = 0.0
    day_change_percent: float = 0.0


@dataclass
class PaperTransaction:
    """Represents a buy/sell transaction"""
    id: str
    symbol: str
    exchange: str
    action: str  # BUY or SELL
    quantity: int
    price: float
    timestamp: str
    notes: str = ""


@dataclass
class PaperPortfolio:
    """Complete paper trading portfolio"""
    user_id: str
    cash_balance: float = 1000000.0  # Start with 10 lakh
    holdings: Dict[str, PaperHolding] = field(default_factory=dict)
    transactions: List[PaperTransaction] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class PaperBrokerClient:
    """
    Paper Trading Broker Client
    
    Features:
    - In-memory portfolio management
    - Buy/sell operations with validation
    - Real-time quotes via yfinance
    - Persistence to JSON file
    - NSE/BSE stock support (append .NS or .BO)
    """
    
    def __init__(
        self,
        user_id: str = "paper_user",
        data_dir: str = "data/paper_trading",
        initial_cash: float = 1000000.0
    ):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = self.data_dir / f"{user_id}_portfolio.json"
        
        # Load or create portfolio
        self.portfolio = self._load_portfolio(initial_cash)
        
        # Quote cache (symbol -> (price, timestamp))
        self._quote_cache: Dict[str, tuple] = {}
        self._cache_ttl = 60  # seconds
        
        logger.info(f"PaperBrokerClient initialized for user: {user_id}")
        logger.info(f"Cash balance: ₹{self.portfolio.cash_balance:,.2f}")
        logger.info(f"Holdings: {len(self.portfolio.holdings)} stocks")
    
    def _load_portfolio(self, initial_cash: float) -> PaperPortfolio:
        """Load portfolio from file or create new"""
        if self.portfolio_file.exists():
            try:
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct portfolio
                portfolio = PaperPortfolio(
                    user_id=data.get('user_id', self.user_id),
                    cash_balance=data.get('cash_balance', initial_cash),
                    created_at=data.get('created_at', datetime.now().isoformat()),
                    updated_at=data.get('updated_at', datetime.now().isoformat())
                )
                
                # Reconstruct holdings
                for symbol, h in data.get('holdings', {}).items():
                    portfolio.holdings[symbol] = PaperHolding(**h)
                
                # Reconstruct transactions
                for t in data.get('transactions', []):
                    portfolio.transactions.append(PaperTransaction(**t))
                
                logger.info(f"Loaded existing portfolio from {self.portfolio_file}")
                return portfolio
                
            except Exception as e:
                logger.error(f"Failed to load portfolio: {e}")
        
        # Create new portfolio
        logger.info("Creating new paper portfolio")
        return PaperPortfolio(user_id=self.user_id, cash_balance=initial_cash)
    
    def _save_portfolio(self):
        """Save portfolio to file"""
        self.portfolio.updated_at = datetime.now().isoformat()
        
        data = {
            'user_id': self.portfolio.user_id,
            'cash_balance': self.portfolio.cash_balance,
            'created_at': self.portfolio.created_at,
            'updated_at': self.portfolio.updated_at,
            'holdings': {s: asdict(h) for s, h in self.portfolio.holdings.items()},
            'transactions': [asdict(t) for t in self.portfolio.transactions]
        }
        
        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Portfolio saved to {self.portfolio_file}")
    
    def _get_yf_symbol(self, symbol: str, exchange: str) -> str:
        """Convert symbol to yfinance format"""
        symbol = symbol.upper().strip()
        exchange = exchange.upper().strip()
        
        if exchange in ('NSE', 'NS'):
            return f"{symbol}.NS"
        elif exchange in ('BSE', 'BO'):
            return f"{symbol}.BO"
        elif exchange in ('NASDAQ', 'NYSE', 'US'):
            return symbol  # US stocks don't need suffix
        else:
            return f"{symbol}.NS"  # Default to NSE
    
    async def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        yf_symbol = self._get_yf_symbol(symbol, exchange)
        cache_key = yf_symbol
        
        # Check cache
        if cache_key in self._quote_cache:
            cached_price, cached_time = self._quote_cache[cache_key]
            if (datetime.now().timestamp() - cached_time) < self._cache_ttl:
                return cached_price
        
        try:
            # Run yfinance in thread pool (it's blocking)
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(yf_symbol))
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            quote = {
                'symbol': symbol,
                'exchange': exchange,
                'yf_symbol': yf_symbol,
                'last_price': info.get('currentPrice') or info.get('regularMarketPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'company_name': info.get('longName') or info.get('shortName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }
            
            # Calculate day change
            if quote['previous_close'] and quote['last_price']:
                quote['day_change'] = quote['last_price'] - quote['previous_close']
                quote['day_change_percent'] = (quote['day_change'] / quote['previous_close']) * 100
            else:
                quote['day_change'] = 0
                quote['day_change_percent'] = 0
            
            # Cache it
            self._quote_cache[cache_key] = (quote, datetime.now().timestamp())
            
            return quote
            
        except Exception as e:
            logger.error(f"Failed to get quote for {yf_symbol}: {e}")
            return {
                'symbol': symbol,
                'exchange': exchange,
                'last_price': 0,
                'error': str(e)
            }
    
    async def search_stocks(self, query: str, exchange: str = "NSE", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for stocks by name, symbol, or sector with fuzzy matching
        
        Returns matching stocks even with partial queries
        """
        from data.stock_database import search_stocks, get_popular_stocks
        
        # If empty query, return popular stocks
        if not query or len(query.strip()) < 1:
            return get_popular_stocks(exchange, limit)
        
        # Search using fuzzy matching
        results = search_stocks(query, exchange, limit)
        
        # Get live quotes for top results (limit to avoid too many API calls)
        for result in results[:5]:
            try:
                quote = await self.get_quote(result['symbol'], exchange)
                if quote.get('last_price', 0) > 0:
                    result['last_price'] = quote.get('last_price', 0)
                    result['day_change_percent'] = quote.get('day_change_percent', 0)
            except:
                pass
        
        return results
    
    async def buy(
        self,
        symbol: str,
        quantity: int,
        exchange: str = "NSE",
        price: Optional[float] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Buy stocks (add to portfolio)"""
        symbol = symbol.upper().strip()
        
        # Get current price if not provided
        if price is None:
            quote = await self.get_quote(symbol, exchange)
            price = quote.get('last_price', 0)
            if price <= 0:
                return {'success': False, 'error': f"Could not get price for {symbol}"}
        
        total_cost = price * quantity
        
        # Check cash balance
        if total_cost > self.portfolio.cash_balance:
            return {
                'success': False,
                'error': f"Insufficient funds. Need ₹{total_cost:,.2f}, have ₹{self.portfolio.cash_balance:,.2f}"
            }
        
        # Deduct cash
        self.portfolio.cash_balance -= total_cost
        
        # Add or update holding
        key = f"{symbol}:{exchange}"
        if key in self.portfolio.holdings:
            # Average up/down
            existing = self.portfolio.holdings[key]
            total_qty = existing.quantity + quantity
            avg_price = (
                (existing.average_price * existing.quantity) + (price * quantity)
            ) / total_qty
            existing.quantity = total_qty
            existing.average_price = avg_price
        else:
            self.portfolio.holdings[key] = PaperHolding(
                symbol=symbol,
                exchange=exchange,
                quantity=quantity,
                average_price=price,
                bought_at=datetime.now().isoformat(),
                last_price=price,
                value=total_cost
            )
        
        # Record transaction
        tx = PaperTransaction(
            id=f"TX-{len(self.portfolio.transactions)+1:04d}",
            symbol=symbol,
            exchange=exchange,
            action="BUY",
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            notes=notes
        )
        self.portfolio.transactions.append(tx)
        
        # Save
        self._save_portfolio()
        
        logger.info(f"BUY: {quantity} x {symbol} @ ₹{price:,.2f} = ₹{total_cost:,.2f}")
        
        return {
            'success': True,
            'transaction': asdict(tx),
            'holding': asdict(self.portfolio.holdings[key]),
            'cash_balance': self.portfolio.cash_balance
        }
    
    async def sell(
        self,
        symbol: str,
        quantity: int,
        exchange: str = "NSE",
        price: Optional[float] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Sell stocks (remove from portfolio)"""
        symbol = symbol.upper().strip()
        key = f"{symbol}:{exchange}"
        
        # Check if we have the holding
        if key not in self.portfolio.holdings:
            return {'success': False, 'error': f"No holding found for {symbol}"}
        
        holding = self.portfolio.holdings[key]
        
        # Check quantity
        if quantity > holding.quantity:
            return {
                'success': False,
                'error': f"Cannot sell {quantity}, only have {holding.quantity}"
            }
        
        # Get current price if not provided
        if price is None:
            quote = await self.get_quote(symbol, exchange)
            price = quote.get('last_price', 0)
            if price <= 0:
                return {'success': False, 'error': f"Could not get price for {symbol}"}
        
        total_value = price * quantity
        
        # Add cash
        self.portfolio.cash_balance += total_value
        
        # Update or remove holding
        if quantity == holding.quantity:
            del self.portfolio.holdings[key]
        else:
            holding.quantity -= quantity
        
        # Record transaction
        tx = PaperTransaction(
            id=f"TX-{len(self.portfolio.transactions)+1:04d}",
            symbol=symbol,
            exchange=exchange,
            action="SELL",
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            notes=notes
        )
        self.portfolio.transactions.append(tx)
        
        # Calculate P&L for this sale
        cost_basis = holding.average_price * quantity
        realized_pnl = total_value - cost_basis
        
        # Save
        self._save_portfolio()
        
        logger.info(f"SELL: {quantity} x {symbol} @ ₹{price:,.2f} = ₹{total_value:,.2f} (P&L: ₹{realized_pnl:,.2f})")
        
        return {
            'success': True,
            'transaction': asdict(tx),
            'realized_pnl': realized_pnl,
            'cash_balance': self.portfolio.cash_balance
        }
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get all holdings with updated prices"""
        holdings = []
        
        for key, holding in self.portfolio.holdings.items():
            # Fetch latest quote
            quote = await self.get_quote(holding.symbol, holding.exchange)
            
            # Update holding with current price
            holding.last_price = quote.get('last_price', holding.average_price)
            holding.value = holding.last_price * holding.quantity
            holding.pnl = (holding.last_price - holding.average_price) * holding.quantity
            if holding.average_price > 0:
                holding.pnl_percent = ((holding.last_price - holding.average_price) / holding.average_price) * 100
            holding.day_change = quote.get('day_change', 0)
            holding.day_change_percent = quote.get('day_change_percent', 0)
            
            holdings.append({
                **asdict(holding),
                'company_name': quote.get('company_name', holding.symbol),
                'sector': quote.get('sector', 'Unknown')
            })
        
        # Save updated prices
        self._save_portfolio()
        
        return holdings
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get complete portfolio summary"""
        holdings = await self.get_holdings()
        
        total_invested = sum(h['average_price'] * h['quantity'] for h in holdings)
        total_current = sum(h['value'] for h in holdings)
        total_pnl = total_current - total_invested
        total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        day_pnl = sum(h['day_change'] * h['quantity'] for h in holdings)
        
        return {
            'user_id': self.portfolio.user_id,
            'cash_balance': self.portfolio.cash_balance,
            'total_invested': total_invested,
            'total_current_value': total_current,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'day_pnl': day_pnl,
            'holdings_count': len(holdings),
            'holdings': holdings,
            'portfolio_value': self.portfolio.cash_balance + total_current,
            'transactions_count': len(self.portfolio.transactions),
            'created_at': self.portfolio.created_at,
            'updated_at': self.portfolio.updated_at
        }
    
    async def get_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent transactions"""
        txns = self.portfolio.transactions[-limit:]
        return [asdict(t) for t in reversed(txns)]  # Most recent first
    
    async def reset_portfolio(self, initial_cash: float = 1000000.0) -> Dict[str, Any]:
        """Reset portfolio to initial state"""
        self.portfolio = PaperPortfolio(
            user_id=self.user_id,
            cash_balance=initial_cash
        )
        self._save_portfolio()
        logger.info(f"Portfolio reset with ₹{initial_cash:,.2f}")
        return {'success': True, 'cash_balance': initial_cash}
    
    # Compatibility methods (match ZerodhaClient interface)
    def get_login_url(self) -> str:
        """Return empty string for paper trading (no OAuth needed)"""
        return ""
    
    async def authenticate(self, request_token: str) -> 'UserCredentials':
        """Mock authentication for paper trading"""
        from dataclasses import dataclass
        
        @dataclass
        class UserCredentials:
            user_id: str
            access_token: str
            refresh_token: str = ""
            
        return UserCredentials(
            user_id=self.user_id,
            access_token="paper_trading_token_" + self.user_id,
            refresh_token=""
        )
    
    async def get_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        return {
            'user_id': self.portfolio.user_id,
            'user_name': 'Paper Trader',
            'email': 'paper@portabull.app',
            'broker': 'Paper Trading',
            'user_type': 'individual'
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions (same as holdings for paper trading)"""
        return await self.get_holdings()


# Factory function
def create_paper_broker(user_id: str = "default") -> PaperBrokerClient:
    """Create paper broker client"""
    return PaperBrokerClient(user_id=user_id)
