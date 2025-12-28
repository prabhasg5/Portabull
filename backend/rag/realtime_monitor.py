"""
Portabull - Real-time Market Monitor
Background refresh, caching, WebSocket streaming, and anomaly detection
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import statistics


@dataclass
class CachedData:
    """Cached market data with timestamp"""
    data: Dict[str, Any]
    fetched_at: datetime
    ttl_seconds: int = 60  # Default 1 minute TTL
    
    def is_valid(self) -> bool:
        """Check if cache is still valid"""
        return datetime.now() - self.fetched_at < timedelta(seconds=self.ttl_seconds)


@dataclass
class PricePoint:
    """Single price point for tracking"""
    symbol: str
    price: float
    timestamp: datetime
    volume: Optional[int] = None
    change_percent: Optional[float] = None


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly"""
    alert_id: str
    alert_type: str  # price_spike, price_drop, volume_surge, unusual_volatility
    severity: str  # low, medium, high, critical
    symbol: str
    title: str
    description: str
    current_value: float
    previous_value: float
    change_percent: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "symbol": self.symbol,
            "title": self.title,
            "description": self.description,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "change_percent": self.change_percent,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


class MarketDataCache:
    """
    Intelligent caching layer for market data
    
    Features:
    - TTL-based expiration
    - Per-symbol caching
    - Automatic cleanup
    """
    
    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl
        self._cache: Dict[str, CachedData] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if valid"""
        async with self._lock:
            if key in self._cache:
                cached = self._cache[key]
                if cached.is_valid():
                    logger.debug(f"Cache HIT for {key}")
                    return cached.data
                else:
                    # Expired, remove it
                    del self._cache[key]
                    logger.debug(f"Cache EXPIRED for {key}")
            return None
    
    async def set(self, key: str, data: Dict[str, Any], ttl: int = None):
        """Set cached data"""
        async with self._lock:
            self._cache[key] = CachedData(
                data=data,
                fetched_at=datetime.now(),
                ttl_seconds=ttl or self.default_ttl
            )
            logger.debug(f"Cache SET for {key}")
    
    async def invalidate(self, key: str):
        """Invalidate specific cache entry"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    async def clear(self):
        """Clear all cache"""
        async with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        valid_count = sum(1 for c in self._cache.values() if c.is_valid())
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_count,
            "expired_entries": len(self._cache) - valid_count
        }


class AnomalyDetector:
    """
    Detects market anomalies based on price and volume changes
    
    Thresholds:
    - Price spike/drop: > 3% change in short period
    - Volume surge: > 200% of average volume
    - Unusual volatility: Standard deviation > 2x normal
    """
    
    # Alert thresholds
    PRICE_CHANGE_LOW = 2.0      # 2% change = low alert
    PRICE_CHANGE_MEDIUM = 5.0   # 5% change = medium alert
    PRICE_CHANGE_HIGH = 8.0     # 8% change = high alert
    PRICE_CHANGE_CRITICAL = 10.0  # 10% change = critical alert
    
    VOLUME_SURGE_THRESHOLD = 2.0  # 200% of average = alert
    VOLATILITY_THRESHOLD = 2.0    # 2x normal std dev = alert
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self._price_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}
        self._alerts: List[AnomalyAlert] = []
        self._alert_counter = 0
        self._callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """Add callback for when anomaly is detected"""
        self._callbacks.append(callback)
    
    def record_price(self, symbol: str, price: float, volume: int = None, change_percent: float = None):
        """Record a price point and check for anomalies"""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.history_size)
            self._volume_history[symbol] = deque(maxlen=self.history_size)
        
        point = PricePoint(
            symbol=symbol,
            price=price,
            timestamp=datetime.now(),
            volume=volume,
            change_percent=change_percent
        )
        
        # Check for anomalies before adding to history
        anomalies = self._detect_anomalies(symbol, point)
        
        # Add to history
        self._price_history[symbol].append(point)
        if volume:
            self._volume_history[symbol].append(volume)
        
        # Process anomalies
        for anomaly in anomalies:
            self._alerts.append(anomaly)
            logger.warning(f"🚨 ANOMALY DETECTED: {anomaly.title}")
            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    asyncio.create_task(callback(anomaly))
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        return anomalies
    
    def _detect_anomalies(self, symbol: str, current: PricePoint) -> List[AnomalyAlert]:
        """Detect anomalies for a symbol"""
        anomalies = []
        
        history = self._price_history.get(symbol, deque())
        if not history:
            return anomalies
        
        # Get previous price
        prev_price = history[-1].price
        
        # Calculate price change
        if prev_price > 0:
            price_change = ((current.price - prev_price) / prev_price) * 100
        else:
            price_change = 0
        
        # Check for price anomalies
        abs_change = abs(price_change)
        if abs_change >= self.PRICE_CHANGE_LOW:
            severity = self._get_severity(abs_change)
            alert_type = "price_spike" if price_change > 0 else "price_drop"
            
            self._alert_counter += 1
            anomaly = AnomalyAlert(
                alert_id=f"ALERT_{self._alert_counter:05d}",
                alert_type=alert_type,
                severity=severity,
                symbol=symbol,
                title=f"{'📈' if price_change > 0 else '📉'} {symbol}: {price_change:+.2f}% price {'surge' if price_change > 0 else 'drop'}",
                description=f"{symbol} has moved {price_change:+.2f}% from ₹{prev_price:.2f} to ₹{current.price:.2f}",
                current_value=current.price,
                previous_value=prev_price,
                change_percent=price_change,
                threshold=self.PRICE_CHANGE_LOW
            )
            anomalies.append(anomaly)
        
        # Check for volume anomalies
        if current.volume and len(self._volume_history.get(symbol, [])) >= 5:
            avg_volume = statistics.mean(self._volume_history[symbol])
            if avg_volume > 0:
                volume_ratio = current.volume / avg_volume
                if volume_ratio >= self.VOLUME_SURGE_THRESHOLD:
                    self._alert_counter += 1
                    anomaly = AnomalyAlert(
                        alert_id=f"ALERT_{self._alert_counter:05d}",
                        alert_type="volume_surge",
                        severity="medium" if volume_ratio < 3 else "high",
                        symbol=symbol,
                        title=f"📊 {symbol}: {volume_ratio:.1f}x volume surge",
                        description=f"{symbol} volume is {volume_ratio:.1f}x the average ({current.volume:,} vs avg {avg_volume:,.0f})",
                        current_value=current.volume,
                        previous_value=avg_volume,
                        change_percent=(volume_ratio - 1) * 100,
                        threshold=self.VOLUME_SURGE_THRESHOLD * 100
                    )
                    anomalies.append(anomaly)
        
        # Check for unusual volatility (need at least 10 data points)
        if len(history) >= 10:
            recent_prices = [p.price for p in list(history)[-10:]]
            try:
                std_dev = statistics.stdev(recent_prices)
                mean_price = statistics.mean(recent_prices)
                if mean_price > 0:
                    volatility = (std_dev / mean_price) * 100
                    # Compare to historical volatility
                    if len(history) >= 20:
                        older_prices = [p.price for p in list(history)[-20:-10]]
                        old_std = statistics.stdev(older_prices) if len(older_prices) > 1 else std_dev
                        old_mean = statistics.mean(older_prices) if older_prices else mean_price
                        old_volatility = (old_std / old_mean) * 100 if old_mean > 0 else volatility
                        
                        if old_volatility > 0 and volatility / old_volatility >= self.VOLATILITY_THRESHOLD:
                            self._alert_counter += 1
                            anomaly = AnomalyAlert(
                                alert_id=f"ALERT_{self._alert_counter:05d}",
                                alert_type="unusual_volatility",
                                severity="medium",
                                symbol=symbol,
                                title=f"⚡ {symbol}: Unusual volatility detected",
                                description=f"{symbol} volatility has increased {volatility/old_volatility:.1f}x (current: {volatility:.2f}%, historical: {old_volatility:.2f}%)",
                                current_value=volatility,
                                previous_value=old_volatility,
                                change_percent=(volatility / old_volatility - 1) * 100,
                                threshold=self.VOLATILITY_THRESHOLD * 100
                            )
                            anomalies.append(anomaly)
            except statistics.StatisticsError:
                pass
        
        return anomalies
    
    def _get_severity(self, abs_change: float) -> str:
        """Get severity level based on price change"""
        if abs_change >= self.PRICE_CHANGE_CRITICAL:
            return "critical"
        elif abs_change >= self.PRICE_CHANGE_HIGH:
            return "high"
        elif abs_change >= self.PRICE_CHANGE_MEDIUM:
            return "medium"
        return "low"
    
    def get_alerts(self, unacknowledged_only: bool = False, limit: int = 50) -> List[AnomalyAlert]:
        """Get recent alerts"""
        alerts = self._alerts
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def clear_alerts(self):
        """Clear all alerts"""
        self._alerts.clear()


class RealtimeMarketMonitor:
    """
    Real-time market monitoring service
    
    Features:
    - Background data refresh at configurable intervals
    - Intelligent caching
    - Anomaly detection
    - WebSocket broadcasting
    """
    
    def __init__(
        self,
        refresh_interval: int = 60,  # seconds
        cache_ttl: int = 60,         # seconds
        rag_engine = None
    ):
        self.refresh_interval = refresh_interval
        self.cache = MarketDataCache(default_ttl=cache_ttl)
        self.anomaly_detector = AnomalyDetector()
        self.rag_engine = rag_engine
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._watched_symbols: Set[str] = set()
        self._websocket_clients: Set = set()
        self._last_refresh: Optional[datetime] = None
        self._refresh_count = 0
        
        # Add anomaly callback for WebSocket broadcast
        self.anomaly_detector.add_callback(self._broadcast_anomaly)
    
    def add_symbols(self, symbols: List[str]):
        """Add symbols to watch list"""
        self._watched_symbols.update(symbols)
        logger.info(f"Now watching {len(self._watched_symbols)} symbols: {self._watched_symbols}")
    
    def remove_symbols(self, symbols: List[str]):
        """Remove symbols from watch list"""
        self._watched_symbols.difference_update(symbols)
    
    def register_websocket(self, websocket):
        """Register a WebSocket client for updates"""
        self._websocket_clients.add(websocket)
        logger.info(f"WebSocket client registered. Total: {len(self._websocket_clients)}")
    
    def unregister_websocket(self, websocket):
        """Unregister a WebSocket client"""
        self._websocket_clients.discard(websocket)
        logger.info(f"WebSocket client unregistered. Total: {len(self._websocket_clients)}")
    
    async def start(self):
        """Start the background monitoring task"""
        if self._running:
            logger.warning("Monitor already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"🔄 Real-time market monitor started (refresh every {self.refresh_interval}s)")
    
    async def stop(self):
        """Stop the background monitoring task"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Real-time market monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                if self._watched_symbols:
                    await self._refresh_market_data()
                await asyncio.sleep(self.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _refresh_market_data(self):
        """Refresh market data for all watched symbols"""
        from rag.market_data_fetcher import get_market_data_fetcher
        
        fetcher = get_market_data_fetcher()
        self._refresh_count += 1
        refresh_start = datetime.now()
        
        logger.info(f"📊 Background refresh #{self._refresh_count} for {len(self._watched_symbols)} symbols")
        
        updates = []
        
        # Fetch market overview (always)
        market_overview = await self._fetch_with_cache("market_overview", fetcher.fetch_market_overview)
        if market_overview:
            updates.append({
                "type": "market_overview",
                "data": market_overview
            })
            # Update RAG if available
            if self.rag_engine:
                market_text = fetcher.format_for_rag(market_overview, "market")
                await self.rag_engine.add_document(
                    content=market_text,
                    doc_id="market_overview",
                    source="realtime_monitor",
                    doc_type="market",
                    metadata={"type": "market_overview", "updated": datetime.now().isoformat()}
                )
        
        # Fetch data for each symbol
        for symbol in list(self._watched_symbols):
            try:
                cache_key = f"stock_{symbol}"
                stock_data = await self._fetch_with_cache(
                    cache_key,
                    lambda: fetcher.fetch_stock_data(symbol, "NSE")
                )
                
                if stock_data and "error" not in stock_data:
                    # Record for anomaly detection
                    self.anomaly_detector.record_price(
                        symbol=symbol,
                        price=stock_data.get("current_price", 0),
                        volume=stock_data.get("volume"),
                        change_percent=self._calc_change_percent(stock_data)
                    )
                    
                    updates.append({
                        "type": "stock_update",
                        "symbol": symbol,
                        "data": {
                            "symbol": symbol,
                            "price": stock_data.get("current_price", 0),
                            "previous_close": stock_data.get("previous_close", 0),
                            "change_percent": self._calc_change_percent(stock_data),
                            "day_high": stock_data.get("day_high", 0),
                            "day_low": stock_data.get("day_low", 0),
                            "volume": stock_data.get("volume", 0),
                            "updated_at": datetime.now().isoformat()
                        }
                    })
                    
                    # Update RAG if available
                    if self.rag_engine:
                        stock_text = fetcher.format_for_rag(stock_data, "stock")
                        await self.rag_engine.add_document(
                            content=stock_text,
                            doc_id=f"stock_data_{symbol}",
                            source="realtime_monitor",
                            doc_type="stock_analysis",
                            metadata={
                                "symbol": symbol,
                                "type": "stock_data",
                                "updated": datetime.now().isoformat()
                            }
                        )
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        self._last_refresh = datetime.now()
        refresh_duration = (self._last_refresh - refresh_start).total_seconds()
        
        logger.info(f"✅ Refresh #{self._refresh_count} complete in {refresh_duration:.2f}s")
        
        # Broadcast updates to all WebSocket clients
        if updates:
            await self._broadcast_updates(updates)
    
    def _calc_change_percent(self, stock_data: Dict) -> float:
        """Calculate change percent from stock data"""
        current = stock_data.get("current_price", 0)
        previous = stock_data.get("previous_close", 0)
        if previous > 0:
            return ((current - previous) / previous) * 100
        return 0
    
    async def _fetch_with_cache(self, cache_key: str, fetch_func) -> Optional[Dict]:
        """Fetch data with caching"""
        # Check cache first
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Fetch fresh data
        try:
            data = await fetch_func()
            if data:
                await self.cache.set(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Fetch error for {cache_key}: {e}")
            return None
    
    async def _broadcast_updates(self, updates: List[Dict]):
        """Broadcast updates to all WebSocket clients"""
        if not self._websocket_clients:
            return
        
        message = json.dumps({
            "type": "market_update",
            "timestamp": datetime.now().isoformat(),
            "updates": updates
        })
        
        disconnected = set()
        for ws in self._websocket_clients:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.debug(f"WebSocket send error: {e}")
                disconnected.add(ws)
        
        # Clean up disconnected clients
        self._websocket_clients -= disconnected
    
    async def _broadcast_anomaly(self, anomaly: AnomalyAlert):
        """Broadcast anomaly alert to all WebSocket clients"""
        if not self._websocket_clients:
            return
        
        message = json.dumps({
            "type": "anomaly_alert",
            "timestamp": datetime.now().isoformat(),
            "alert": anomaly.to_dict()
        })
        
        disconnected = set()
        for ws in self._websocket_clients:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.debug(f"WebSocket send error: {e}")
                disconnected.add(ws)
        
        self._websocket_clients -= disconnected
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            "running": self._running,
            "refresh_interval_seconds": self.refresh_interval,
            "watched_symbols": list(self._watched_symbols),
            "websocket_clients": len(self._websocket_clients),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "refresh_count": self._refresh_count,
            "cache_stats": self.cache.get_stats(),
            "active_alerts": len(self.anomaly_detector.get_alerts(unacknowledged_only=True))
        }


# Singleton instance
_monitor_instance: Optional[RealtimeMarketMonitor] = None


def get_realtime_monitor(
    refresh_interval: int = 60,
    cache_ttl: int = 60,
    rag_engine = None
) -> RealtimeMarketMonitor:
    """Get or create the realtime monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = RealtimeMarketMonitor(
            refresh_interval=refresh_interval,
            cache_ttl=cache_ttl,
            rag_engine=rag_engine
        )
    return _monitor_instance
