/**
 * Portabull - Real-time Market WebSocket Hook
 * Connects to the backend WebSocket for live market updates
 */

import { useEffect, useRef, useState, useCallback } from 'react';

export interface MarketUpdate {
  type: 'market_overview' | 'stock_update';
  symbol?: string;
  data: {
    symbol?: string;
    price?: number;
    previous_close?: number;
    change_percent?: number;
    day_high?: number;
    day_low?: number;
    volume?: number;
    updated_at?: string;
    indices?: Record<string, {
      current: number;
      previous_close: number;
      change: number;
      change_percent: number;
    }>;
  };
}

export interface AnomalyAlert {
  alert_id: string;
  alert_type: 'price_spike' | 'price_drop' | 'volume_surge' | 'unusual_volatility';
  severity: 'low' | 'medium' | 'high' | 'critical';
  symbol: string;
  title: string;
  description: string;
  current_value: number;
  previous_value: number;
  change_percent: number;
  threshold: number;
  timestamp: string;
  acknowledged: boolean;
}

export interface WebSocketMessage {
  type: 'connected' | 'pong' | 'subscribed' | 'unsubscribed' | 'market_update' | 'anomaly_alert' | 'error';
  timestamp: string;
  message?: string;
  updates?: MarketUpdate[];
  alert?: AnomalyAlert;
  symbols?: string[];
}

interface UseMarketWebSocketOptions {
  autoConnect?: boolean;
  onUpdate?: (updates: MarketUpdate[]) => void;
  onAnomaly?: (alert: AnomalyAlert) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
}

export function useMarketWebSocket(options: UseMarketWebSocketOptions = {}) {
  const {
    autoConnect = true,
    onUpdate,
    onAnomaly,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [marketData, setMarketData] = useState<Record<string, MarketUpdate['data']>>({});
  const [alerts, setAlerts] = useState<AnomalyAlert[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000; // 3 seconds

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionStatus('connecting');

    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = import.meta.env.VITE_WS_URL || `${protocol}//${window.location.hostname}:8000`;
    const wsUrl = `${host}/ws/market`;

    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('🔌 WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        onConnect?.();

        // Start ping interval to keep connection alive
        pingIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000); // Ping every 30 seconds
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          switch (message.type) {
            case 'connected':
              console.log('✅ Market feed connected:', message.message);
              break;
              
            case 'pong':
              // Keep-alive response, nothing to do
              break;
              
            case 'market_update':
              setLastUpdate(new Date());
              if (message.updates) {
                // Update local market data
                const newData = { ...marketData };
                message.updates.forEach((update) => {
                  const key = update.symbol || 'market_overview';
                  newData[key] = update.data;
                });
                setMarketData(newData);
                onUpdate?.(message.updates);
              }
              break;
              
            case 'anomaly_alert':
              if (message.alert) {
                console.log('🚨 Anomaly detected:', message.alert.title);
                setAlerts((prev) => [message.alert!, ...prev].slice(0, 50)); // Keep last 50 alerts
                onAnomaly?.(message.alert);
              }
              break;
              
            case 'subscribed':
              console.log('📡 Subscribed to:', message.symbols);
              break;
              
            case 'unsubscribed':
              console.log('📴 Unsubscribed from:', message.symbols);
              break;
              
            case 'error':
              console.error('WebSocket error message:', message.message);
              break;
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('🔌 WebSocket disconnected', event.code, event.reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        onDisconnect?.();

        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }

        // Attempt to reconnect if not a clean close
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          console.log(`Reconnecting in ${reconnectDelay}ms (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})...`);
          reconnectTimeoutRef.current = setTimeout(connect, reconnectDelay);
        }
      };

      wsRef.current.onerror = (event) => {
        console.error('WebSocket error:', event);
        setConnectionStatus('error');
        onError?.(new Error('WebSocket connection error'));
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionStatus('error');
      onError?.(error as Error);
    }
  }, [onConnect, onDisconnect, onUpdate, onAnomaly, onError, marketData]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }
    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, []);

  const subscribe = useCallback((symbols: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        symbols,
      }));
    }
  }, []);

  const unsubscribe = useCallback((symbols: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'unsubscribe',
        symbols,
      }));
    }
  }, []);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect]); // Only run on mount/unmount

  return {
    isConnected,
    connectionStatus,
    lastUpdate,
    marketData,
    alerts,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    clearAlerts,
  };
}

export default useMarketWebSocket;
