/**
 * Portabull - Real-time Market Monitor Component
 * Shows live market data and anomaly alerts
 */

import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMarketWebSocket, AnomalyAlert } from '../hooks/useMarketWebSocket';

interface AlertCardProps {
  alert: AnomalyAlert;
  onDismiss?: () => void;
}

const AlertCard: React.FC<AlertCardProps> = ({ alert, onDismiss }) => {
  const severityColors = {
    low: 'bg-blue-500/20 border-blue-500 text-blue-400',
    medium: 'bg-yellow-500/20 border-yellow-500 text-yellow-400',
    high: 'bg-orange-500/20 border-orange-500 text-orange-400',
    critical: 'bg-red-500/20 border-red-500 text-red-400',
  };

  const alertIcons = {
    price_spike: '📈',
    price_drop: '📉',
    volume_surge: '📊',
    unusual_volatility: '⚡',
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 100 }}
      className={`p-4 rounded-lg border ${severityColors[alert.severity]} backdrop-blur-sm`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{alertIcons[alert.alert_type]}</span>
          <div>
            <h4 className="font-semibold">{alert.title}</h4>
            <p className="text-sm opacity-80">{alert.description}</p>
            <p className="text-xs opacity-60 mt-1">
              {new Date(alert.timestamp).toLocaleTimeString()}
            </p>
          </div>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-white/60 hover:text-white transition-colors"
          >
            ✕
          </button>
        )}
      </div>
    </motion.div>
  );
};

interface MarketStatusProps {
  className?: string;
}

export const MarketStatus: React.FC<MarketStatusProps> = ({ className = '' }) => {
  const {
    isConnected,
    connectionStatus,
    lastUpdate,
    marketData,
    alerts,
    clearAlerts,
  } = useMarketWebSocket({
    autoConnect: true,
    onAnomaly: (alert) => {
      // Show browser notification for high/critical alerts
      if (['high', 'critical'].includes(alert.severity)) {
        if (Notification.permission === 'granted') {
          new Notification(`🚨 ${alert.title}`, {
            body: alert.description,
            icon: '/vite.svg',
          });
        }
      }
    },
  });

  // Request notification permission on mount
  useEffect(() => {
    if (Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  const statusColors = {
    connected: 'text-green-400',
    connecting: 'text-yellow-400',
    disconnected: 'text-gray-400',
    error: 'text-red-400',
  };

  const statusIcons = {
    connected: '🟢',
    connecting: '🟡',
    disconnected: '⚪',
    error: '🔴',
  };

  return (
    <div className={`${className}`}>
      {/* Connection Status */}
      <div className="flex items-center justify-between mb-4 p-3 bg-gray-800/50 rounded-lg">
        <div className="flex items-center gap-2">
          <span>{statusIcons[connectionStatus]}</span>
          <span className={`text-sm ${statusColors[connectionStatus]}`}>
            {connectionStatus === 'connected'
              ? 'Live'
              : connectionStatus === 'connecting'
              ? 'Connecting...'
              : connectionStatus === 'error'
              ? 'Connection Error'
              : 'Disconnected'}
          </span>
        </div>
        {lastUpdate && (
          <span className="text-xs text-gray-500">
            Last update: {lastUpdate.toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* Market Overview */}
      {marketData.market_overview && (
        <div className="mb-4 p-3 bg-gray-800/50 rounded-lg">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Market Indices</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {marketData.market_overview.indices &&
              Object.entries(marketData.market_overview.indices).map(([name, data]: [string, any]) => (
                <div key={name} className="flex justify-between items-center">
                  <span className="text-gray-300">{name}</span>
                  <span
                    className={
                      data.change_percent >= 0 ? 'text-green-400' : 'text-red-400'
                    }
                  >
                    {data.change_percent >= 0 ? '▲' : '▼'} {Math.abs(data.change_percent).toFixed(2)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-400">
              Alerts ({alerts.length})
            </h3>
            <button
              onClick={clearAlerts}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              Clear All
            </button>
          </div>
          <AnimatePresence>
            {alerts.slice(0, 5).map((alert) => (
              <AlertCard key={alert.alert_id} alert={alert} />
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
};

// Floating Alerts Component (for showing alerts anywhere in the app)
export const FloatingAlerts: React.FC = () => {
  const { alerts } = useMarketWebSocket({ autoConnect: true });
  const [visibleAlerts, setVisibleAlerts] = React.useState<AnomalyAlert[]>([]);

  useEffect(() => {
    // Show new alerts that are high or critical severity
    const newHighAlerts = alerts.filter(
      (a) =>
        ['high', 'critical'].includes(a.severity) &&
        !visibleAlerts.find((v) => v.alert_id === a.alert_id)
    );

    if (newHighAlerts.length > 0) {
      setVisibleAlerts((prev) => [...newHighAlerts, ...prev].slice(0, 3));

      // Auto-dismiss after 10 seconds
      setTimeout(() => {
        setVisibleAlerts((prev) =>
          prev.filter((a) => !newHighAlerts.find((n) => n.alert_id === a.alert_id))
        );
      }, 10000);
    }
  }, [alerts]);

  const dismissAlert = (alertId: string) => {
    setVisibleAlerts((prev) => prev.filter((a) => a.alert_id !== alertId));
  };

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
      <AnimatePresence>
        {visibleAlerts.map((alert) => (
          <AlertCard
            key={alert.alert_id}
            alert={alert}
            onDismiss={() => dismissAlert(alert.alert_id)}
          />
        ))}
      </AnimatePresence>
    </div>
  );
};

export default MarketStatus;
