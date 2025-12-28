import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDocumentWebSocket, DocumentEvent } from '../hooks/useDocumentWebSocket';

const docTypeConfig: Record<string, { icon: string; color: string; bg: string }> = {
  stock_data: { icon: '📈', color: 'text-green-400', bg: 'bg-green-500/20' },
  news: { icon: '📰', color: 'text-blue-400', bg: 'bg-blue-500/20' },
  hypothesis: { icon: '💡', color: 'text-yellow-400', bg: 'bg-yellow-500/20' },
  market_overview: { icon: '🌍', color: 'text-amber-400', bg: 'bg-amber-500/20' },
  sector: { icon: '🏭', color: 'text-orange-400', bg: 'bg-orange-500/20' },
  portfolio: { icon: '💼', color: 'text-cyan-400', bg: 'bg-cyan-500/20' },
  default: { icon: '📄', color: 'text-gray-400', bg: 'bg-gray-500/20' }
};

interface LiveDocumentFeedProps {
  maxEvents?: number;
  compact?: boolean;
}

export const LiveDocumentFeed: React.FC<LiveDocumentFeedProps> = ({
  maxEvents = 15,
  compact = false
}) => {
  const { events, isConnected, error, clearEvents } = useDocumentWebSocket();
  const [stats, setStats] = useState({
    total: 0,
    byType: {} as Record<string, number>
  });

  // Update stats when events change
  useEffect(() => {
    const byType: Record<string, number> = {};
    events.forEach(e => {
      byType[e.doc_type] = (byType[e.doc_type] || 0) + 1;
    });
    setStats({
      total: events.length,
      byType
    });
  }, [events]);

  const getDocConfig = (docType: string) => {
    return docTypeConfig[docType] || docTypeConfig.default;
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const displayEvents = events.slice(0, maxEvents);

  if (compact) {
    return (
      <div className="bg-gray-800/50 rounded-lg p-4 backdrop-blur">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-sm font-medium text-gray-300">Live Documents</span>
          </div>
          <span className="text-xs text-gray-500">{stats.total} events</span>
        </div>
        
        <div className="space-y-1 max-h-48 overflow-y-auto">
          <AnimatePresence mode="popLayout">
            {displayEvents.slice(0, 5).map((event, index) => {
              const config = getDocConfig(event.doc_type);
              return (
                <motion.div
                  key={`${event.doc_id}-${event.timestamp}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                  className={`flex items-center gap-2 p-2 rounded ${config.bg}`}
                >
                  <span>{config.icon}</span>
                  <span className={`text-xs ${config.color} truncate flex-1`}>
                    {event.doc_type}: {event.symbol || event.doc_id}
                  </span>
                  <span className="text-xs text-gray-500">
                    {formatTime(event.timestamp)}
                  </span>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur border border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <h3 className="text-lg font-semibold text-white">Live Document Feed</h3>
          </div>
          <span className="text-xs text-gray-500 bg-gray-700 px-2 py-1 rounded">
            Pathway Streaming
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">{stats.total} documents</span>
          <button
            onClick={clearEvents}
            className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Connection Status */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Stats Bar */}
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(stats.byType).map(([type, count]) => {
          const config = getDocConfig(type);
          return (
            <div
              key={type}
              className={`flex items-center gap-1 px-2 py-1 rounded ${config.bg}`}
            >
              <span>{config.icon}</span>
              <span className={`text-xs ${config.color}`}>{type}</span>
              <span className="text-xs text-gray-400">({count})</span>
            </div>
          );
        })}
      </div>

      {/* Events List */}
      <div className="space-y-2 max-h-96 overflow-y-auto custom-scrollbar">
        <AnimatePresence mode="popLayout">
          {displayEvents.map((event, index) => {
            const config = getDocConfig(event.doc_type);
            return (
              <motion.div
                key={`${event.doc_id}-${event.timestamp}`}
                initial={{ opacity: 0, y: -20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 20, scale: 0.95 }}
                transition={{ 
                  duration: 0.3,
                  delay: index * 0.02
                }}
                className={`p-3 rounded-lg ${config.bg} border border-gray-700/50`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start gap-2 flex-1 min-w-0">
                    <span className="text-lg">{config.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className={`text-sm font-medium ${config.color}`}>
                          {event.doc_type}
                        </span>
                        {event.symbol && (
                          <span className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded">
                            {event.symbol}
                          </span>
                        )}
                        <span className="text-xs text-gray-500">
                          {event.source}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400 mt-1 truncate">
                        {event.content_preview}
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      event.event_type === 'added' ? 'bg-green-500/30 text-green-400' :
                      event.event_type === 'updated' ? 'bg-blue-500/30 text-blue-400' :
                      event.event_type === 'deleted' ? 'bg-red-500/30 text-red-400' :
                      'bg-gray-500/30 text-gray-400'
                    }`}>
                      {event.event_type}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatTime(event.timestamp)}
                    </span>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {displayEvents.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <span className="text-4xl mb-2 block">📡</span>
            <p>Waiting for documents...</p>
            <p className="text-xs mt-1">Documents will appear here as they are processed</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveDocumentFeed;
