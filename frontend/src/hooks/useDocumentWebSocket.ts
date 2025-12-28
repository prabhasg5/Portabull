import { useState, useEffect, useCallback, useRef } from 'react';

export interface DocumentEvent {
  event_type: 'added' | 'updated' | 'deleted' | 'cleared';
  doc_id: string;
  doc_type: string;
  symbol: string;
  content_preview: string;
  source: string;
  timestamp: string;
}

interface UseDocumentWebSocketReturn {
  events: DocumentEvent[];
  isConnected: boolean;
  error: string | null;
  clearEvents: () => void;
}

export function useDocumentWebSocket(): UseDocumentWebSocketReturn {
  const [events, setEvents] = useState<DocumentEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//localhost:8000/ws/documents`;
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Document feed WebSocket connected');
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'document_event') {
            setEvents(prev => {
              // Add new event at the beginning, keep last 50
              const newEvents = [data.data as DocumentEvent, ...prev];
              return newEvents.slice(0, 50);
            });
          }
        } catch (e) {
          console.error('Error parsing document event:', e);
        }
      };

      ws.onerror = (event) => {
        console.error('Document feed WebSocket error:', event);
        setError('WebSocket connection error');
      };

      ws.onclose = () => {
        console.log('Document feed WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt to reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 5000);
      };

    } catch (e) {
      console.error('Error creating document WebSocket:', e);
      setError('Failed to create WebSocket connection');
    }
  }, []);

  useEffect(() => {
    connect();

    // Ping every 30 seconds to keep connection alive
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    return () => {
      clearInterval(pingInterval);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  return {
    events,
    isConnected,
    error,
    clearEvents
  };
}
