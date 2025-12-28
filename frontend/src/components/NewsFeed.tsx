import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useThemeStore } from '../store/themeStore';

interface NewsArticle {
  article_id: string;
  title: string;
  summary: string;
  source: string;
  url: string;
  published_at: string;
  symbols: string[];
  categories: string[];
  sentiment: 'positive' | 'negative' | 'neutral' | null;
}

const sentimentConfig = {
  positive: { icon: '📈', color: 'text-green-400', bg: 'bg-green-500/20', border: 'border-green-500/30' },
  negative: { icon: '📉', color: 'text-red-400', bg: 'bg-red-500/20', border: 'border-red-500/30' },
  neutral: { icon: '➡️', color: 'text-gray-400', bg: 'bg-gray-500/20', border: 'border-gray-500/30' }
};

interface NewsFeedProps {
  symbol?: string;
  compact?: boolean;
}

export const NewsFeed: React.FC<NewsFeedProps> = ({ symbol, compact = false }) => {
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [filter, setFilter] = useState<'all' | 'positive' | 'negative' | 'neutral'>('all');
  const [error, setError] = useState<string | null>(null);
  const { isDarkMode } = useThemeStore();

  const fetchNews = useCallback(async () => {
    try {
      setLoading(true);
      let url = 'http://localhost:8000/api/news?limit=30';
      if (symbol) url += `&symbol=${symbol}`;
      if (filter !== 'all') url += `&sentiment=${filter}`;

      const response = await fetch(url);
      const data = await response.json();
      setArticles(data.articles || []);
      setError(null);
    } catch (e) {
      setError('Failed to fetch news');
      console.error('Error fetching news:', e);
    } finally {
      setLoading(false);
    }
  }, [symbol, filter]);

  const fetchLatestNews = async () => {
    try {
      setFetching(true);
      const response = await fetch('http://localhost:8000/api/news/fetch', {
        method: 'POST'
      });
      const data = await response.json();
      await fetchNews();
      setError(null);
    } catch (e) {
      setError('Failed to fetch latest news');
      console.error('Error fetching news:', e);
    } finally {
      setFetching(false);
    }
  };

  const toggleStreaming = async () => {
    try {
      if (streaming) {
        await fetch('http://localhost:8000/api/news/stream/stop', { method: 'POST' });
        setStreaming(false);
      } else {
        await fetch('http://localhost:8000/api/news/stream/start', { method: 'POST' });
        setStreaming(true);
      }
    } catch (e) {
      console.error('Error toggling streaming:', e);
    }
  };

  useEffect(() => {
    fetchNews();
  }, [fetchNews]);

  // Auto-refresh when streaming
  useEffect(() => {
    if (streaming) {
      const interval = setInterval(fetchNews, 30000);
      return () => clearInterval(interval);
    }
  }, [streaming, fetchNews]);

  const getSentimentConfig = (sentiment: string | null) => {
    return sentimentConfig[sentiment as keyof typeof sentimentConfig] || sentimentConfig.neutral;
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  if (compact) {
    return (
      <div className="bg-gray-800/50 rounded-lg p-4 backdrop-blur">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span>📰</span>
            <span className="text-sm font-medium text-gray-300">News Feed</span>
            {streaming && (
              <span className="flex items-center gap-1 text-xs text-green-400">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                Live
              </span>
            )}
          </div>
        </div>
        
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {articles.slice(0, 5).map((article) => {
            const config = getSentimentConfig(article.sentiment);
            return (
              <a
                key={article.article_id}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className={`block p-2 rounded ${config.bg} hover:opacity-80 transition-opacity`}
              >
                <p className="text-xs text-gray-300 line-clamp-2">{article.title}</p>
                <div className="flex items-center justify-between mt-1 text-xs text-gray-500">
                  <span>{article.source}</span>
                  <span>{formatTime(article.published_at)}</span>
                </div>
              </a>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className={`rounded-xl p-6 backdrop-blur border ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200 shadow-sm'}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">📰</span>
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Financial News</h3>
          {streaming && (
            <span className="flex items-center gap-1 text-xs text-green-400 bg-green-500/20 px-2 py-1 rounded">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
              Live Streaming
            </span>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={fetchLatestNews}
            disabled={fetching}
            className="text-xs text-white px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 transition-colors disabled:opacity-50 flex items-center gap-1"
          >
            {fetching ? (
              <>
                <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Fetching...
              </>
            ) : (
              <>🔄 Fetch Latest</>
            )}
          </button>
          <button
            onClick={toggleStreaming}
            className={`text-xs px-3 py-1.5 rounded transition-colors flex items-center gap-1 ${
              streaming
                ? 'bg-red-600 hover:bg-red-500 text-white'
                : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
            }`}
          >
            {streaming ? '⏹️ Stop' : '▶️ Stream'}
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2 mb-4">
        {(['all', 'positive', 'negative', 'neutral'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`text-xs px-3 py-1.5 rounded transition-colors ${
              filter === f
                ? f === 'positive' ? 'bg-green-600 text-white' :
                  f === 'negative' ? 'bg-red-600 text-white' :
                  f === 'neutral' ? 'bg-gray-600 text-white' :
                  'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
          >
            {f === 'all' ? '🌐 All' :
             f === 'positive' ? '📈 Positive' :
             f === 'negative' ? '📉 Negative' :
             '➡️ Neutral'}
          </button>
        ))}
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Articles List */}
      <div className="space-y-3 max-h-[500px] overflow-y-auto custom-scrollbar">
        <AnimatePresence mode="popLayout">
          {articles.map((article, index) => {
            const config = getSentimentConfig(article.sentiment);
            return (
              <motion.a
                key={article.article_id}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ delay: index * 0.02 }}
                className={`block p-4 rounded-lg ${config.bg} border ${config.border} hover:opacity-90 transition-all`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <span className="text-lg">{config.icon}</span>
                      <span className="text-xs text-gray-500">{article.source}</span>
                      {article.symbols.length > 0 && (
                        <div className="flex gap-1">
                          {article.symbols.slice(0, 3).map((s) => (
                            <span key={s} className="text-xs bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded">
                              {s}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <h4 className="text-sm font-medium text-gray-200 line-clamp-2">
                      {article.title}
                    </h4>
                    <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                      {article.summary}
                    </p>
                  </div>
                  <span className="text-xs text-gray-500 whitespace-nowrap">
                    {formatTime(article.published_at)}
                  </span>
                </div>
              </motion.a>
            );
          })}
        </AnimatePresence>

        {articles.length === 0 && !loading && (
          <div className="text-center py-12 text-gray-500">
            <span className="text-4xl mb-3 block">📰</span>
            <p className="text-lg">No news articles yet</p>
            <p className="text-sm mt-2">Click "Fetch Latest" to get financial news</p>
          </div>
        )}

        {loading && (
          <div className="text-center py-8">
            <svg className="animate-spin h-8 w-8 mx-auto text-blue-500" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <p className="text-gray-500 mt-2">Loading news...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default NewsFeed;
