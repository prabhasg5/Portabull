import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useThemeStore } from '../store/themeStore';

interface Hypothesis {
  hypothesis_id: string;
  title: string;
  description: string;
  hypothesis_type: string;
  affected_symbols: string[];
  confidence: number;
  supporting_evidence: string[];
  potential_actions: string[];
  risk_factors: string[];
  time_horizon: string;
  generated_at: string;
  status: string;
}

const typeConfig: Record<string, { icon: string; color: string; bg: string }> = {
  bullish: { icon: '🟢', color: 'text-green-400', bg: 'bg-green-500/20' },
  bearish: { icon: '🔴', color: 'text-red-400', bg: 'bg-red-500/20' },
  neutral: { icon: '🟡', color: 'text-yellow-400', bg: 'bg-yellow-500/20' },
  risk: { icon: '⚠️', color: 'text-orange-400', bg: 'bg-orange-500/20' },
  opportunity: { icon: '💎', color: 'text-cyan-400', bg: 'bg-cyan-500/20' },
  hedging: { icon: '🛡️', color: 'text-amber-400', bg: 'bg-amber-500/20' },
  rebalancing: { icon: '⚖️', color: 'text-blue-400', bg: 'bg-blue-500/20' }
};

export const HypothesisPanel: React.FC = () => {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { isDarkMode } = useThemeStore();

  const fetchHypotheses = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/hypotheses');
      const data = await response.json();
      setHypotheses(data.hypotheses || []);
      setError(null);
    } catch (e) {
      setError('Failed to fetch hypotheses');
      console.error('Error fetching hypotheses:', e);
    } finally {
      setLoading(false);
    }
  };

  const generateHypotheses = async () => {
    try {
      setGenerating(true);
      const response = await fetch('http://localhost:8000/api/hypotheses/generate', {
        method: 'POST'
      });
      const data = await response.json();
      setHypotheses(data.hypotheses || []);
      setError(null);
    } catch (e) {
      setError('Failed to generate hypotheses');
      console.error('Error generating hypotheses:', e);
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    fetchHypotheses();
  }, []);

  const getTypeConfig = (type: string) => {
    return typeConfig[type] || { icon: '📋', color: 'text-gray-400', bg: 'bg-gray-500/20' };
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className={`rounded-xl p-6 backdrop-blur border ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200 shadow-sm'}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">💡</span>
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Investment Hypotheses</h3>
          <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'text-gray-500 bg-gray-700' : 'text-gray-600 bg-gray-100'}`}>
            Auto-Generated
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={fetchHypotheses}
            disabled={loading}
            className={`text-xs px-3 py-1.5 rounded transition-colors disabled:opacity-50 ${
              isDarkMode 
                ? 'text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600'
                : 'text-gray-600 hover:text-gray-900 bg-gray-100 hover:bg-gray-200'
            }`}
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
          <button
            onClick={generateHypotheses}
            disabled={generating}
            className="text-xs text-white px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 transition-colors disabled:opacity-50 flex items-center gap-1"
          >
            {generating ? (
              <>
                <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Generating...
              </>
            ) : (
              <>✨ Generate New</>
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Hypotheses List */}
      <div className="space-y-3 max-h-[600px] overflow-y-auto custom-scrollbar">
        <AnimatePresence mode="popLayout">
          {hypotheses.map((hyp) => {
            const config = getTypeConfig(hyp.hypothesis_type);
            const isExpanded = expanded === hyp.hypothesis_id;

            return (
              <motion.div
                key={hyp.hypothesis_id}
                layout
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className={`p-4 rounded-lg ${config.bg} border border-gray-700/50 cursor-pointer transition-all hover:border-gray-600`}
                onClick={() => setExpanded(isExpanded ? null : hyp.hypothesis_id)}
              >
                {/* Header */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-2 flex-1">
                    <span className="text-lg">{config.icon}</span>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <h4 className={`font-medium ${config.color}`}>{hyp.title}</h4>
                        <span className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded">
                          {hyp.time_horizon}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400 mt-1 line-clamp-2">
                        {hyp.description}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex flex-col items-end gap-1">
                    <div className="flex items-center gap-1">
                      <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            hyp.confidence > 0.7 ? 'bg-green-500' :
                            hyp.confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${hyp.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400">{Math.round(hyp.confidence * 100)}%</span>
                    </div>
                    <motion.span
                      animate={{ rotate: isExpanded ? 180 : 0 }}
                      className="text-gray-500"
                    >
                      ▼
                    </motion.span>
                  </div>
                </div>

                {/* Symbols */}
                {hyp.affected_symbols.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {hyp.affected_symbols.map((symbol) => (
                      <span
                        key={symbol}
                        className="text-xs bg-gray-700/50 text-gray-300 px-2 py-0.5 rounded"
                      >
                        {symbol}
                      </span>
                    ))}
                  </div>
                )}

                {/* Expanded Content */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-4 pt-4 border-t border-gray-700/50 space-y-4"
                    >
                      {/* Supporting Evidence */}
                      <div>
                        <h5 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-1">
                          <span>📊</span> Supporting Evidence
                        </h5>
                        <ul className="space-y-1">
                          {hyp.supporting_evidence.map((evidence, idx) => (
                            <li key={idx} className="text-sm text-gray-400 flex items-start gap-2">
                              <span className="text-gray-500">•</span>
                              {evidence}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Potential Actions */}
                      <div>
                        <h5 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-1">
                          <span>🎯</span> Potential Actions
                        </h5>
                        <ul className="space-y-1">
                          {hyp.potential_actions.map((action, idx) => (
                            <li key={idx} className="text-sm text-green-400/80 flex items-start gap-2">
                              <span className="text-gray-500">→</span>
                              {action}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Risk Factors */}
                      <div>
                        <h5 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-1">
                          <span>⚠️</span> Risk Factors
                        </h5>
                        <ul className="space-y-1">
                          {hyp.risk_factors.map((risk, idx) => (
                            <li key={idx} className="text-sm text-orange-400/80 flex items-start gap-2">
                              <span className="text-gray-500">!</span>
                              {risk}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Footer */}
                      <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-700/50">
                        <span>Generated: {formatDate(hyp.generated_at)}</span>
                        <span className={`px-2 py-0.5 rounded ${
                          hyp.status === 'active' ? 'bg-green-500/20 text-green-400' :
                          hyp.status === 'validated' ? 'bg-blue-500/20 text-blue-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {hyp.status}
                        </span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {hypotheses.length === 0 && !loading && (
          <div className="text-center py-12 text-gray-500">
            <span className="text-4xl mb-3 block">💡</span>
            <p className="text-lg">No hypotheses yet</p>
            <p className="text-sm mt-2">Click "Generate New" to analyze your portfolio</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default HypothesisPanel;
