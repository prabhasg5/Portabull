import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Tool {
  name: string;
  type: string;
  description: string;
  parameters: Record<string, {
    type: string;
    required?: boolean;
    default?: any;
    description?: string;
    options?: string[];
  }>;
}

interface ToolCall {
  tool: string;
  parameters: Record<string, any>;
  called_at: string;
  result: any;
  success: boolean;
  error: string | null;
  execution_time_ms: number;
}

const toolIcons: Record<string, string> = {
  price_lookup: '💰',
  news_fetch: '📰',
  technical_analysis: '📊',
  fundamental_analysis: '📈',
  market_sentiment: '🎭',
  sector_analysis: '🏭',
  portfolio_metrics: '💼',
  hypothesis_check: '🔍'
};

export const AgentToolsPanel: React.FC = () => {
  const [tools, setTools] = useState<Tool[]>([]);
  const [history, setHistory] = useState<ToolCall[]>([]);
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);
  const [params, setParams] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ToolCall | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTools();
    fetchHistory();
  }, []);

  const fetchTools = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/tools');
      const data = await response.json();
      setTools(data.tools || []);
    } catch (e) {
      console.error('Error fetching tools:', e);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/tools/history?limit=10');
      const data = await response.json();
      setHistory(data.history || []);
    } catch (e) {
      console.error('Error fetching history:', e);
    }
  };

  const callTool = async () => {
    if (!selectedTool) return;

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const response = await fetch(
        `http://localhost:8000/api/tools/call?tool_name=${selectedTool.type}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        }
      );

      const data = await response.json();
      setResult(data);
      fetchHistory();
    } catch (e: any) {
      setError(e.message || 'Tool call failed');
    } finally {
      setLoading(false);
    }
  };

  const selectTool = (tool: Tool) => {
    setSelectedTool(tool);
    setParams({});
    setResult(null);
    setError(null);

    // Set default values
    const defaults: Record<string, any> = {};
    Object.entries(tool.parameters).forEach(([key, param]) => {
      if (param.default !== undefined) {
        defaults[key] = param.default;
      }
    });
    setParams(defaults);
  };

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur border border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🛠️</span>
          <h3 className="text-lg font-semibold text-white">Agent Tools</h3>
          <span className="text-xs text-gray-500 bg-gray-700 px-2 py-1 rounded">
            {tools.length} Available
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Tools List */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-400">Available Tools</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {tools.map((tool) => (
              <motion.button
                key={tool.name}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => selectTool(tool)}
                className={`w-full text-left p-3 rounded-lg border transition-colors ${
                  selectedTool?.name === tool.name
                    ? 'bg-blue-600/30 border-blue-500'
                    : 'bg-gray-700/50 border-gray-600 hover:border-gray-500'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-xl">{toolIcons[tool.type] || '🔧'}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-200">{tool.name}</p>
                    <p className="text-xs text-gray-400 truncate">{tool.description}</p>
                  </div>
                </div>
              </motion.button>
            ))}
          </div>

          {/* Recent History */}
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-400 mb-2">Recent Calls</h4>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {history.slice(0, 5).map((call, idx) => (
                <div
                  key={idx}
                  className={`flex items-center justify-between p-2 rounded text-xs ${
                    call.success ? 'bg-green-500/10' : 'bg-red-500/10'
                  }`}
                >
                  <span className="flex items-center gap-1">
                    <span>{toolIcons[call.tool] || '🔧'}</span>
                    <span className="text-gray-300">{call.tool}</span>
                  </span>
                  <span className="text-gray-500">{call.execution_time_ms.toFixed(0)}ms</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Tool Configuration & Results */}
        <div className="space-y-4">
          {selectedTool ? (
            <>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl">{toolIcons[selectedTool.type] || '🔧'}</span>
                  <h4 className="text-sm font-medium text-white">{selectedTool.name}</h4>
                </div>
                <p className="text-xs text-gray-400 mb-4">{selectedTool.description}</p>

                {/* Parameters */}
                <div className="space-y-3">
                  {Object.entries(selectedTool.parameters).map(([key, param]) => (
                    <div key={key}>
                      <label className="block text-xs text-gray-400 mb-1">
                        {key}
                        {param.required && <span className="text-red-400 ml-1">*</span>}
                      </label>
                      {param.options ? (
                        <select
                          value={params[key] || param.default || ''}
                          onChange={(e) => setParams({ ...params, [key]: e.target.value })}
                          className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white"
                        >
                          {param.options.map((opt) => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      ) : param.type === 'array' ? (
                        <input
                          type="text"
                          placeholder={`${key} (comma-separated)`}
                          value={Array.isArray(params[key]) ? params[key].join(', ') : params[key] || ''}
                          onChange={(e) => setParams({
                            ...params,
                            [key]: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                          })}
                          className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white"
                        />
                      ) : (
                        <input
                          type={param.type === 'integer' ? 'number' : 'text'}
                          placeholder={param.description || key}
                          value={params[key] ?? param.default ?? ''}
                          onChange={(e) => setParams({
                            ...params,
                            [key]: param.type === 'integer' ? parseInt(e.target.value) : e.target.value
                          })}
                          className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white"
                        />
                      )}
                    </div>
                  ))}
                </div>

                <button
                  onClick={callTool}
                  disabled={loading}
                  className="w-full mt-4 bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Executing...
                    </>
                  ) : (
                    <>🚀 Execute Tool</>
                  )}
                </button>
              </div>

              {/* Results */}
              {error && (
                <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
                  {error}
                </div>
              )}

              {result && (
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-300">Result</span>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      result.success ? 'bg-green-500/30 text-green-400' : 'bg-red-500/30 text-red-400'
                    }`}>
                      {result.success ? 'Success' : 'Failed'} • {result.execution_time_ms.toFixed(0)}ms
                    </span>
                  </div>
                  <pre className="text-xs text-gray-400 bg-gray-800 rounded p-3 overflow-x-auto max-h-64 overflow-y-auto">
                    {JSON.stringify(result.result, null, 2)}
                  </pre>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              <div className="text-center">
                <span className="text-4xl mb-2 block">👈</span>
                <p>Select a tool to configure and execute</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AgentToolsPanel;
