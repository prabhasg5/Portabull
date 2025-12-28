import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api';
import { useThemeStore } from '../store/themeStore';

interface Quote {
  symbol: string;
  exchange: string;
  last_price: number;
  day_change: number;
  day_change_percent: number;
  company_name: string;
  sector: string;
}

interface TradeModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  quote: Quote | null;
  action: 'buy' | 'sell';
  maxQuantity?: number;
}

const TradeModal: React.FC<TradeModalProps> = ({
  isOpen,
  onClose,
  symbol,
  quote,
  action,
  maxQuantity
}) => {
  const [quantity, setQuantity] = useState(1);
  const queryClient = useQueryClient();

  const tradeMutation = useMutation({
    mutationFn: async () => {
      const endpoint = action === 'buy' ? '/api/paper/buy' : '/api/paper/sell';
      const response = await api.post(endpoint, {
        symbol,
        quantity,
        exchange: quote?.exchange || 'NSE'
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paper-portfolio'] });
      queryClient.invalidateQueries({ queryKey: ['paper-holdings'] });
      onClose();
    }
  });

  if (!isOpen || !quote) return null;

  const totalValue = quote.last_price * quantity;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-gray-800 rounded-xl p-6 w-full max-w-md mx-4"
      >
        <h3 className="text-xl font-bold text-white mb-4">
          {action === 'buy' ? '🛒 Buy' : '💰 Sell'} {symbol}
        </h3>

        <div className="space-y-4">
          <div className="bg-gray-700 rounded-lg p-4">
            <p className="text-gray-400 text-sm">Current Price</p>
            <p className="text-2xl font-bold text-white">
              ₹{quote.last_price.toLocaleString()}
            </p>
            <p className={`text-sm ${quote.day_change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {quote.day_change >= 0 ? '+' : ''}{quote.day_change_percent.toFixed(2)}% today
            </p>
          </div>

          <div>
            <label className="text-gray-400 text-sm block mb-2">Quantity</label>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setQuantity(Math.max(1, quantity - 1))}
                className="bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-gray-600"
              >
                -
              </button>
              <input
                type="number"
                value={quantity}
                onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 1))}
                className="bg-gray-700 text-white text-center px-4 py-2 rounded-lg w-24"
                min={1}
                max={action === 'sell' && maxQuantity ? maxQuantity : undefined}
              />
              <button
                onClick={() => setQuantity(quantity + 1)}
                className="bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-gray-600"
              >
                +
              </button>
            </div>
            {action === 'sell' && maxQuantity && (
              <p className="text-gray-500 text-xs mt-1">Max: {maxQuantity}</p>
            )}
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <p className="text-gray-400 text-sm">Total Value</p>
            <p className="text-2xl font-bold text-white">
              ₹{totalValue.toLocaleString()}
            </p>
          </div>

          {tradeMutation.isError && (
            <div className="bg-red-900/50 text-red-300 p-3 rounded-lg text-sm">
              {(tradeMutation.error as any)?.response?.data?.detail || 'Trade failed'}
            </div>
          )}

          <div className="flex space-x-3">
            <button
              onClick={onClose}
              className="flex-1 bg-gray-700 text-white py-3 rounded-lg hover:bg-gray-600"
            >
              Cancel
            </button>
            <button
              onClick={() => tradeMutation.mutate()}
              disabled={tradeMutation.isPending}
              className={`flex-1 py-3 rounded-lg font-semibold ${
                action === 'buy'
                  ? 'bg-green-600 hover:bg-green-500 text-white'
                  : 'bg-red-600 hover:bg-red-500 text-white'
              } disabled:opacity-50`}
            >
              {tradeMutation.isPending ? 'Processing...' : action === 'buy' ? 'Buy' : 'Sell'}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

interface SearchResult {
  symbol: string;
  name: string;
  sector: string;
  exchange: string;
  last_price?: number;
  day_change_percent?: number;
}

const StockSearch: React.FC<{ onSelectStock: (symbol: string, quote: Quote) => void }> = ({
  onSelectStock
}) => {
  const [query, setQuery] = useState('');
  const [exchange, setExchange] = useState('NSE');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [showResults, setShowResults] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  // Debounce the search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300); // 300ms debounce

    return () => clearTimeout(timer);
  }, [query]);

  // Search for stocks as user types
  const searchQuery = useQuery({
    queryKey: ['stock-search', debouncedQuery, exchange],
    queryFn: async () => {
      const response = await api.get(`/api/paper/search?query=${debouncedQuery}&exchange=${exchange}&limit=8`);
      return response.data.results as SearchResult[];
    },
    enabled: debouncedQuery.length >= 1,
    staleTime: 30000, // Cache for 30 seconds
  });

  // Get quote for selected stock
  const quoteQuery = useQuery({
    queryKey: ['stock-quote', selectedSymbol, exchange],
    queryFn: async () => {
      if (!selectedSymbol) return null;
      const response = await api.get(`/api/paper/quote/${selectedSymbol}?exchange=${exchange}`);
      return response.data;
    },
    enabled: !!selectedSymbol
  });

  const handleSelectResult = (result: SearchResult) => {
    setSelectedSymbol(result.symbol);
    setQuery(result.symbol);
    setShowResults(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toUpperCase();
    setQuery(value);
    setShowResults(true);
    setSelectedSymbol(null); // Clear selection when typing
  };

  return (
    <div className="bg-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-4">🔍 Search Stocks</h3>

      <div className="flex space-x-2 mb-4">
        <div className="flex-1 relative">
          <input
            type="text"
            value={query}
            onChange={handleInputChange}
            onFocus={() => setShowResults(true)}
            placeholder="Search by name, symbol, or sector..."
            className="w-full bg-gray-700 text-white px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
          />
          
          {/* Search Results Dropdown */}
          <AnimatePresence>
            {showResults && debouncedQuery.length >= 1 && searchQuery.data && searchQuery.data.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute z-50 top-full left-0 right-0 mt-1 bg-gray-700 rounded-lg shadow-xl border border-gray-600 max-h-64 overflow-y-auto"
              >
                {searchQuery.data.map((result, index) => (
                  <button
                    key={`${result.symbol}-${index}`}
                    onClick={() => handleSelectResult(result)}
                    className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-600 transition-colors text-left border-b border-gray-600 last:border-b-0"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-white font-bold">{result.symbol}</span>
                        <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded">
                          {result.sector}
                        </span>
                      </div>
                      <span className="text-gray-400 text-sm">{result.name}</span>
                    </div>
                    {result.last_price && (
                      <div className="text-right">
                        <span className="text-white font-semibold">₹{result.last_price.toLocaleString()}</span>
                        {result.day_change_percent !== undefined && (
                          <span className={`block text-xs ${result.day_change_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {result.day_change_percent >= 0 ? '+' : ''}{result.day_change_percent.toFixed(2)}%
                          </span>
                        )}
                      </div>
                    )}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Loading indicator */}
          {searchQuery.isLoading && debouncedQuery.length >= 1 && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <div className="w-5 h-5 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          )}
        </div>
        
        <select
          value={exchange}
          onChange={(e) => {
            setExchange(e.target.value);
            setSelectedSymbol(null);
          }}
          className="bg-gray-700 text-white px-4 py-3 rounded-lg"
        >
          <option value="NSE">NSE</option>
          <option value="BSE">BSE</option>
          <option value="US">US</option>
        </select>
      </div>

      {/* No results message */}
      {showResults && debouncedQuery.length >= 1 && searchQuery.data && searchQuery.data.length === 0 && !searchQuery.isLoading && (
        <div className="text-gray-400 text-center py-4 mb-4">
          No stocks found for "{debouncedQuery}". Try a different search term.
        </div>
      )}

      {/* Selected Stock Quote */}
      {quoteQuery.isLoading && selectedSymbol && (
        <div className="text-gray-400 text-center py-4">Loading quote for {selectedSymbol}...</div>
      )}

      {quoteQuery.data && quoteQuery.data.last_price > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-700 rounded-lg p-4"
        >
          <div className="flex justify-between items-start mb-3">
            <div>
              <h4 className="text-white font-bold text-lg">{quoteQuery.data.symbol}</h4>
              <p className="text-gray-400 text-sm">{quoteQuery.data.company_name}</p>
              <p className="text-gray-500 text-xs">{quoteQuery.data.sector}</p>
            </div>
            <div className="text-right">
              <p className="text-white font-bold text-xl">
                ₹{quoteQuery.data.last_price?.toLocaleString() || 0}
              </p>
              <p className={`text-sm ${(quoteQuery.data.day_change || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {(quoteQuery.data.day_change || 0) >= 0 ? '+' : ''}
                {(quoteQuery.data.day_change_percent || 0).toFixed(2)}%
              </p>
            </div>
          </div>

          <button
            onClick={() => onSelectStock(quoteQuery.data.symbol, quoteQuery.data)}
            className="w-full bg-green-600 hover:bg-green-500 text-white py-2 rounded-lg font-semibold transition-colors"
          >
            Buy {quoteQuery.data.symbol}
          </button>
        </motion.div>
      )}

      {quoteQuery.data && quoteQuery.data.error && (
        <div className="text-red-400 text-center py-4">
          Could not get quote for {selectedSymbol}. Please try again.
        </div>
      )}

      {/* Popular Stocks hint */}
      {!debouncedQuery && (
        <div className="mt-4 text-center">
          <p className="text-gray-500 text-sm mb-2">Popular searches:</p>
          <div className="flex flex-wrap gap-2 justify-center">
            {['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI'].map((symbol) => (
              <button
                key={symbol}
                onClick={() => {
                  setQuery(symbol);
                  setDebouncedQuery(symbol);
                  setSelectedSymbol(symbol);
                }}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-full text-sm transition-colors"
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const PaperTradingPanel: React.FC = () => {
  const [selectedStock, setSelectedStock] = useState<{ symbol: string; quote: Quote } | null>(null);
  const [tradeAction, setTradeAction] = useState<'buy' | 'sell'>('buy');
  const [showTradeModal, setShowTradeModal] = useState(false);
  const queryClient = useQueryClient();
  const { isDarkMode } = useThemeStore();

  const portfolioQuery = useQuery({
    queryKey: ['paper-portfolio'],
    queryFn: async () => {
      const response = await api.get('/api/paper/portfolio');
      return response.data;
    },
    refetchInterval: 30000
  });

  const transactionsQuery = useQuery({
    queryKey: ['paper-transactions'],
    queryFn: async () => {
      const response = await api.get('/api/paper/transactions?limit=10');
      return response.data.transactions;
    }
  });

  const resetMutation = useMutation({
    mutationFn: async () => {
      const response = await api.post('/api/paper/reset');
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paper-portfolio'] });
      queryClient.invalidateQueries({ queryKey: ['paper-transactions'] });
    }
  });

  const handleSelectStock = (symbol: string, quote: Quote) => {
    setSelectedStock({ symbol, quote });
    setTradeAction('buy');
    setShowTradeModal(true);
  };

  const handleSellFromHolding = (holding: any) => {
    setSelectedStock({
      symbol: holding.symbol,
      quote: {
        symbol: holding.symbol,
        exchange: holding.exchange,
        last_price: holding.last_price,
        day_change: holding.day_change || 0,
        day_change_percent: holding.day_change_percent || 0,
        company_name: holding.company_name || holding.symbol,
        sector: holding.sector || 'Unknown'
      }
    });
    setTradeAction('sell');
    setShowTradeModal(true);
  };

  const portfolio = portfolioQuery.data;

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className={`bg-gradient-to-r rounded-xl p-6 ${isDarkMode ? 'from-amber-900 to-orange-900' : 'from-amber-100 to-orange-100'}`}>
        <div className="flex justify-between items-center mb-4">
          <h2 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>📊 Paper Trading Portfolio</h2>
          <button
            onClick={() => resetMutation.mutate()}
            disabled={resetMutation.isPending}
            className={`text-sm underline ${isDarkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'}`}
          >
            {resetMutation.isPending ? 'Resetting...' : 'Reset Portfolio'}
          </button>
        </div>

        {portfolio && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Portfolio Value</p>
              <p className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                ₹{(portfolio.portfolio_value || 0).toLocaleString()}
              </p>
            </div>
            <div>
              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Cash Balance</p>
              <p className="text-2xl font-bold text-green-500">
                ₹{(portfolio.cash_balance || 0).toLocaleString()}
              </p>
            </div>
            <div>
              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Total P&L</p>
              <p className={`text-2xl font-bold ${(portfolio.total_pnl || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {(portfolio.total_pnl || 0) >= 0 ? '+' : ''}₹{(portfolio.total_pnl || 0).toLocaleString()}
              </p>
            </div>
            <div>
              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Holdings</p>
              <p className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                {portfolio.holdings_count || 0}
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Stock Search */}
        <StockSearch onSelectStock={handleSelectStock} />

        {/* Holdings */}
        <div className={`rounded-xl p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white shadow-sm border border-gray-200'}`}>
          <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>💼 Your Holdings</h3>

          {portfolio?.holdings?.length === 0 && (
            <p className={`text-center py-8 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              No holdings yet. Search and buy stocks to build your portfolio!
            </p>
          )}

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {portfolio?.holdings?.map((holding: any) => (
              <motion.div
                key={`${holding.symbol}:${holding.exchange}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-gray-700 rounded-lg p-4"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-white font-semibold">{holding.symbol}</h4>
                    <p className="text-gray-400 text-sm">
                      {holding.quantity} shares @ ₹{holding.average_price?.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-white font-semibold">
                      ₹{(holding.value || 0).toLocaleString()}
                    </p>
                    <p className={`text-sm ${(holding.pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {(holding.pnl || 0) >= 0 ? '+' : ''}₹{(holding.pnl || 0).toFixed(2)}
                      ({(holding.pnl_percent || 0).toFixed(2)}%)
                    </p>
                  </div>
                </div>

                <div className="flex space-x-2 mt-3">
                  <button
                    onClick={() => handleSelectStock(holding.symbol, {
                      symbol: holding.symbol,
                      exchange: holding.exchange,
                      last_price: holding.last_price,
                      day_change: holding.day_change || 0,
                      day_change_percent: holding.day_change_percent || 0,
                      company_name: holding.company_name || holding.symbol,
                      sector: holding.sector || 'Unknown'
                    })}
                    className="flex-1 bg-green-600/20 text-green-400 py-1 rounded text-sm hover:bg-green-600/30"
                  >
                    Buy More
                  </button>
                  <button
                    onClick={() => handleSellFromHolding(holding)}
                    className="flex-1 bg-red-600/20 text-red-400 py-1 rounded text-sm hover:bg-red-600/30"
                  >
                    Sell
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Transactions */}
      <div className={`rounded-xl p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white shadow-sm border border-gray-200'}`}>
        <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>📜 Recent Transactions</h3>

        {transactionsQuery.data?.length === 0 && (
          <p className={`text-center py-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No transactions yet</p>
        )}

        <div className="space-y-2">
          {transactionsQuery.data?.slice(0, 5).map((tx: any) => (
            <div
              key={tx.id}
              className="flex justify-between items-center bg-gray-700 rounded-lg p-3"
            >
              <div className="flex items-center space-x-3">
                <span className={`text-lg ${tx.action === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                  {tx.action === 'BUY' ? '🛒' : '💰'}
                </span>
                <div>
                  <p className="text-white font-medium">
                    {tx.action} {tx.quantity} {tx.symbol}
                  </p>
                  <p className="text-gray-400 text-xs">
                    {new Date(tx.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
              <p className="text-white font-medium">
                ₹{(tx.price * tx.quantity).toLocaleString()}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Trade Modal */}
      <AnimatePresence>
        {showTradeModal && selectedStock && (
          <TradeModal
            isOpen={showTradeModal}
            onClose={() => setShowTradeModal(false)}
            symbol={selectedStock.symbol}
            quote={selectedStock.quote}
            action={tradeAction}
            maxQuantity={
              tradeAction === 'sell'
                ? portfolio?.holdings?.find(
                    (h: any) => h.symbol === selectedStock.symbol
                  )?.quantity
                : undefined
            }
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default PaperTradingPanel;
