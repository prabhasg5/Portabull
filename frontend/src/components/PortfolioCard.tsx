import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { useThemeStore } from '../store/themeStore'

interface Holding {
  symbol?: string
  tradingsymbol?: string
  exchange: string
  quantity: number
  average_price: number
  last_price: number
  pnl: number
  pnl_percent: number
  value: number
  day_change: number
  day_change_percent: number
}

interface PortfolioCardProps {
  holding: Holding
}

export default function PortfolioCard({ holding }: PortfolioCardProps) {
  const isPositive = holding.pnl >= 0;
  const dayPositive = holding.day_change >= 0;
  const displaySymbol = holding.tradingsymbol || holding.symbol;
  const { isDarkMode } = useThemeStore()

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl p-4 border transition-all ${
        isDarkMode 
          ? 'bg-white/5 border-white/10 hover:border-amber-500/30'
          : 'bg-white border-gray-200 hover:border-amber-500/30 shadow-sm'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{displaySymbol}</h3>
          <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{holding.exchange}</p>
        </div>
        <div
          className={`flex items-center gap-1 px-2 py-1 rounded-lg text-sm ${
            dayPositive ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'
          }`}
        >
          {dayPositive ? (
            <TrendingUp className="w-3 h-3" />
          ) : (
            <TrendingDown className="w-3 h-3" />
          )}
          {dayPositive ? '+' : ''}{holding.day_change_percent.toFixed(2)}%
        </div>
      </div>

      {/* Price Info */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className={`text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Current Price</p>
          <p className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            ₹{holding.last_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </p>
        </div>
        <div>
          <p className={`text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Avg. Cost</p>
          <p className={`text-lg font-semibold ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            ₹{holding.average_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </p>
        </div>
      </div>

      {/* Holdings Info */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className={`text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Quantity</p>
          <p className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{holding.quantity}</p>
        </div>
        <div>
          <p className={`text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Value</p>
          <p className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            ₹{holding.value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </p>
        </div>
      </div>

      {/* P&L */}
      <div
        className={`rounded-lg p-3 ${
          isPositive ? 'bg-green-500/10' : 'bg-red-500/10'
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className={`text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Total P&L</p>
            <p
              className={`text-lg font-bold ${
                isPositive ? 'text-green-500' : 'text-red-500'
              }`}
            >
              {isPositive ? '+' : ''}₹{holding.pnl.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </p>
          </div>
          <div className="text-right">
            <p className={`text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Returns</p>
            <p
              className={`text-lg font-bold ${
                isPositive ? 'text-green-500' : 'text-red-500'
              }`}
            >
              {isPositive ? '+' : ''}{holding.pnl_percent.toFixed(2)}%
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
