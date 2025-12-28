import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  MessageSquare,
  BarChart3,
  Bell,
  LogOut,
  ChevronRight,
  Wallet,
  Newspaper,
  Lightbulb,
  Sun,
  Moon,
} from 'lucide-react'
import { agentsApi, alertsApi } from '../api'
import { useAuthStore } from '../store/authStore'
import { useThemeStore } from '../store/themeStore'
import ChatInterface from '../components/ChatInterface'
import PortfolioCard from '../components/PortfolioCard'
import AlertPanel from '../components/AlertPanel'
import PaperTradingPanel from '../components/PaperTradingPanel'
import HypothesisPanel from '../components/HypothesisPanel'
import NewsFeed from '../components/NewsFeed'

type TabType = 'chat' | 'portfolio' | 'trade' | 'alerts' | 'news' | 'hypotheses'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabType>('chat')
  const { user, logout } = useAuthStore()
  const { isDarkMode, toggleTheme } = useThemeStore()

  // Fetch portfolio data
  const { data: portfolio, isLoading: portfolioLoading } = useQuery({
    queryKey: ['paper-portfolio'],
    queryFn: async () => {
      const response = await fetch('/api/paper/portfolio');
      return await response.json();
    },
    refetchInterval: 30000,
  })

  // Fetch agents
  const { data: agents } = useQuery({
    queryKey: ['agents'],
    queryFn: agentsApi.getAgents,
  })

  // Fetch alerts
  const { data: alerts } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => alertsApi.getAlerts(5),
    refetchInterval: 10000,
  })

  const tabs = [
    { id: 'chat' as TabType, label: 'Chat', icon: MessageSquare },
    { id: 'trade' as TabType, label: 'Trade', icon: Wallet },
    { id: 'news' as TabType, label: 'News', icon: Newspaper },
    { id: 'hypotheses' as TabType, label: 'Insights', icon: Lightbulb },
    { id: 'alerts' as TabType, label: 'Alerts', icon: Bell, badge: alerts?.length },
    { id: 'portfolio' as TabType, label: 'Portfolio', icon: BarChart3 },
  ]

  const summary = portfolio || {}
  const pnlPositive = (summary.total_pnl || 0) >= 0

  return (
    <div className={`min-h-screen flex ${isDarkMode ? 'bg-[#0f0f23]' : 'bg-gray-100'}`}>
      {/* Sidebar */}
      <div className={`w-48 flex flex-col items-center py-6 border-r ${isDarkMode ? 'bg-[#1a1a2e] border-gray-800' : 'bg-white border-gray-200'}`}>
        {/* Logo */}
        <div className="w-32 h-32 mb-6">
          <img src="/portabull.png" alt="Portabull" className="w-full h-full object-contain" />
        </div>

        {/* Navigation */}
        <nav className="flex-1 flex flex-col gap-2 w-full px-3">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative w-full h-11 rounded-xl flex items-center gap-3 px-3 transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-amber-500/20 text-amber-500'
                  : isDarkMode 
                    ? 'text-gray-400 hover:text-white hover:bg-white/5'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              <tab.icon className="w-5 h-5 flex-shrink-0" />
              <span className="text-sm font-medium">{tab.label}</span>
              {tab.badge && tab.badge > 0 && (
                <span className="absolute top-2 right-2 w-5 h-5 bg-red-500 rounded-full text-xs flex items-center justify-center text-white">
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </nav>

        {/* Bottom Actions */}
        <div className="flex flex-col gap-2 w-full px-3">
          <button
            onClick={toggleTheme}
            className={`w-full h-11 rounded-xl flex items-center gap-3 px-3 transition-all ${
              isDarkMode 
                ? 'text-gray-400 hover:text-amber-400 hover:bg-amber-500/10'
                : 'text-gray-600 hover:text-amber-500 hover:bg-amber-50'
            }`}
          >
            {isDarkMode ? <Sun className="w-5 h-5 flex-shrink-0" /> : <Moon className="w-5 h-5 flex-shrink-0" />}
            <span className="text-sm font-medium">{isDarkMode ? 'Light' : 'Dark'}</span>
          </button>
          <button
            onClick={logout}
            className={`w-full h-11 rounded-xl flex items-center gap-3 px-3 transition-all ${
              isDarkMode
                ? 'text-gray-400 hover:text-red-400 hover:bg-red-500/10'
                : 'text-gray-600 hover:text-red-500 hover:bg-red-50'
            }`}
          >
            <LogOut className="w-5 h-5 flex-shrink-0" />
            <span className="text-sm font-medium">Logout</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className={`h-16 border-b flex items-center justify-between px-6 ${isDarkMode ? 'bg-[#1a1a2e] border-gray-800' : 'bg-white border-gray-200'}`}>
          <div>
            <h1 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {tabs.find((t) => t.id === activeTab)?.label}
            </h1>
            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Welcome back, {user?.userId || 'Investor'}
            </p>
          </div>

          {/* Portfolio Summary */}
          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Portfolio Value</p>
              <p className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                ₹{(summary.total_current_value || 0).toLocaleString('en-IN', {
                  minimumFractionDigits: 2,
                })}
              </p>
            </div>
            <div
              className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
                pnlPositive ? 'bg-green-500/10' : 'bg-red-500/10'
              }`}
            >
              {pnlPositive ? (
                <TrendingUp className="w-5 h-5 text-green-400" />
              ) : (
                <TrendingDown className="w-5 h-5 text-red-400" />
              )}
              <div>
                <p
                  className={`font-bold ${
                    pnlPositive ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {pnlPositive ? '+' : ''}
                  ₹{(summary.total_pnl || 0).toLocaleString('en-IN', {
                    minimumFractionDigits: 2,
                  })}
                </p>
                <p
                  className={`text-xs ${
                    pnlPositive ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {pnlPositive ? '+' : ''}
                  {(summary.total_pnl_percent || 0).toFixed(2)}%
                </p>
              </div>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <main className={`flex-1 overflow-hidden ${isDarkMode ? '' : 'bg-gray-50'}`}>
          <AnimatePresence mode="wait">
            {activeTab === 'chat' && (
              <motion.div
                key="chat"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="h-full"
              >
                <ChatInterface portfolio={portfolio} />
              </motion.div>
            )}

            {activeTab === 'portfolio' && (
              <motion.div
                key="portfolio"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="h-full overflow-auto p-6"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {portfolio?.holdings?.map((holding: any) => (
                    <PortfolioCard key={holding.tradingsymbol || holding.symbol} holding={holding} />
                  ))}
                </div>
              </motion.div>
            )}

            {activeTab === 'trade' && (
              <motion.div
                key="trade"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="h-full overflow-auto p-6"
              >
                <PaperTradingPanel />
              </motion.div>
            )}

            {activeTab === 'alerts' && (
              <motion.div
                key="alerts"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="h-full overflow-auto p-6"
              >
                <AlertPanel alerts={alerts} />
              </motion.div>
            )}

            {activeTab === 'news' && (
              <motion.div
                key="news"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="h-full overflow-auto p-6"
              >
                <NewsFeed />
              </motion.div>
            )}

            {activeTab === 'hypotheses' && (
              <motion.div
                key="hypotheses"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="h-full overflow-auto p-6"
              >
                <HypothesisPanel />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}
