import { motion } from 'framer-motion'
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Volume2,
  Activity,
  CheckCircle,
  Clock,
} from 'lucide-react'
import { alertsApi } from '../api'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useThemeStore } from '../store/themeStore'

interface Alert {
  id: string
  type: string
  severity: string
  title: string
  description: string
  symbols: string[]
  action: string
  timestamp: string
  acknowledged?: boolean
}

interface AlertPanelProps {
  alerts?: Alert[]
}

const alertIcons: Record<string, any> = {
  price_spike: TrendingUp,
  price_drop: TrendingDown,
  volume_anomaly: Volume2,
  portfolio_drawdown: Activity,
  unusual_activity: AlertTriangle,
}

const severityColors: Record<string, string> = {
  info: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
  low: 'bg-green-500/10 border-green-500/30 text-green-400',
  medium: 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400',
  high: 'bg-orange-500/10 border-orange-500/30 text-orange-400',
  critical: 'bg-red-500/10 border-red-500/30 text-red-400',
}

export default function AlertPanel({ alerts }: AlertPanelProps) {
  const queryClient = useQueryClient()
  const { isDarkMode } = useThemeStore()

  const acknowledgeMutation = useMutation({
    mutationFn: (alertId: string) => alertsApi.acknowledgeAlert(alertId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] })
    },
  })

  if (!alerts || alerts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[400px] text-center">
        <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center mb-4">
          <CheckCircle className="w-8 h-8 text-green-500" />
        </div>
        <h3 className={`text-xl font-bold mb-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>All Clear!</h3>
        <p className={`max-w-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          No alerts at the moment. We're continuously monitoring your portfolio
          for unusual activity.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Portfolio Alerts</h2>
          <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Real-time monitoring and anomaly detection
          </p>
        </div>
        <div className={`flex items-center gap-2 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          <Clock className="w-4 h-4" />
          Updating every 10s
        </div>
      </div>

      {alerts.map((alert, index) => {
        const Icon = alertIcons[alert.type] || AlertTriangle
        const colorClass = severityColors[alert.severity] || severityColors.info

        return (
          <motion.div
            key={alert.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`rounded-xl p-4 border ${colorClass} ${
              alert.acknowledged ? 'opacity-50' : ''
            }`}
          >
            <div className="flex items-start gap-4">
              {/* Icon */}
              <div className="flex-shrink-0">
                <Icon className="w-6 h-6" />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-4 mb-2">
                  <h3 className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{alert.title}</h3>
                  <span className={`text-xs whitespace-nowrap ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </span>
                </div>

                <p className={`text-sm mb-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{alert.description}</p>

                {/* Symbols */}
                {alert.symbols && alert.symbols.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-3">
                    {alert.symbols.map((symbol) => (
                      <span
                        key={symbol}
                        className={`px-2 py-1 rounded text-xs ${isDarkMode ? 'bg-white/10 text-white' : 'bg-gray-100 text-gray-800'}`}
                      >
                        {symbol}
                      </span>
                    ))}
                  </div>
                )}

                {/* Action */}
                <div className="flex items-center justify-between">
                  <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    <strong>Action:</strong> {alert.action}
                  </p>

                  {!alert.acknowledged && (
                    <button
                      onClick={() => acknowledgeMutation.mutate(alert.id)}
                      disabled={acknowledgeMutation.isPending}
                      className={`px-3 py-1 rounded-lg text-sm transition-all ${
                        isDarkMode 
                          ? 'bg-white/10 text-white hover:bg-white/20'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      Acknowledge
                    </button>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}
