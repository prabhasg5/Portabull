import { motion } from 'framer-motion'
import { Brain, Shield, TrendingUp, Zap, CheckCircle } from 'lucide-react'

interface Agent {
  role: string
  name: string
  description: string
  color: string
}

interface AgentPanelProps {
  agents?: Agent[]
}

const agentIcons: Record<string, any> = {
  macro_analyst: Brain,
  risk_manager: Shield,
  long_term_investor: TrendingUp,
  high_returns_specialist: Zap,
}

const agentColors: Record<string, string> = {
  macro_analyst: 'from-green-500 to-emerald-600',
  risk_manager: 'from-red-500 to-rose-600',
  long_term_investor: 'from-blue-500 to-indigo-600',
  high_returns_specialist: 'from-orange-500 to-amber-600',
}

export default function AgentPanel({ agents }: AgentPanelProps) {
  const defaultAgents: Agent[] = [
    {
      role: 'macro_analyst',
      name: 'Macro Analyst',
      description: 'Focuses on macroeconomic trends, sector analysis, and market cycles',
      color: '#4CAF50',
    },
    {
      role: 'risk_manager',
      name: 'Risk Manager',
      description: 'Focuses on risk assessment, portfolio volatility, and downside protection',
      color: '#F44336',
    },
    {
      role: 'long_term_investor',
      name: 'Long-term Investor',
      description: 'Focuses on fundamental analysis, value investing, and wealth building',
      color: '#2196F3',
    },
    {
      role: 'high_returns_specialist',
      name: 'High Returns Specialist',
      description: 'Focuses on growth opportunities, momentum, and alpha generation',
      color: '#FF9800',
    },
  ]

  const displayAgents = agents || defaultAgents

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">
          Dragon Hatchling Architecture
        </h2>
        <p className="text-gray-400 max-w-2xl mx-auto">
          Our multi-agent system uses specialized AI analysts that debate and collaborate
          to provide you with comprehensive portfolio insights.
        </p>
      </div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {displayAgents.map((agent, index) => {
          const Icon = agentIcons[agent.role] || Brain
          const gradient = agentColors[agent.role] || 'from-gray-500 to-gray-600'

          return (
            <motion.div
              key={agent.role}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/5 rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all"
            >
              {/* Agent Icon */}
              <div
                className={`w-14 h-14 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center mb-4`}
              >
                <Icon className="w-7 h-7 text-white" />
              </div>

              {/* Agent Info */}
              <h3 className="text-xl font-bold text-white mb-2">{agent.name}</h3>
              <p className="text-gray-400 mb-4">{agent.description}</p>

              {/* Status */}
              <div className="flex items-center gap-2 text-green-400">
                <CheckCircle className="w-4 h-4" />
                <span className="text-sm">Active</span>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* How It Works */}
      <div className="mt-12 bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-bold text-white mb-4">How Dragon Hatchling Works</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4">
            <div className="w-10 h-10 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto mb-3 font-bold">
              1
            </div>
            <h4 className="text-white font-medium mb-1">Query</h4>
            <p className="text-gray-400 text-sm">You ask a question about your portfolio</p>
          </div>
          
          <div className="text-center p-4">
            <div className="w-10 h-10 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto mb-3 font-bold">
              2
            </div>
            <h4 className="text-white font-medium mb-1">Analyze</h4>
            <p className="text-gray-400 text-sm">Each agent analyzes from their perspective</p>
          </div>
          
          <div className="text-center p-4">
            <div className="w-10 h-10 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto mb-3 font-bold">
              3
            </div>
            <h4 className="text-white font-medium mb-1">Debate</h4>
            <p className="text-gray-400 text-sm">Agents debate disagreements</p>
          </div>
          
          <div className="text-center p-4">
            <div className="w-10 h-10 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto mb-3 font-bold">
              4
            </div>
            <h4 className="text-white font-medium mb-1">Consensus</h4>
            <p className="text-gray-400 text-sm">Unified recommendation is generated</p>
          </div>
        </div>
      </div>
    </div>
  )
}
