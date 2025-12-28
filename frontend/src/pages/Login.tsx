import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { TrendingUp, Shield, Brain, Zap, ArrowRight } from 'lucide-react'
import { authApi } from '../api'
import { useAuthStore } from '../store/authStore'

export default function Login() {
  const navigate = useNavigate()
  const { login } = useAuthStore()

  const handleZerodhaLogin = async () => {
    try {
      const response = await authApi.getLoginUrl()
      if (!response.login_url || response.login_url === '' || response.login_url === null) {
        alert('Zerodha login is not available in demo mode. Redirecting to demo...')
        await handleDemoLogin()
        return
      }
      window.location.href = response.login_url
    } catch (error) {
      alert('Failed to get Zerodha login URL. Try demo mode instead.')
      await handleDemoLogin()
    }
  }

  const handleDemoLogin = async () => {
    try {
      const response = await authApi.mockLogin()
      login(response.user_id, response.access_token)
      navigate('/dashboard')
    } catch (error) {
      alert('Demo login failed. Please try again.')
    }
  }

  const features = [
    {
      icon: Brain,
      title: 'Multi-Agent AI',
      description: 'Four specialized AI analysts debate to give you the best advice',
      color: 'text-amber-400',
    },
    {
      icon: TrendingUp,
      title: 'Portfolio Analysis',
      description: 'Deep analysis of your holdings with actionable insights',
      color: 'text-emerald-400',
    },
    {
      icon: Shield,
      title: 'Risk Management',
      description: 'Real-time monitoring and anomaly detection',
      color: 'text-red-400',
    },
    {
      icon: Zap,
      title: 'Real-time Updates',
      description: 'Stay informed with instant market alerts',
      color: 'text-yellow-400',
    },
  ]

  return (
    <div className="min-h-screen flex">
      {/* Left Panel - Hero */}
      <div className="flex-1 flex flex-col justify-center px-12 lg:px-20 bg-gradient-to-br from-[#0f0f23] to-[#1a1a2e]">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {/* Logo */}
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-xl overflow-hidden">
              <img src="/portabull.png" alt="Portabull" className="w-full h-full object-contain" />
            </div>
            <span className="text-3xl font-bold text-amber-400">Portabull</span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl lg:text-5xl font-bold text-white mb-6 leading-tight">
            AI-Powered
            <br />
            <span className="gradient-text">Portfolio Intelligence</span>
          </h1>

          <p className="text-gray-400 text-lg mb-10 max-w-xl">
            Connect your Zerodha account and let our multi-agent AI system analyze 
            your portfolio, monitor risks, and provide expert-level insights in real-time.
          </p>

          {/* Features Grid */}
          <div className="grid grid-cols-2 gap-4 mb-10">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 * index }}
                className="p-4 rounded-xl glass"
              >
                <feature.icon className={`w-6 h-6 ${feature.color} mb-2`} />
                <h3 className="text-white font-semibold mb-1">{feature.title}</h3>
                <p className="text-gray-400 text-sm">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Right Panel - Login */}
      <div className="flex-1 flex flex-col justify-center items-center px-12 bg-[#1a1a2e]">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="w-full max-w-md"
        >
          <div className="text-center mb-10">
            <h2 className="text-2xl font-bold text-white mb-2">Get Started</h2>
            <p className="text-gray-400">
              Connect your brokerage account to unlock AI insights
            </p>
          </div>

          {/* Login Buttons */}
          <div className="space-y-4">
            <button
              onClick={handleZerodhaLogin}
              className="w-full py-4 px-6 bg-[#387ED1] hover:bg-[#2a6cb8] text-white font-semibold rounded-xl flex items-center justify-center gap-3 transition-all duration-200 transform hover:scale-[1.02]"
            >
              <img 
                src="https://kite.zerodha.com/static/images/kite-logo.svg" 
                alt="Zerodha" 
                className="w-6 h-6"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none'
                }}
              />
              Continue with Zerodha
              <ArrowRight className="w-5 h-5" />
            </button>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-700"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-[#1a1a2e] text-gray-400">or</span>
              </div>
            </div>

            <button
              onClick={handleDemoLogin}
              className="w-full py-4 px-6 bg-transparent border border-gray-600 hover:border-amber-500 text-white font-semibold rounded-xl flex items-center justify-center gap-3 transition-all duration-200"
            >
              Try Demo Mode
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>

          {/* Disclaimer */}
          <p className="text-gray-500 text-xs text-center mt-8">
            By continuing, you agree to our Terms of Service and Privacy Policy.
            <br />
            We only request read-only access to your portfolio.
          </p>
        </motion.div>
      </div>
    </div>
  )
}
