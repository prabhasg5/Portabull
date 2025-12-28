import { Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import { useAuthStore } from './store/authStore'
import { FloatingAlerts } from './components/MarketStatus'

function App() {
  const { checkAuth } = useAuthStore()

  useEffect(() => {
    checkAuth()
  }, [checkAuth])

  return (
    <div className="min-h-screen bg-[#0f0f23]">
      {/* Floating alerts for real-time anomaly notifications */}
      <FloatingAlerts />
      
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </div>
  )
}

export default App
