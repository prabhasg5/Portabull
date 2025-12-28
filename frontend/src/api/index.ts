import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Portfolio API
export const portfolioApi = {
  getPortfolio: async () => {
    const response = await api.get('/api/portfolio')
    return response.data
  },

  getHoldings: async () => {
    const response = await api.get('/api/portfolio/holdings')
    return response.data
  },

  getPositions: async () => {
    const response = await api.get('/api/portfolio/positions')
    return response.data
  },

  analyzePortfolio: async () => {
    const response = await api.get('/api/portfolio/analyze')
    return response.data
  },
}

// Chat API
export const chatApi = {
  sendMessage: async (message: string, showDebate: boolean = false) => {
    const response = await api.post('/api/chat', {
      message,
      show_debate: showDebate,
    })
    return response.data
  },
}

// Agents API
export const agentsApi = {
  getAgents: async () => {
    const response = await api.get('/api/agents')
    return response.data
  },

  getStatus: async () => {
    const response = await api.get('/api/agents/status')
    return response.data
  },
}

// Alerts API
export const alertsApi = {
  getAlerts: async (limit: number = 10) => {
    const response = await api.get(`/api/alerts?limit=${limit}`)
    return response.data
  },

  acknowledgeAlert: async (alertId: string) => {
    const response = await api.post(`/api/alerts/${alertId}/acknowledge`)
    return response.data
  },
}

// Auth API
export const authApi = {
  getLoginUrl: async () => {
    const response = await api.get('/auth/login')
    return response.data
  },

  mockLogin: async () => {
    const response = await api.post('/auth/mock-login')
    return response.data
  },
}
