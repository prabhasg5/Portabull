import { create } from 'zustand'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  agentPerspectives?: Record<string, string>
  deliberation?: string
  showDebate?: boolean
}

interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  showDebateView: boolean
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  setLoading: (loading: boolean) => void
  toggleDebateView: () => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isLoading: false,
  showDebateView: false,

  addMessage: (message) => {
    set((state) => ({
      messages: [
        ...state.messages,
        {
          ...message,
          id: Date.now().toString(),
          timestamp: new Date(),
        },
      ],
    }))
  },

  setLoading: (loading) => {
    set({ isLoading: loading })
  },

  toggleDebateView: () => {
    set((state) => ({ showDebateView: !state.showDebateView }))
  },

  clearMessages: () => {
    set({ messages: [] })
  },
}))
