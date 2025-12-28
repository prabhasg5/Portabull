import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface User {
  userId: string
  accessToken: string
}

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  login: (userId: string, accessToken: string) => void
  logout: () => void
  checkAuth: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,

      login: (userId: string, accessToken: string) => {
        set({
          user: { userId, accessToken },
          isAuthenticated: true,
        })
      },

      logout: () => {
        set({
          user: null,
          isAuthenticated: false,
        })
      },

      checkAuth: () => {
        // Check URL params for OAuth callback
        const params = new URLSearchParams(window.location.search)
        const token = params.get('token')
        const userId = params.get('user')

        if (token && userId) {
          get().login(userId, token)
          // Clean up URL
          window.history.replaceState({}, '', '/dashboard')
        }
      },
    }),
    {
      name: 'portabull-auth',
    }
  )
)
