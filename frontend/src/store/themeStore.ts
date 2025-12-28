import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ThemeState {
  isDarkMode: boolean
  toggleTheme: () => void
  setDarkMode: (isDark: boolean) => void
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      isDarkMode: true,
      toggleTheme: () => set((state) => {
        const newMode = !state.isDarkMode
        // Update document class
        if (newMode) {
          document.documentElement.classList.add('dark')
          document.documentElement.classList.remove('light')
        } else {
          document.documentElement.classList.remove('dark')
          document.documentElement.classList.add('light')
        }
        return { isDarkMode: newMode }
      }),
      setDarkMode: (isDark) => set(() => {
        if (isDark) {
          document.documentElement.classList.add('dark')
          document.documentElement.classList.remove('light')
        } else {
          document.documentElement.classList.remove('dark')
          document.documentElement.classList.add('light')
        }
        return { isDarkMode: isDark }
      }),
    }),
    {
      name: 'portabull-theme',
      onRehydrateStorage: () => (state) => {
        // Apply theme on initial load
        if (state?.isDarkMode) {
          document.documentElement.classList.add('dark')
          document.documentElement.classList.remove('light')
        } else {
          document.documentElement.classList.remove('dark')
          document.documentElement.classList.add('light')
        }
      },
    }
  )
)
