/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Custom colors for agents
        'macro': '#4CAF50',
        'risk': '#F44336',
        'longterm': '#2196F3',
        'highreturns': '#FF9800',
        // Brand colors - Gold/Amber theme for Portabull
        'primary': '#f59e0b',
        'secondary': '#d97706',
        'accent': '#10b981',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
      }
    },
  },
  plugins: [],
}
