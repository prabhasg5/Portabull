import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import {
  Send,
  Eye,
  EyeOff,
  Loader2,
  User,
  Bot,
  Sparkles,
  RotateCcw,
} from 'lucide-react'
import { chatApi } from '../api'
import { useChatStore, ChatMessage } from '../store/chatStore'
import { useThemeStore } from '../store/themeStore'

interface ChatInterfaceProps {
  portfolio?: any // Accepts paper trading portfolio shape
}

const SUGGESTED_QUESTIONS = [
  "Analyze my portfolio and suggest improvements",
  "What's the risk level of my current holdings?",
  "Should I rebalance my portfolio?",
  "Which stocks should I consider selling?",
  "What's the sector breakdown of my portfolio?",
]

export default function ChatInterface({ portfolio }: ChatInterfaceProps) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { isDarkMode } = useThemeStore()
  
  const { messages, isLoading, addMessage, setLoading, clearMessages } = useChatStore()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (messageText?: string) => {
    const text = messageText || input.trim()
    if (!text || isLoading) return

    // Add user message
    addMessage({
      role: 'user',
      content: text,
    })

    setInput('')
    setLoading(true)

    try {
      const response = await chatApi.sendMessage(text, true)
      
      addMessage({
        role: 'assistant',
        content: response.response,
        agentPerspectives: response.agent_perspectives,
        deliberation: response.deliberation,
      })
    } catch (error) {
      console.error('Chat error:', error)
      addMessage({
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
      })
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="h-full flex">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h2 className={`text-2xl font-bold mb-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                How can I help you today?
              </h2>
              <p className={`text-center max-w-md mb-8 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                I'm your AI portfolio advisor powered by multiple specialized agents.
                Ask me anything about your investments!
              </p>
              
              {/* Suggested Questions */}
              <div className="flex flex-wrap gap-2 max-w-2xl justify-center">
                {SUGGESTED_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    onClick={() => handleSend(question)}
                    className={`px-4 py-2 rounded-full text-sm transition-all ${
                      isDarkMode
                        ? 'bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10 hover:border-amber-500/50'
                        : 'bg-gray-100 border border-gray-200 text-gray-700 hover:bg-amber-50 hover:border-amber-500/50'
                    }`}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}
          
          {isLoading && (
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className={`rounded-2xl rounded-tl-none px-4 py-3 ${isDarkMode ? 'bg-white/5' : 'bg-gray-100'}`}>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className={`p-4 border-t ${isDarkMode ? 'border-gray-800' : 'border-gray-200'}`}>
          <div className="flex items-center gap-4 mb-3">
            {messages.length > 0 && (
              <button
                onClick={clearMessages}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all ${
                  isDarkMode 
                    ? 'bg-white/5 text-gray-400 border border-white/10 hover:border-white/20'
                    : 'bg-gray-100 text-gray-600 border border-gray-200 hover:border-gray-300'
                }`}
              >
                <RotateCcw className="w-4 h-4" />
                Clear Chat
              </button>
            )}
          </div>
          
          <div className="flex items-end gap-3">
            <div className={`flex-1 rounded-2xl border focus-within:border-amber-500/50 transition-all ${
              isDarkMode 
                ? 'bg-white/5 border-white/10'
                : 'bg-white border-gray-200'
            }`}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about your portfolio..."
                className={`w-full bg-transparent px-4 py-3 resize-none focus:outline-none ${
                  isDarkMode ? 'text-white placeholder-gray-500' : 'text-gray-900 placeholder-gray-400'
                }`}
                rows={1}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
              className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-white disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-all"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const [showPerspectives, setShowPerspectives] = useState(false)
  const isUser = message.role === 'user'
  const { isDarkMode } = useThemeStore()

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
    >
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser
            ? 'bg-amber-500'
            : 'bg-gradient-to-br from-amber-500 to-orange-600'
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </div>

      <div className={`flex-1 max-w-3xl ${isUser ? 'text-right' : ''}`}>
        <div
          className={`inline-block rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-amber-500 text-white rounded-tr-none'
              : isDarkMode 
                ? 'bg-white/5 text-gray-200 rounded-tl-none'
                : 'bg-gray-100 text-gray-800 rounded-tl-none'
          }`}
        >
          <ReactMarkdown className={`prose prose-sm max-w-none ${isDarkMode ? 'prose-invert' : ''}`}>
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Agent Perspectives Toggle */}
        {message.agentPerspectives && Object.keys(message.agentPerspectives).length > 0 && (
          <div className="mt-3">
            <button
              onClick={() => setShowPerspectives(!showPerspectives)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all ${
                showPerspectives
                  ? 'bg-amber-500/20 text-amber-500 border border-amber-500/50'
                  : isDarkMode
                    ? 'bg-white/5 text-gray-400 border border-white/10 hover:border-amber-500/30 hover:text-amber-400'
                    : 'bg-gray-100 text-gray-600 border border-gray-200 hover:border-amber-500/30 hover:text-amber-500'
              }`}
            >
              {showPerspectives ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              {showPerspectives ? 'Hide Agent Debate' : 'Show Agent Debate'}
            </button>

            <AnimatePresence>
              {showPerspectives && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-3 space-y-3"
                >
                  {Object.entries(message.agentPerspectives).map(([role, content]) => (
                    <AgentPerspectiveCard key={role} role={role} content={content} />
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        <p className="text-xs text-gray-500 mt-1">
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </motion.div>
  )
}

function AgentPerspectiveCard({ role, content }: { role: string; content: string }) {
  const agentConfig: Record<string, { color: string; name: string }> = {
    macro_analyst: { color: 'border-green-500 bg-green-500/10', name: 'Macro Analyst' },
    risk_manager: { color: 'border-red-500 bg-red-500/10', name: 'Risk Manager' },
    long_term_investor: { color: 'border-blue-500 bg-blue-500/10', name: 'Long-term Investor' },
    high_returns_specialist: { color: 'border-orange-500 bg-orange-500/10', name: 'High Returns Specialist' },
  }

  const config = agentConfig[role] || { color: 'border-gray-500', name: role }

  return (
    <div className={`border-l-2 ${config.color} rounded-r-lg px-4 py-3`}>
      <p className="text-xs font-semibold text-white mb-1">{config.name}</p>
      <p className="text-sm text-gray-300 line-clamp-4">{content}</p>
    </div>
  )
}
