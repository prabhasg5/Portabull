"""
Portabull Configuration Module
Centralized configuration management for the AI Stock Market Agent
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application Settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Portabull"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Broker Provider: "paper", "zerodha", "mock"
    BROKER_PROVIDER: str = Field(default="paper", description="Broker provider to use")
    
    # Zerodha Kite Connect
    KITE_API_KEY: str = Field(default="", description="Zerodha Kite API Key")
    KITE_API_SECRET: str = Field(default="", description="Zerodha Kite API Secret")
    KITE_REDIRECT_URL: str = "http://localhost:8000/auth/callback"
    
    # LLaMA Configuration
    LLAMA_MODEL_PATH: str = Field(
        default="models/llama-2-7b-chat.gguf",
        description="Path to LLaMA model file"
    )
    LLAMA_API_URL: Optional[str] = Field(
        default=None,
        description="LLaMA API endpoint (if using hosted API)"
    )
    LLAMA_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for hosted LLaMA service"
    )
    
    # Together AI / Groq (Alternative LLaMA providers)
    TOGETHER_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # Vector Database
    VECTOR_DB_PATH: str = "data/vector_store"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Pathway Configuration
    PATHWAY_PERSISTENCE_PATH: str = "data/pathway_state"
    PATHWAY_CACHE_DIR: str = "data/cache"
    
    # Market Data
    MARKET_DATA_UPDATE_INTERVAL: int = 5  # seconds
    PORTFOLIO_SYNC_INTERVAL: int = 30  # seconds
    
    # Anomaly Detection
    ANOMALY_THRESHOLD_PERCENT: float = 5.0
    VOLATILITY_ALERT_THRESHOLD: float = 3.0
    
    # Authentication
    SECRET_KEY: str = Field(
        default="your-super-secret-key-change-in-production",
        description="JWT Secret Key"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Redis (for session management)
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/portabull.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Agent Role Configurations
AGENT_ROLES = {
    "macro_analyst": {
        "name": "Macro Analyst",
        "description": "Focuses on macroeconomic trends, sector analysis, and market cycles",
        "system_prompt": """You are an expert Macro Analyst AI. Your role is to:
- Analyze macroeconomic indicators (GDP, inflation, interest rates, employment)
- Evaluate sector rotations and market cycles
- Assess global economic trends and their impact on portfolios
- Provide insights on monetary and fiscal policies
- Consider geopolitical factors affecting markets

When analyzing a portfolio, focus on how macro factors might affect the holdings.
Be specific with data and provide actionable insights.""",
        "priority": 1,
        "color": "#4CAF50"
    },
    "risk_manager": {
        "name": "Risk Manager",
        "description": "Focuses on risk assessment, portfolio volatility, and downside protection",
        "system_prompt": """You are an expert Risk Manager AI. Your role is to:
- Calculate and analyze portfolio risk metrics (Beta, Sharpe Ratio, VaR, Max Drawdown)
- Identify concentration risks and correlation issues
- Suggest hedging strategies and risk mitigation
- Monitor position sizing and leverage
- Alert on excessive exposure to specific sectors/stocks

When analyzing a portfolio, focus on potential risks and how to protect capital.
Use quantitative risk measures and be conservative in your assessments.""",
        "priority": 2,
        "color": "#F44336"
    },
    "long_term_investor": {
        "name": "Long-term Investor",
        "description": "Focuses on fundamental analysis, value investing, and wealth building",
        "system_prompt": """You are an expert Long-term Investor AI (like Warren Buffett). Your role is to:
- Analyze company fundamentals (PE, PB, ROE, Debt/Equity, FCF)
- Evaluate competitive advantages (moats)
- Assess management quality and corporate governance
- Focus on intrinsic value and margin of safety
- Recommend buy-and-hold strategies for wealth creation

When analyzing a portfolio, focus on long-term value creation potential.
Ignore short-term volatility and focus on business quality.""",
        "priority": 3,
        "color": "#2196F3"
    },
    "high_returns_specialist": {
        "name": "High Returns Specialist",
        "description": "Focuses on growth opportunities, momentum, and alpha generation",
        "system_prompt": """You are an expert High Returns Specialist AI. Your role is to:
- Identify high-growth opportunities and emerging trends
- Analyze momentum indicators and technical patterns
- Spot potential multibaggers and turnaround stories
- Evaluate risk-reward ratios for aggressive positions
- Suggest tactical allocation shifts for higher returns

When analyzing a portfolio, focus on return enhancement opportunities.
Balance aggression with calculated risk-taking.""",
        "priority": 4,
        "color": "#FF9800"
    }
}

# Market Constants
INDIAN_MARKET_OPEN = "09:15"
INDIAN_MARKET_CLOSE = "15:30"
TRADING_HOLIDAYS_URL = "https://www.nseindia.com/api/holiday-master"

# Monitoring Thresholds
ALERT_THRESHOLDS = {
    "price_change_percent": 5.0,
    "volume_spike_multiplier": 3.0,
    "sector_rotation_threshold": 0.1,
    "correlation_breakdown_threshold": 0.3,
    "vix_spike_threshold": 25.0
}
