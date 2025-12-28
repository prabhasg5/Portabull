# Portabull - Technical Architecture Document

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Dragon Hatchling Architecture](#2-dragon-hatchling-architecture)
3. [Pathway Dynamic RAG](#3-pathway-dynamic-rag)
4. [Zerodha Integration](#4-zerodha-integration)
5. [Real-time Monitoring](#5-real-time-monitoring)
6. [API Design](#6-api-design)
7. [Data Flow](#7-data-flow)
8. [Security Considerations](#8-security-considerations)
9. [Deployment Guide](#9-deployment-guide)

---

## 1. System Overview

### 1.1 High-Level Architecture

Portabull is a three-tier application:

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION TIER                         │
│                     (React + TypeScript + Vite)                  │
├─────────────────────────────────────────────────────────────────┤
│                        APPLICATION TIER                          │
│                    (FastAPI + Python 3.10+)                      │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Agents     │  │   RAG        │  │    Monitoring        │  │
│  │   Module     │  │   Module     │  │    Module            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                         DATA TIER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  ChromaDB    │  │   Redis      │  │    Pathway State     │  │
│  │  (Vectors)   │  │  (Sessions)  │  │    (Streaming)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | React 18 + Vite | User interface |
| Backend | FastAPI | REST API + WebSocket |
| Agents | Custom Python | Multi-agent reasoning |
| RAG | Pathway + ChromaDB | Dynamic knowledge retrieval |
| LLM | LLaMA 3.2 | Natural language processing |
| Broker API | Kite Connect | Portfolio data access |

### 1.3 Key Design Decisions

1. **Read-Only Access**: No trading capabilities - purely advisory
2. **Real-Time First**: WebSockets for live updates
3. **Multi-Agent Debate**: Diverse perspectives for better decisions
4. **Dynamic RAG**: Live data integration, not static embeddings
5. **Modular Architecture**: Easy to extend and maintain

---

## 2. Dragon Hatchling Architecture

### 2.1 Overview

The Dragon Hatchling architecture is a multi-agent system where specialized AI "hatchlings" (young dragons) collaborate and debate to provide comprehensive investment advice.

### 2.2 Agent Hierarchy

```
                    ┌─────────────────┐
                    │  ORCHESTRATOR   │
                    │  (Dragon Lord)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│    ANALYST    │  │    ANALYST    │  │    ANALYST    │
│   HATCHLINGS  │  │   HATCHLINGS  │  │   HATCHLINGS  │
└───────────────┘  └───────────────┘  └───────────────┘
```

### 2.3 Agent Roles

#### 🌍 Macro Analyst
```python
class MacroAnalystAgent(BaseAgent):
    """
    Focus Areas:
    - GDP, inflation, interest rates
    - Sector rotation patterns
    - Global economic trends
    - Monetary policy impact
    
    Example Analysis:
    "RBI's recent rate pause suggests banking stocks 
    may see margin pressure ease. Your 30% banking 
    allocation is well-positioned for this cycle."
    """
```

#### 🛡️ Risk Manager
```python
class RiskManagerAgent(BaseAgent):
    """
    Focus Areas:
    - Portfolio beta and volatility
    - Concentration risk
    - Downside scenarios
    - Hedging strategies
    
    Example Analysis:
    "Your portfolio has a beta of 1.3, meaning 30% 
    more volatile than Nifty. Consider adding 
    defensive stocks or gold for balance."
    """
```

#### 📈 Long-term Investor
```python
class LongTermInvestorAgent(BaseAgent):
    """
    Focus Areas:
    - Fundamental analysis (PE, ROE, FCF)
    - Competitive moats
    - Management quality
    - Intrinsic value
    
    Example Analysis:
    "HDFC Bank trades at 2.8x book with 17% ROE. 
    Despite short-term pressures, its deposit 
    franchise provides a durable moat."
    """
```

#### ⚡ High Returns Specialist
```python
class HighReturnsSpecialistAgent(BaseAgent):
    """
    Focus Areas:
    - Growth opportunities
    - Momentum indicators
    - Tactical allocation
    - Alpha generation
    
    Example Analysis:
    "The AI/tech sector shows strong momentum. 
    Consider increasing exposure to TCS and Infosys 
    for potential 20%+ upside in the current cycle."
    """
```

### 2.4 Debate Mechanism

The debate process follows these steps:

```
STEP 1: PARALLEL ANALYSIS
─────────────────────────
Each agent analyzes the query/portfolio independently
and produces their perspective.

STEP 2: ARGUMENT COLLECTION
───────────────────────────
Arguments are collected with:
- Content (the argument)
- Type (SUPPORT / OPPOSE / NEUTRAL)
- Confidence (0.0 - 1.0)
- Supporting data

STEP 3: DEBATE ROUNDS
─────────────────────
Agents see each other's arguments and can:
- Reinforce their position
- Counter opposing views
- Revise based on new information

STEP 4: CONSENSUS BUILDING
──────────────────────────
The Orchestrator:
- Weighs arguments by confidence
- Identifies majority positions
- Synthesizes final recommendation
```

### 2.5 Implementation Details

```python
@dataclass
class Argument:
    agent_role: AgentRole
    content: str
    argument_type: ArgumentType  # SUPPORT, OPPOSE, NEUTRAL
    confidence: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any]

@dataclass
class Debate:
    topic: str
    context: Dict[str, Any]
    arguments: List[Argument]
    consensus: Optional[str]
    final_recommendation: Optional[str]
```

### 2.6 Benefits of Multi-Agent Approach

| Single Agent | Multi-Agent (Dragon Hatchling) |
|--------------|-------------------------------|
| Single perspective | Four diverse perspectives |
| Biased to one strategy | Balanced advice |
| No internal checks | Agents challenge each other |
| Black-box reasoning | Transparent debate |
| Overconfident | Confidence-weighted |

---

## 3. Pathway Dynamic RAG

### 3.1 Why Dynamic RAG?

Traditional RAG:
```
Documents → Embed Once → Store → Retrieve (Static)
```

Pathway Dynamic RAG:
```
Live Data Streams → Continuous Processing → 
Real-time Indexing → Fresh Retrieval (Dynamic)
```

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   PATHWAY RAG ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  DATA        │    │  PROCESSING  │    │  INDEX       │  │
│  │  SOURCES     │───▶│  PIPELINE    │───▶│  UPDATE      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│        │                    │                    │          │
│        ▼                    ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ • Portfolio  │    │ • Chunking   │    │ • ChromaDB   │  │
│  │ • Market     │    │ • Embedding  │    │ • FAISS      │  │
│  │ • News       │    │ • Metadata   │    │ • In-memory  │  │
│  │ • Analyst    │    │ • Filtering  │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    STREAMING LAYER                    │   │
│  │        (Pathway handles real-time updates)            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Data Sources

| Source | Type | Update Frequency |
|--------|------|------------------|
| Portfolio | Structured | 30 seconds |
| Market Quotes | Structured | Real-time |
| News Articles | Unstructured | As available |
| Analyst Reports | Unstructured | Daily |
| Corporate Actions | Structured | As announced |

### 3.4 Implementation

```python
class PathwayRAGEngine:
    """
    Real-time RAG engine using Pathway framework
    
    Features:
    - Streaming document ingestion
    - Automatic re-indexing on updates
    - Multi-source data fusion
    """
    
    async def update_market_context(self, portfolio_data):
        """Update RAG with latest portfolio data"""
        
        # Format holdings for embedding
        holdings_text = self._format_holdings_for_rag(portfolio_data)
        
        # Add to dynamic index
        await self.add_document(
            content=holdings_text,
            doc_id="portfolio_holdings",
            source="zerodha",
            metadata={"updated": datetime.now().isoformat()}
        )
    
    async def retrieve(self, query: str, top_k: int = 5):
        """Retrieve relevant context for query"""
        
        query_embedding = await self._generate_embedding(query)
        
        # Search with latest indexed data
        results = self.vector_store.search(query_embedding, top_k)
        
        return results
```

### 3.5 Embedding Strategy

```
┌─────────────────────────────────────────────────────────┐
│               EMBEDDING PIPELINE                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Raw Text                                                │
│      │                                                   │
│      ▼                                                   │
│  ┌─────────────┐                                        │
│  │  Chunking   │ ← 512 tokens, 50 token overlap         │
│  └─────────────┘                                        │
│      │                                                   │
│      ▼                                                   │
│  ┌─────────────────────────────────────────┐            │
│  │  Sentence Transformers                   │            │
│  │  (all-MiniLM-L6-v2)                     │            │
│  │  384-dimensional embeddings              │            │
│  └─────────────────────────────────────────┘            │
│      │                                                   │
│      ▼                                                   │
│  ┌─────────────┐                                        │
│  │  ChromaDB   │ ← Persistent vector store              │
│  └─────────────┘                                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Zerodha Integration

### 4.1 Authentication Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  User   │     │ Portabull│     │ Zerodha │     │ Kite    │
│         │     │         │     │  Login  │     │  API    │
└────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
     │               │               │               │
     │  1. Click    │               │               │
     │  Login       │               │               │
     │──────────────>               │               │
     │               │               │               │
     │               │ 2. Redirect  │               │
     │               │ to Zerodha   │               │
     │<──────────────────────────────               │
     │               │               │               │
     │  3. Enter    │               │               │
     │  Credentials │               │               │
     │──────────────────────────────>               │
     │               │               │               │
     │               │ 4. Callback  │               │
     │               │ with token   │               │
     │               │<──────────────               │
     │               │               │               │
     │               │ 5. Exchange  │               │
     │               │ for access   │               │
     │               │ token        │               │
     │               │──────────────────────────────>
     │               │               │               │
     │               │ 6. Access    │               │
     │               │ token        │               │
     │               │<──────────────────────────────
     │               │               │               │
     │ 7. Dashboard │               │               │
     │<──────────────               │               │
```

### 4.2 API Permissions (Read-Only)

| Endpoint | Permission | Data |
|----------|------------|------|
| `/portfolio/holdings` | Read | Stock holdings |
| `/portfolio/positions` | Read | Day/Net positions |
| `/user/profile` | Read | User info |
| `/market/quote` | Read | Live prices |
| `/instruments` | Read | Instrument master |

**NOT Permitted**: Order placement, modification, cancellation

### 4.3 Data Models

```python
@dataclass
class Holding:
    tradingsymbol: str      # e.g., "RELIANCE"
    exchange: str           # "NSE" or "BSE"
    isin: str              # Unique identifier
    quantity: int          # Number of shares
    average_price: float   # Cost basis
    last_price: float      # Current price
    pnl: float             # Profit/Loss
    pnl_percent: float     # P&L percentage
    value: float           # Current value
    day_change: float      # Today's change
    day_change_percent: float
```

### 4.4 Rate Limiting

- API calls: 10 requests/second
- Historical data: 3 requests/second
- WebSocket: 1 connection per user
- Token validity: Until 6 AM next day

---

## 5. Real-time Monitoring

### 5.1 Monitoring Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    PORTFOLIO MONITOR                           │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │  DATA       │     │  ANOMALY    │     │  ALERT      │      │
│  │  COLLECTOR  │────▶│  DETECTOR   │────▶│  GENERATOR  │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
│        │                   │                   │               │
│        ▼                   ▼                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │ • Zerodha   │     │ • Price     │     │ • WebSocket │      │
│  │ • Market    │     │ • Volume    │     │ • Push      │      │
│  │ • News      │     │ • Pattern   │     │ • Email     │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### 5.2 Anomaly Detection Rules

| Alert Type | Trigger Condition | Severity |
|------------|-------------------|----------|
| Price Spike | ≥ 5% intraday move | Medium |
| Price Drop | ≤ -5% intraday move | Medium |
| Volume Anomaly | 3x average volume | Medium |
| Portfolio Drawdown | 5% from peak | High |
| Concentration Risk | Single stock > 30% | Medium |

### 5.3 Alert Lifecycle

```
1. DETECTION
   └─ Anomaly detected by monitoring engine

2. CLASSIFICATION
   └─ Severity assigned (low/medium/high/critical)

3. NOTIFICATION
   └─ Real-time push via WebSocket

4. DISCUSSION (for high severity)
   └─ Agents debate recommended action

5. USER ACTION
   └─ Acknowledge or act on alert

6. ARCHIVE
   └─ Stored for pattern analysis
```

### 5.4 Proactive Discussion

When a critical alert is detected:

```python
async def start_proactive_discussion(self, alert, portfolio_data):
    """
    Initiate agent debate about handling an alert
    """
    
    # Create debate topic from alert
    topic = f"Response to {alert.type}: {alert.title}"
    
    # Gather agent perspectives
    debate = await self.orchestrator._facilitate_debate(
        topic=topic,
        context={"alert": alert, "portfolio": portfolio_data}
    )
    
    # Push discussion to user
    await self.notify_user(debate)
```

---

## 6. API Design

### 6.1 RESTful Endpoints

```yaml
openapi: 3.0.0
info:
  title: Portabull API
  version: 1.0.0

paths:
  /api/portfolio:
    get:
      summary: Get portfolio summary
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PortfolioSummary'
  
  /api/portfolio/analyze:
    get:
      summary: Trigger comprehensive analysis
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UnifiedAnalysis'
  
  /api/chat:
    post:
      summary: Chat with AI agent
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                show_debate:
                  type: boolean
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatResponse'
```

### 6.2 WebSocket Protocol

```javascript
// Connection
ws://localhost:8000/ws/{user_id}

// Client → Server Messages
{
  "type": "chat",
  "content": "Analyze my portfolio",
  "show_debate": true
}

{
  "type": "subscribe_portfolio"
}

// Server → Client Messages
{
  "type": "chat_response",
  "data": {
    "answer": "...",
    "agent_perspectives": {...}
  }
}

{
  "type": "portfolio_update",
  "data": {...}
}

{
  "type": "alert",
  "data": {
    "id": "...",
    "severity": "high",
    "title": "..."
  }
}
```

---

## 7. Data Flow

### 7.1 Query Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query                                                  │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────┐                                            │
│  │ RAG Context │◀── Portfolio + Market Data                 │
│  │ Retrieval   │                                            │
│  └─────────────┘                                            │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────────────────────────────────┐        │
│  │           PARALLEL AGENT ANALYSIS                │        │
│  │                                                  │        │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐            │        │
│  │  │Macro│  │Risk │  │Long │  │High │            │        │
│  │  │     │  │     │  │Term │  │Ret  │            │        │
│  │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘            │        │
│  │     │        │        │        │                │        │
│  │     └────────┴────┬───┴────────┘                │        │
│  │                   │                              │        │
│  └───────────────────│──────────────────────────────┘        │
│                      ▼                                       │
│              ┌───────────────┐                              │
│              │   Synthesis   │                              │
│              │    Engine     │                              │
│              └───────┬───────┘                              │
│                      │                                       │
│                      ▼                                       │
│              ┌───────────────┐                              │
│              │    Response   │                              │
│              └───────────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Data Update Flow

```
Portfolio Changes
      │
      ▼
┌──────────────┐
│   Zerodha    │──── Every 30 seconds
│   Polling    │
└──────────────┘
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
┌──────────────┐                  ┌──────────────┐
│   RAG        │                  │   Anomaly    │
│   Update     │                  │   Detection  │
└──────────────┘                  └──────────────┘
      │                                  │
      ▼                                  ▼
┌──────────────┐                  ┌──────────────┐
│  ChromaDB    │                  │   Alerts     │
│  Re-index    │                  │   Generated  │
└──────────────┘                  └──────────────┘
      │                                  │
      └──────────────────────────────────┘
                     │
                     ▼
             ┌──────────────┐
             │   WebSocket  │
             │   Broadcast  │
             └──────────────┘
```

---

## 8. Security Considerations

### 8.1 Authentication & Authorization

| Layer | Mechanism |
|-------|-----------|
| User Auth | OAuth2 via Zerodha |
| API Auth | JWT tokens |
| Session | Redis with TTL |
| WebSocket | Token validation per connection |

### 8.2 Data Protection

- **In Transit**: TLS 1.3 encryption
- **At Rest**: Encrypted vector store
- **API Keys**: Environment variables, not hardcoded
- **No Storage**: Portfolio data not persisted long-term

### 8.3 Access Control

```python
# Read-only Zerodha permissions
ZERODHA_PERMISSIONS = {
    "portfolio.read": True,
    "market.read": True,
    "orders.place": False,  # DISABLED
    "orders.modify": False, # DISABLED
    "funds.transfer": False # DISABLED
}
```

---

## 9. Deployment Guide

### 9.1 Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

### 9.2 Production (Docker)

```dockerfile
# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - KITE_API_KEY=${KITE_API_KEY}
      - TOGETHER_API_KEY=${TOGETHER_API_KEY}
    
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### 9.3 Environment Variables

```bash
# Required
KITE_API_KEY=xxx
KITE_API_SECRET=xxx
SECRET_KEY=xxx

# LLaMA (choose one)
TOGETHER_API_KEY=xxx
# OR
GROQ_API_KEY=xxx
# OR
LLAMA_MODEL_PATH=./models/llama.gguf

# Optional
DEBUG=false
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
```

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| Dragon Hatchling | Multi-agent architecture with specialized AI analysts |
| RAG | Retrieval Augmented Generation - enhancing LLM with external data |
| Pathway | Real-time data processing framework |
| Kite Connect | Zerodha's trading API |
| Vector Store | Database for semantic embeddings |
| Orchestrator | Central coordinator for multi-agent system |
| Anomaly Detection | Automated identification of unusual patterns |
| WebSocket | Full-duplex communication protocol |

---

*Document Version: 1.0.0*
*Last Updated: December 2024*
