"""
Portabull - Main FastAPI Application
AI Stock Market Agent with Dragon Hatchling Architecture

This is the main entry point for the Portabull backend.
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from loguru import logger
import sys
import json

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("logs/portabull.log", rotation="10 MB", retention="7 days", level="DEBUG")

# Import modules
from config import get_settings, AGENT_ROLES
from integrations.zerodha import create_zerodha_client, ZerodhaCredentials
from integrations.llm_client import create_llm_client
from agents.orchestrator import AgentOrchestrator
from rag.pathway_rag import create_rag_engine
from monitoring.portfolio_monitor import PortfolioMonitor

# Initialize FastAPI app
app = FastAPI(
    title="Portabull",
    description="AI-Powered Stock Market Agent with Multi-Agent Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global state (in production, use Redis or database)
class AppState:
    """Application state container"""
    zerodha_client = None
    llm_client = None
    orchestrator = None
    rag_engine = None
    portfolio_monitor = None
    realtime_monitor = None  # Real-time market monitor
    user_sessions: Dict[str, ZerodhaCredentials] = {}
    websocket_connections: Dict[str, WebSocket] = {}

state = AppState()


# ============================================
# Pydantic Models
# ============================================

class LoginResponse(BaseModel):
    login_url: str
    message: str

class AuthCallbackRequest(BaseModel):
    request_token: str

class AuthResponse(BaseModel):
    access_token: str
    user_id: str
    message: str

class ChatMessage(BaseModel):
    message: str
    show_debate: bool = False
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    agent_perspectives: Optional[Dict[str, str]] = None
    deliberation: Optional[str] = None
    timestamp: str

class PortfolioAnalysisResponse(BaseModel):
    summary: Dict[str, Any]
    agent_analyses: Dict[str, Any]
    consensus_points: List[str]
    disagreement_points: List[str]
    recommendations: List[str]
    priority_actions: List[str]
    confidence_score: float

class AlertResponse(BaseModel):
    id: str
    type: str
    severity: str
    title: str
    description: str
    symbols: List[str]
    action: str
    timestamp: str

class AgentInfo(BaseModel):
    role: str
    name: str
    description: str
    color: str


# ============================================
# Startup and Shutdown
# ============================================

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    settings = get_settings()
    
    logger.info("🚀 Starting Portabull...")
    
    # Initialize broker client based on BROKER_PROVIDER setting
    broker_provider = getattr(settings, 'BROKER_PROVIDER', 'paper')
    
    if broker_provider == "paper":
        from integrations.paper_broker import PaperBrokerClient
        state.zerodha_client = PaperBrokerClient(user_id="default")
        logger.info("📊 Using Paper Trading mode")
    elif broker_provider == "zerodha" and settings.KITE_API_KEY and settings.KITE_API_KEY != "your_kite_api_key":
        state.zerodha_client = create_zerodha_client(
            api_key=settings.KITE_API_KEY,
            api_secret=settings.KITE_API_SECRET,
            use_mock=False
        )
        logger.info("📊 Using Zerodha Live client")
    else:
        # Default to paper trading
        from integrations.paper_broker import PaperBrokerClient
        state.zerodha_client = PaperBrokerClient(user_id="default")
        logger.info("📊 Using Paper Trading mode (default)")
    
    # Initialize LLM client
    if settings.TOGETHER_API_KEY:
        state.llm_client = create_llm_client(
            provider="together",
            api_key=settings.TOGETHER_API_KEY
        )
    elif settings.GROQ_API_KEY:
        state.llm_client = create_llm_client(
            provider="groq",
            api_key=settings.GROQ_API_KEY
        )
    else:
        state.llm_client = create_llm_client(provider="mock")
    
    # Initialize orchestrator
    state.orchestrator = AgentOrchestrator(llm_client=state.llm_client)
    
    # Initialize RAG engine
    state.rag_engine = create_rag_engine()
    
    # Initialize Real-time Market Monitor
    from rag.realtime_monitor import get_realtime_monitor
    state.realtime_monitor = get_realtime_monitor(
        refresh_interval=60,  # Refresh every 60 seconds
        cache_ttl=60,         # Cache valid for 60 seconds
        rag_engine=state.rag_engine
    )
    
    # Get portfolio symbols and start monitoring
    try:
        portfolio = await state.zerodha_client.get_portfolio_summary()
        symbols = []
        for h in portfolio.get("holdings", []):
            symbol = h.get("symbol") or h.get("tradingsymbol", "")
            symbol = symbol.replace(".NS", "").replace(".BO", "").strip()
            if symbol:
                symbols.append(symbol)
        
        if symbols:
            state.realtime_monitor.add_symbols(symbols)
            await state.realtime_monitor.start()
            logger.info(f"🔄 Real-time monitor started for {len(symbols)} symbols")
    except Exception as e:
        logger.warning(f"Could not start real-time monitor: {e}")
    
    logger.info("✅ Portabull started successfully!")
    logger.info(f"🤖 Using {type(state.llm_client).__name__} for LLM")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down Portabull...")
    
    if state.portfolio_monitor:
        await state.portfolio_monitor.stop()
    
    # Stop real-time monitor
    if state.realtime_monitor:
        await state.realtime_monitor.stop()
    
    # Close WebSocket connections
    for ws in state.websocket_connections.values():
        await ws.close()
    
    logger.info("Portabull shutdown complete")


# ============================================
# Authentication Routes
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Portabull",
        "version": "1.0.0",
        "description": "AI-Powered Stock Market Agent",
        "status": "running"
    }


@app.get("/auth/login", response_model=LoginResponse)
async def login():
    """Get Zerodha login URL"""
    login_url = state.zerodha_client.get_login_url()
    return LoginResponse(
        login_url=login_url,
        message="Redirect user to login URL"
    )


@app.get("/auth/callback")
async def auth_callback(request_token: str):
    """Handle Zerodha OAuth callback"""
    try:
        credentials = await state.zerodha_client.authenticate(request_token)
        
        # Store session
        state.user_sessions[credentials.user_id] = credentials
        
        # Start portfolio monitoring for this user
        if not state.portfolio_monitor:
            state.portfolio_monitor = PortfolioMonitor(
                zerodha_client=state.zerodha_client,
                orchestrator=state.orchestrator,
                on_alert=lambda alert: asyncio.create_task(broadcast_alert(alert))
            )
            asyncio.create_task(state.portfolio_monitor.start())
        
        # Redirect to frontend
        return RedirectResponse(
            url=f"http://localhost:3000/dashboard?token={credentials.access_token}&user={credentials.user_id}"
        )
        
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/mock-login", response_model=AuthResponse)
async def mock_login():
    """Mock login for development"""
    credentials = await state.zerodha_client.authenticate("mock_token")
    state.user_sessions[credentials.user_id] = credentials
    
    return AuthResponse(
        access_token=credentials.access_token,
        user_id=credentials.user_id,
        message="Mock login successful"
    )


# ============================================
# Portfolio Routes
# ============================================

@app.get("/api/portfolio")
async def get_portfolio():
    """Get user's portfolio summary"""
    try:
        portfolio = await state.zerodha_client.get_portfolio_summary()
        
        # Update RAG context with latest portfolio
        await state.rag_engine.update_market_context(portfolio)
        
        return portfolio
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/holdings")
async def get_holdings():
    """Get user's stock holdings"""
    try:
        holdings = await state.zerodha_client.get_holdings()
        return {"holdings": [h.to_dict() for h in holdings]}
    except Exception as e:
        logger.error(f"Failed to get holdings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/positions")
async def get_positions():
    """Get user's trading positions"""
    try:
        positions = await state.zerodha_client.get_positions()
        return {
            "net": [p.to_dict() for p in positions["net"]],
            "day": [p.to_dict() for p in positions["day"]]
        }
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/analyze", response_model=PortfolioAnalysisResponse)
async def analyze_portfolio():
    """
    Perform comprehensive portfolio analysis using all agents
    
    This triggers the Dragon Hatchling architecture to:
    1. Analyze portfolio from each agent's perspective
    2. Identify consensus and disagreements
    3. Debate contentious points
    4. Synthesize recommendations
    """
    try:
        # Get portfolio data
        portfolio = await state.zerodha_client.get_portfolio_summary()
        
        # Get market context from RAG
        market_context = {
            "timestamp": datetime.now().isoformat(),
            "market_status": "open",  # Would check actual market hours
        }
        
        # Run comprehensive analysis
        analysis = await state.orchestrator.analyze_portfolio(
            portfolio_data=portfolio,
            market_context=market_context
        )
        
        return PortfolioAnalysisResponse(
            summary=analysis.portfolio_summary,
            agent_analyses={
                role: {
                    "summary": a.summary,
                    "insights": a.insights,
                    "recommendations": a.recommendations,
                    "risk_factors": a.risk_factors,
                    "confidence": a.confidence
                }
                for role, a in analysis.agent_analyses.items()
            },
            consensus_points=analysis.consensus_points,
            disagreement_points=analysis.disagreement_points,
            recommendations=analysis.final_recommendations,
            priority_actions=analysis.priority_actions,
            confidence_score=analysis.confidence_score
        )
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Chat Routes
# ============================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Chat with the AI agent using Dynamic RAG
    
    The agent will:
    1. Fetch real-time market data for portfolio stocks (Dynamic RAG)
    2. Gather perspectives from all specialist agents
    3. Optionally show the deliberation process
    4. Synthesize a unified response with real market context
    """
    try:
        # Get current portfolio context
        portfolio = await state.zerodha_client.get_portfolio_summary()
        
        # ============================================
        # DYNAMIC RAG: Fetch real-time market data
        # ============================================
        # Extract symbols from portfolio for market data fetch
        # Supports both 'symbol' (paper broker) and 'tradingsymbol' (Zerodha) formats
        portfolio_symbols = []
        if portfolio.get("holdings"):
            for h in portfolio["holdings"]:
                symbol = h.get("symbol") or h.get("tradingsymbol", "")
                symbol = symbol.replace(".NS", "").replace(".BO", "").strip()
                if symbol:
                    portfolio_symbols.append(symbol)
        
        if portfolio.get("positions"):
            for p in portfolio["positions"]:
                symbol = p.get("symbol") or p.get("tradingsymbol", "")
                symbol = symbol.replace(".NS", "").replace(".BO", "").strip()
                if symbol:
                    portfolio_symbols.append(symbol)
        
        # Remove duplicates
        portfolio_symbols = list(set(portfolio_symbols))
        
        logger.info(f"Portfolio symbols for Dynamic RAG: {portfolio_symbols}")
        
        # Fetch real market data for portfolio stocks (Dynamic RAG)
        if portfolio_symbols:
            logger.info(f"Dynamic RAG: Updating market data for {portfolio_symbols}")
            try:
                await state.rag_engine.update_dynamic_market_data(
                    symbols=portfolio_symbols,
                    exchange="NSE"  # Default to NSE for Indian stocks
                )
            except Exception as e:
                logger.error(f"Dynamic RAG update failed: {e}")
                # Continue even if RAG update fails
        
        # Also update portfolio holdings/positions in RAG
        await state.rag_engine.update_market_context(portfolio)
        
        # Get relevant documents from RAG (now includes real market data!)
        relevant_docs = await state.rag_engine.retrieve(
            query=message.message,
            top_k=8  # Increased to get more context from dynamic data
        )
        
        # Build context with dynamic market intelligence
        context = {
            "portfolio": portfolio,
            "market_context": message.context or {},
            "relevant_docs": [doc.content for doc in relevant_docs],
            "rag_info": {
                "documents_searched": len(state.rag_engine.documents),
                "relevant_found": len(relevant_docs)
            }
        }
        
        logger.info(f"RAG Context: {len(relevant_docs)} relevant documents from {len(state.rag_engine.documents)} total")
        
        # Get response from orchestrator
        result = await state.orchestrator.answer_query(
            query=message.message,
            context=context,
            show_debate=message.show_debate
        )
        
        return ChatResponse(
            response=result["answer"],
            agent_perspectives=result.get("agent_perspectives"),
            deliberation=result.get("deliberation"),
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Agent Routes
# ============================================

@app.get("/api/agents", response_model=List[AgentInfo])
async def get_agents():
    """Get information about all available agents"""
    return [
        AgentInfo(
            role=role,
            name=config["name"],
            description=config["description"],
            color=config["color"]
        )
        for role, config in AGENT_ROLES.items()
    ]


@app.get("/api/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    return state.orchestrator.get_agent_status()


# ============================================
# RAG Routes
# ============================================

@app.get("/api/rag/status")
async def get_rag_status():
    """
    Get the current status of the Dynamic RAG system
    
    Shows:
    - Total documents indexed
    - Document types breakdown
    - Last update timestamp
    """
    documents = state.rag_engine.documents
    
    # Count by type
    type_counts = {}
    symbols = set()
    latest_update = None
    
    for doc_id, doc in documents.items():
        doc_type = doc.metadata.get("type", "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        if doc.metadata.get("symbol"):
            symbols.add(doc.metadata["symbol"])
        
        if doc.timestamp and (latest_update is None or doc.timestamp > latest_update):
            latest_update = doc.timestamp
    
    return {
        "total_documents": len(documents),
        "document_types": type_counts,
        "symbols_indexed": list(symbols),
        "last_update": latest_update.isoformat() if latest_update else None,
        "rag_framework": "Pathway",
        "embedding_model": state.rag_engine.embedding_model
    }


@app.post("/api/rag/refresh")
async def refresh_rag(symbols: List[str] = None):
    """
    Manually refresh the RAG with latest market data
    
    If symbols not provided, uses current portfolio symbols
    """
    try:
        if not symbols:
            # Get symbols from portfolio
            portfolio = await state.zerodha_client.get_portfolio_summary()
            symbols = []
            if portfolio.get("holdings"):
                symbols.extend([
                    h.get("tradingsymbol", "").replace(".NS", "").replace(".BO", "")
                    for h in portfolio["holdings"]
                    if h.get("tradingsymbol")
                ])
            if portfolio.get("positions"):
                symbols.extend([
                    p.get("tradingsymbol", "").replace(".NS", "").replace(".BO", "")
                    for p in portfolio["positions"]
                    if p.get("tradingsymbol")
                ])
            symbols = list(set(symbols))
        
        if symbols:
            result = await state.rag_engine.update_dynamic_market_data(
                symbols=symbols,
                exchange="NSE"
            )
            return {
                "success": True,
                "message": f"RAG refreshed with data for {result['stocks_updated']} stocks",
                "details": result
            }
        else:
            return {
                "success": False,
                "message": "No symbols found in portfolio to refresh"
            }
            
    except Exception as e:
        logger.error(f"RAG refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/documents")
async def get_rag_documents(doc_type: str = None, symbol: str = None, limit: int = 10):
    """
    Get documents stored in the RAG system
    
    Optional filters:
    - doc_type: Filter by document type (stock_data, news, market_overview, etc.)
    - symbol: Filter by stock symbol
    - limit: Max number of documents to return
    """
    documents = state.rag_engine.documents
    
    results = []
    for doc_id, doc in documents.items():
        # Apply filters
        if doc_type and doc.metadata.get("type") != doc_type:
            continue
        if symbol and doc.metadata.get("symbol") != symbol:
            continue
        
        results.append({
            "id": doc_id,
            "content_preview": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
            "metadata": doc.metadata,
            "timestamp": doc.timestamp.isoformat() if doc.timestamp else None
        })
        
        if len(results) >= limit:
            break
    
    return {
        "total_matching": len(results),
        "documents": results
    }


# ============================================
# Real-time Monitor Routes
# ============================================

@app.get("/api/monitor/status")
async def get_monitor_status():
    """
    Get the status of the real-time market monitor
    
    Shows:
    - Running state
    - Watched symbols
    - Refresh count
    - Cache statistics
    - Active alerts count
    """
    if not state.realtime_monitor:
        return {"error": "Real-time monitor not initialized"}
    
    return state.realtime_monitor.get_status()


@app.post("/api/monitor/start")
async def start_monitor(refresh_interval: int = 60):
    """Start the real-time market monitor"""
    if not state.realtime_monitor:
        from rag.realtime_monitor import get_realtime_monitor
        state.realtime_monitor = get_realtime_monitor(
            refresh_interval=refresh_interval,
            cache_ttl=60,
            rag_engine=state.rag_engine
        )
    
    await state.realtime_monitor.start()
    return {"success": True, "message": "Monitor started", "status": state.realtime_monitor.get_status()}


@app.post("/api/monitor/stop")
async def stop_monitor():
    """Stop the real-time market monitor"""
    if state.realtime_monitor:
        await state.realtime_monitor.stop()
        return {"success": True, "message": "Monitor stopped"}
    return {"success": False, "message": "Monitor not running"}


@app.post("/api/monitor/watch")
async def add_watch_symbols(symbols: List[str]):
    """Add symbols to the watch list"""
    if not state.realtime_monitor:
        return {"error": "Real-time monitor not initialized"}
    
    state.realtime_monitor.add_symbols(symbols)
    return {
        "success": True,
        "message": f"Added {len(symbols)} symbols to watch list",
        "watching": list(state.realtime_monitor._watched_symbols)
    }


@app.delete("/api/monitor/watch")
async def remove_watch_symbols(symbols: List[str]):
    """Remove symbols from the watch list"""
    if not state.realtime_monitor:
        return {"error": "Real-time monitor not initialized"}
    
    state.realtime_monitor.remove_symbols(symbols)
    return {
        "success": True,
        "message": f"Removed {len(symbols)} symbols from watch list",
        "watching": list(state.realtime_monitor._watched_symbols)
    }


@app.get("/api/monitor/anomalies")
async def get_anomalies(unacknowledged_only: bool = False, limit: int = 50):
    """Get detected market anomalies"""
    if not state.realtime_monitor:
        return {"error": "Real-time monitor not initialized"}
    
    alerts = state.realtime_monitor.anomaly_detector.get_alerts(
        unacknowledged_only=unacknowledged_only,
        limit=limit
    )
    
    return {
        "total": len(alerts),
        "alerts": [a.to_dict() for a in alerts]
    }


@app.post("/api/monitor/anomalies/{alert_id}/acknowledge")
async def acknowledge_anomaly(alert_id: str):
    """Acknowledge an anomaly alert"""
    if not state.realtime_monitor:
        return {"error": "Real-time monitor not initialized"}
    
    success = state.realtime_monitor.anomaly_detector.acknowledge_alert(alert_id)
    return {"success": success}


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if not state.realtime_monitor:
        return {"error": "Real-time monitor not initialized"}
    
    return state.realtime_monitor.cache.get_stats()


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    if not state.realtime_monitor:
        return {"error": "Real-time monitor not initialized"}
    
    await state.realtime_monitor.cache.clear()
    return {"success": True, "message": "Cache cleared"}


# ============================================
# Pathway Streaming Pipeline Routes
# ============================================

@app.get("/api/streaming/status")
async def get_streaming_status():
    """
    Get status of the Pathway streaming pipeline
    
    Shows:
    - Total documents in pipeline
    - Documents by type
    - Recent document events
    - Pipeline statistics
    """
    from rag.pathway_streaming import get_streaming_pipeline
    pipeline = get_streaming_pipeline()
    return pipeline.get_stats()


@app.get("/api/streaming/documents")
async def get_streaming_documents(doc_type: Optional[str] = None, limit: int = 50):
    """Get documents from the streaming pipeline"""
    from rag.pathway_streaming import get_streaming_pipeline
    pipeline = get_streaming_pipeline()
    
    docs = pipeline.get_all_documents()
    
    if doc_type:
        docs = [d for d in docs if d.get("doc_type") == doc_type]
    
    return {
        "total": len(docs),
        "documents": docs[:limit]
    }


@app.post("/api/streaming/document")
async def add_streaming_document(
    doc_id: str,
    content: str,
    doc_type: str,
    symbol: str = "",
    source: str = "api"
):
    """
    Add a document to the streaming pipeline
    
    This demonstrates Pathway's live document indexing -
    the document will be immediately available for retrieval
    """
    from rag.pathway_streaming import get_streaming_pipeline
    pipeline = get_streaming_pipeline()
    
    event = await pipeline.add_document(
        doc_id=doc_id,
        content=content,
        doc_type=doc_type,
        symbol=symbol,
        source=source
    )
    
    return {
        "success": True,
        "event": event.to_dict()
    }


@app.get("/api/streaming/events")
async def get_streaming_events(limit: int = 20):
    """Get recent document events from the streaming pipeline"""
    from rag.pathway_streaming import get_document_broadcaster
    broadcaster = get_document_broadcaster()
    
    events = broadcaster.get_recent_events(limit)
    return {
        "total": len(events),
        "events": [e.to_dict() for e in events]
    }


@app.post("/api/streaming/search")
async def search_streaming_pipeline(
    query: str,
    top_k: int = 5,
    doc_type: Optional[str] = None,
    symbol: Optional[str] = None
):
    """Search documents in the streaming pipeline using vector similarity"""
    from rag.pathway_streaming import get_streaming_pipeline
    pipeline = get_streaming_pipeline()
    
    results = await pipeline.retrieve(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        symbol=symbol
    )
    
    return {
        "query": query,
        "results_count": len(results),
        "results": [
            {k: v for k, v in r.items() if k != "embedding"}
            for r in results
        ]
    }


# ============================================
# Hypothesis Generator Routes
# ============================================

@app.get("/api/hypotheses")
async def get_hypotheses():
    """
    Get all active investment hypotheses
    
    Hypotheses are automatically generated based on:
    - Portfolio concentration
    - Performance patterns
    - Sector allocation
    - Risk factors
    - Market opportunities
    """
    from rag.hypothesis_generator import get_hypothesis_generator
    generator = get_hypothesis_generator(state.llm_client)
    
    hypotheses = generator.get_active_hypotheses()
    return {
        "total": len(hypotheses),
        "hypotheses": [h.to_dict() for h in hypotheses]
    }


@app.post("/api/hypotheses/generate")
async def generate_hypotheses():
    """
    Generate new investment hypotheses based on current portfolio
    
    This analyzes your portfolio and market conditions to automatically
    generate actionable investment hypotheses.
    """
    from rag.hypothesis_generator import get_hypothesis_generator
    from rag.pathway_streaming import get_streaming_pipeline
    
    generator = get_hypothesis_generator(state.llm_client)
    pipeline = get_streaming_pipeline()
    
    # Get portfolio data
    portfolio = await state.zerodha_client.get_portfolio_summary()
    
    # Get market data (if available)
    market_data = {}
    if state.realtime_monitor:
        market_data = state.realtime_monitor.get_status()
    
    # Generate hypotheses
    hypotheses = await generator.generate_hypotheses(portfolio, market_data)
    
    # Add hypotheses to streaming pipeline for RAG
    for hyp in hypotheses:
        await pipeline.add_document(
            doc_id=f"hypothesis_{hyp.hypothesis_id}",
            content=f"""
INVESTMENT HYPOTHESIS: {hyp.title}

Type: {hyp.hypothesis_type}
Confidence: {hyp.confidence * 100:.0f}%
Time Horizon: {hyp.time_horizon}
Affected Stocks: {', '.join(hyp.affected_symbols) if hyp.affected_symbols else 'Portfolio-wide'}

Description:
{hyp.description}

Supporting Evidence:
{chr(10).join('• ' + e for e in hyp.supporting_evidence)}

Potential Actions:
{chr(10).join('• ' + a for a in hyp.potential_actions)}

Risk Factors:
{chr(10).join('• ' + r for r in hyp.risk_factors)}
""",
            doc_type="hypothesis",
            symbol=hyp.affected_symbols[0] if hyp.affected_symbols else "",
            source="hypothesis_generator"
        )
    
    return {
        "success": True,
        "generated": len(hypotheses),
        "hypotheses": [h.to_dict() for h in hypotheses]
    }


@app.post("/api/hypotheses/{hypothesis_id}/invalidate")
async def invalidate_hypothesis(hypothesis_id: str, reason: str = ""):
    """Invalidate a hypothesis"""
    from rag.hypothesis_generator import get_hypothesis_generator
    generator = get_hypothesis_generator()
    
    generator.invalidate_hypothesis(hypothesis_id, reason)
    return {"success": True, "hypothesis_id": hypothesis_id}


# ============================================
# Agent Tools Routes
# ============================================

@app.get("/api/tools")
async def get_available_tools():
    """
    Get list of all tools available to agents
    
    Each tool has:
    - Name and description
    - Parameter schema
    - Usage examples
    """
    from agents.tools import get_agent_toolkit
    toolkit = get_agent_toolkit()
    
    return {
        "tools": toolkit.get_available_tools()
    }


@app.post("/api/tools/call")
async def call_tool(tool_name: str, parameters: Dict[str, Any]):
    """
    Directly call an agent tool
    
    Available tools:
    - price_lookup: Get stock prices
    - news_fetch: Get stock news
    - technical_analysis: Get technical indicators
    - fundamental_analysis: Get company fundamentals
    - market_sentiment: Get market sentiment
    - sector_analysis: Analyze sector performance
    - portfolio_metrics: Calculate portfolio metrics
    - hypothesis_check: Validate investment hypotheses
    """
    from agents.tools import get_agent_toolkit, ToolType
    toolkit = get_agent_toolkit()
    
    # Map tool name to ToolType
    tool_map = {
        "price_lookup": ToolType.PRICE_LOOKUP,
        "lookup_price": ToolType.PRICE_LOOKUP,
        "news_fetch": ToolType.NEWS_FETCH,
        "fetch_news": ToolType.NEWS_FETCH,
        "technical_analysis": ToolType.TECHNICAL_ANALYSIS,
        "fundamental_analysis": ToolType.FUNDAMENTAL_ANALYSIS,
        "market_sentiment": ToolType.MARKET_SENTIMENT,
        "sector_analysis": ToolType.SECTOR_ANALYSIS,
        "portfolio_metrics": ToolType.PORTFOLIO_METRICS,
        "hypothesis_check": ToolType.HYPOTHESIS_CHECK
    }
    
    tool_type = tool_map.get(tool_name.lower())
    if not tool_type:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
    
    result = await toolkit.call_tool(tool_type, parameters)
    return result.to_dict()


@app.get("/api/tools/history")
async def get_tool_history(limit: int = 20):
    """Get recent tool call history"""
    from agents.tools import get_agent_toolkit
    toolkit = get_agent_toolkit()
    
    return {
        "history": toolkit.get_call_history(limit)
    }


# ============================================
# News Scraper Routes
# ============================================

@app.get("/api/news")
async def get_news(
    limit: int = 20,
    symbol: Optional[str] = None,
    sentiment: Optional[str] = None
):
    """
    Get recent financial news
    
    Filters:
    - symbol: Filter by stock symbol
    - sentiment: Filter by sentiment (positive, negative, neutral)
    """
    from rag.news_scraper import get_news_scraper
    scraper = get_news_scraper()
    
    articles = scraper.get_recent_articles(
        limit=limit,
        symbol=symbol,
        sentiment=sentiment
    )
    
    return {
        "total": len(articles),
        "articles": [a.to_dict() for a in articles]
    }


@app.post("/api/news/fetch")
async def fetch_news():
    """
    Fetch latest news from all RSS sources
    
    This triggers an immediate fetch from all configured news sources.
    New articles are automatically added to the streaming pipeline.
    """
    from rag.news_scraper import get_news_scraper, integrate_with_pipeline
    from rag.pathway_streaming import get_streaming_pipeline
    
    scraper = get_news_scraper()
    pipeline = get_streaming_pipeline()
    
    # Integrate scraper with pipeline (idempotent)
    await integrate_with_pipeline(pipeline)
    
    # Fetch news
    articles = await scraper.fetch_once()
    
    # Add to streaming pipeline
    for article in articles[:20]:  # Limit to 20 most recent
        await pipeline.add_document(
            doc_id=f"news_{article.article_id}",
            content=article.to_rag_content(),
            doc_type="news",
            symbol=article.symbols[0] if article.symbols else "",
            source=article.source,
            metadata={
                "title": article.title,
                "sentiment": article.sentiment,
                "symbols": article.symbols
            }
        )
    
    return {
        "success": True,
        "articles_fetched": len(articles),
        "added_to_pipeline": min(len(articles), 20)
    }


@app.post("/api/news/stream/start")
async def start_news_streaming(interval_seconds: int = 120):
    """
    Start streaming news in the background
    
    News will be fetched every `interval_seconds` and automatically
    added to the RAG pipeline.
    """
    from rag.news_scraper import get_news_scraper, integrate_with_pipeline
    from rag.pathway_streaming import get_streaming_pipeline
    
    scraper = get_news_scraper()
    pipeline = get_streaming_pipeline()
    
    # Integrate scraper with pipeline
    await integrate_with_pipeline(pipeline)
    
    # Start streaming in background
    asyncio.create_task(scraper.start_streaming(interval_seconds))
    
    return {
        "success": True,
        "message": f"News streaming started (interval: {interval_seconds}s)"
    }


@app.post("/api/news/stream/stop")
async def stop_news_streaming():
    """Stop background news streaming"""
    from rag.news_scraper import get_news_scraper
    scraper = get_news_scraper()
    
    scraper.stop_streaming()
    return {"success": True, "message": "News streaming stopped"}


@app.get("/api/news/stats")
async def get_news_stats():
    """Get news scraper statistics"""
    from rag.news_scraper import get_news_scraper
    scraper = get_news_scraper()
    
    return scraper.get_stats()


@app.get("/api/news/portfolio")
async def get_portfolio_news():
    """
    Get news articles related to portfolio holdings
    
    Automatically extracts symbols from your portfolio and
    returns relevant news for each stock.
    """
    from rag.news_scraper import get_news_scraper
    scraper = get_news_scraper()
    
    # Get portfolio symbols
    portfolio = await state.zerodha_client.get_portfolio_summary()
    symbols = []
    for h in portfolio.get("holdings", []):
        symbol = h.get("symbol") or h.get("tradingsymbol", "")
        symbol = symbol.replace(".NS", "").replace(".BO", "").strip()
        if symbol:
            symbols.append(symbol)
    
    # Get news for portfolio symbols
    news_by_symbol = scraper.get_articles_for_symbols(symbols)
    
    return {
        "portfolio_symbols": symbols,
        "news_by_symbol": {
            symbol: [a.to_dict() for a in articles]
            for symbol, articles in news_by_symbol.items()
        }
    }


# ============================================
# WebSocket for Live Document Feed
# ============================================

@app.websocket("/ws/documents")
async def websocket_document_feed(websocket: WebSocket):
    """
    WebSocket endpoint for live document feed
    
    Shows documents being added to the streaming pipeline in real-time.
    This demonstrates Pathway's live document indexing capability.
    """
    await websocket.accept()
    
    from rag.pathway_streaming import get_document_broadcaster
    broadcaster = get_document_broadcaster()
    
    logger.info("WebSocket client connected to document feed")
    
    # Create callback to send events to this websocket
    async def send_event(event):
        try:
            await websocket.send_json({
                "type": "document_event",
                "data": event.to_dict()
            })
        except Exception as e:
            logger.error(f"Error sending document event: {e}")
    
    # Subscribe to document events
    broadcaster.subscribe(send_event)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Portabull document feed",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send recent events
        recent_events = broadcaster.get_recent_events(10)
        for event in recent_events:
            await websocket.send_json({
                "type": "document_event",
                "data": event.to_dict()
            })
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass
                
    except Exception as e:
        logger.error(f"Document feed WebSocket error: {e}")
    finally:
        broadcaster.unsubscribe(send_event)
        logger.info("WebSocket client disconnected from document feed")


# ============================================
# WebSocket Routes
# ============================================

@app.websocket("/ws/market")
async def websocket_market_feed(websocket: WebSocket):
    """
    WebSocket endpoint for real-time market updates
    
    Streams:
    - Market overview updates
    - Stock price updates  
    - Anomaly alerts
    
    Connect to receive live data without polling.
    """
    await websocket.accept()
    
    if state.realtime_monitor:
        state.realtime_monitor.register_websocket(websocket)
    
    logger.info("WebSocket client connected to market feed")
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Portabull market feed",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle ping/pong for keep-alive
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Handle subscribe to specific symbols
                elif message.get("type") == "subscribe":
                    symbols = message.get("symbols", [])
                    if symbols and state.realtime_monitor:
                        state.realtime_monitor.add_symbols(symbols)
                        await websocket.send_json({
                            "type": "subscribed",
                            "symbols": symbols,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Handle unsubscribe
                elif message.get("type") == "unsubscribe":
                    symbols = message.get("symbols", [])
                    if symbols and state.realtime_monitor:
                        state.realtime_monitor.remove_symbols(symbols)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "symbols": symbols,
                            "timestamp": datetime.now().isoformat()
                        })
                        
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if state.realtime_monitor:
            state.realtime_monitor.unregister_websocket(websocket)
        logger.info("WebSocket client disconnected from market feed")


# ============================================
# Alert Routes
# ============================================

@app.get("/api/alerts", response_model=List[AlertResponse])
async def get_alerts(limit: int = 10, unacknowledged_only: bool = False):
    """Get recent alerts"""
    if not state.portfolio_monitor:
        return []
    
    if unacknowledged_only:
        alerts = state.portfolio_monitor.get_unacknowledged_alerts()
    else:
        alerts = state.portfolio_monitor.get_recent_alerts(limit)
    
    return [
        AlertResponse(
            id=a.id,
            type=a.alert_type.value,
            severity=a.severity.value,
            title=a.title,
            description=a.description,
            symbols=a.affected_symbols,
            action=a.recommended_action,
            timestamp=a.timestamp.isoformat()
        )
        for a in alerts
    ]


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    if state.portfolio_monitor:
        state.portfolio_monitor.acknowledge_alert(alert_id)
    return {"status": "acknowledged"}


# ============================================
# WebSocket for Real-time Updates
# ============================================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    state.websocket_connections[user_id] = websocket
    
    logger.info(f"WebSocket connected: {user_id}")
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat":
                # Process chat message
                result = await state.orchestrator.answer_query(
                    query=message.get("content", ""),
                    context=message.get("context", {}),
                    show_debate=message.get("show_debate", False)
                )
                
                await websocket.send_json({
                    "type": "chat_response",
                    "data": result
                })
            
            elif message.get("type") == "subscribe_portfolio":
                # Send portfolio updates
                portfolio = await state.zerodha_client.get_portfolio_summary()
                await websocket.send_json({
                    "type": "portfolio_update",
                    "data": portfolio
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {user_id}")
        del state.websocket_connections[user_id]


async def broadcast_alert(alert):
    """Broadcast alert to all connected clients"""
    alert_data = alert.to_dict()
    
    for user_id, ws in state.websocket_connections.items():
        try:
            await ws.send_json({
                "type": "alert",
                "data": alert_data
            })
        except Exception as e:
            logger.error(f"Failed to send alert to {user_id}: {e}")


# ============================================
# Paper Trading Routes
# ============================================

class BuyRequest(BaseModel):
    symbol: str
    quantity: int
    exchange: str = "NSE"
    price: Optional[float] = None
    notes: str = ""

class SellRequest(BaseModel):
    symbol: str
    quantity: int
    exchange: str = "NSE"
    price: Optional[float] = None
    notes: str = ""

class SearchRequest(BaseModel):
    query: str
    exchange: str = "NSE"


@app.get("/api/paper/portfolio")
async def get_paper_portfolio():
    """Get paper trading portfolio summary"""
    try:
        # Check if using paper broker
        if hasattr(state.zerodha_client, 'get_portfolio_summary'):
            portfolio = await state.zerodha_client.get_portfolio_summary()
            return portfolio
        else:
            raise HTTPException(status_code=400, detail="Paper trading not enabled")
    except Exception as e:
        logger.error(f"Failed to get paper portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/holdings")
async def get_paper_holdings():
    """Get paper trading holdings"""
    try:
        if hasattr(state.zerodha_client, 'get_holdings'):
            holdings = await state.zerodha_client.get_holdings()
            return {"holdings": holdings}
        else:
            raise HTTPException(status_code=400, detail="Paper trading not enabled")
    except Exception as e:
        logger.error(f"Failed to get paper holdings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper/buy")
async def paper_buy(request: BuyRequest):
    """Buy stocks in paper trading"""
    try:
        if not hasattr(state.zerodha_client, 'buy'):
            raise HTTPException(status_code=400, detail="Paper trading not enabled")
        
        result = await state.zerodha_client.buy(
            symbol=request.symbol,
            quantity=request.quantity,
            exchange=request.exchange,
            price=request.price,
            notes=request.notes
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Buy failed'))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper buy failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper/sell")
async def paper_sell(request: SellRequest):
    """Sell stocks in paper trading"""
    try:
        if not hasattr(state.zerodha_client, 'sell'):
            raise HTTPException(status_code=400, detail="Paper trading not enabled")
        
        result = await state.zerodha_client.sell(
            symbol=request.symbol,
            quantity=request.quantity,
            exchange=request.exchange,
            price=request.price,
            notes=request.notes
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Sell failed'))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper sell failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/quote/{symbol}")
async def get_paper_quote(symbol: str, exchange: str = "NSE"):
    """Get stock quote for paper trading"""
    try:
        if not hasattr(state.zerodha_client, 'get_quote'):
            raise HTTPException(status_code=400, detail="Paper trading not enabled")
        
        quote = await state.zerodha_client.get_quote(symbol, exchange)
        return quote
        
    except Exception as e:
        logger.error(f"Failed to get quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/search")
async def search_stocks(query: str, exchange: str = "NSE"):
    """Search for stocks"""
    try:
        if not hasattr(state.zerodha_client, 'search_stocks'):
            raise HTTPException(status_code=400, detail="Paper trading not enabled")
        
        results = await state.zerodha_client.search_stocks(query, exchange)
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Stock search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/transactions")
async def get_paper_transactions(limit: int = 50):
    """Get paper trading transaction history"""
    try:
        if not hasattr(state.zerodha_client, 'get_transactions'):
            raise HTTPException(status_code=400, detail="Paper trading not enabled")
        
        transactions = await state.zerodha_client.get_transactions(limit)
        return {"transactions": transactions}
        
    except Exception as e:
        logger.error(f"Failed to get transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper/reset")
async def reset_paper_portfolio(initial_cash: float = 1000000.0):
    """Reset paper trading portfolio"""
    try:
        if not hasattr(state.zerodha_client, 'reset_portfolio'):
            raise HTTPException(status_code=400, detail="Paper trading not enabled")
        
        result = await state.zerodha_client.reset_portfolio(initial_cash)
        return result
        
    except Exception as e:
        logger.error(f"Failed to reset portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Health Check
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "zerodha": "connected" if state.zerodha_client else "disconnected",
            "llm": "connected" if state.llm_client else "disconnected",
            "orchestrator": "active" if state.orchestrator else "inactive",
            "rag": "active" if state.rag_engine else "inactive"
        }
    }


# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
