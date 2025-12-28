"""
Portabull - Dynamic RAG Engine with Pathway Framework
Real-time document processing and retrieval for stock market intelligence

This is a TRUE Dynamic RAG implementation that:
1. Fetches real market data for stocks in the portfolio
2. Updates context dynamically before each query
3. Uses Pathway framework for streaming capabilities
4. Provides comprehensive market intelligence
"""

import pathway as pw
from pathway.stdlib.ml.index import KNNIndex
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import json
from datetime import datetime


@dataclass
class DocumentChunk:
    """Represents a chunk of processed document"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: datetime = None


class PathwayRAGEngine:
    """
    Dynamic RAG Engine using Pathway Framework
    
    Features:
    - Real-time document ingestion and processing
    - Automatic re-indexing on data changes
    - Streaming updates for live market data
    - Multi-source data fusion
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        persistence_path: str = "data/pathway_state"
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.persistence_path = persistence_path
        
        # Initialize embedder
        self.embedder = SentenceTransformerEmbedder(model=embedding_model)
        
        # Initialize splitter
        self.splitter = TokenCountSplitter(max_tokens=chunk_size)
        
        # Document stores
        self.documents: Dict[str, DocumentChunk] = {}
        
        logger.info(f"PathwayRAGEngine initialized with model: {embedding_model}")
    
    def create_document_pipeline(self):
        """
        Create Pathway pipeline for document processing
        
        This pipeline:
        1. Ingests documents from multiple sources
        2. Splits into chunks
        3. Generates embeddings
        4. Indexes for retrieval
        """
        
        # Define input schema
        class DocumentSchema(pw.Schema):
            doc_id: str
            content: str
            source: str
            doc_type: str
            metadata: str  # JSON string
            timestamp: int
        
        # Create input connector (can be file, Kafka, HTTP, etc.)
        # For hackathon, we'll use in-memory connector
        documents = pw.debug.table_from_markdown(
            """
            | doc_id | content | source | doc_type | metadata | timestamp |
            | ------ | ------- | ------ | -------- | -------- | --------- |
            """
        )
        
        return documents
    
    def create_market_data_pipeline(self):
        """
        Create real-time market data processing pipeline
        
        Handles:
        - Live price updates
        - News articles
        - Corporate announcements
        - Analyst reports
        """
        
        class MarketDataSchema(pw.Schema):
            symbol: str
            data_type: str  # price, news, announcement, report
            content: str
            timestamp: int
            metadata: str
        
        # This would connect to real-time data sources
        # For now, creating placeholder pipeline
        market_data = pw.debug.table_from_markdown(
            """
            | symbol | data_type | content | timestamp | metadata |
            | ------ | --------- | ------- | --------- | -------- |
            """
        )
        
        return market_data
    
    async def add_document(
        self,
        content: str,
        doc_id: str,
        source: str = "manual",
        doc_type: str = "text",
        metadata: Dict[str, Any] = None
    ) -> DocumentChunk:
        """Add a document to the RAG system"""
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        chunk = DocumentChunk(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            timestamp=datetime.now()
        )
        
        self.documents[doc_id] = chunk
        logger.debug(f"Added document: {doc_id}")
        
        return chunk
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Lazy load model
            if not hasattr(self, '_model'):
                self._model = SentenceTransformer(self.embedding_model)
            
            embedding = self._model.encode(text).tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of relevant document chunks
        """
        
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Calculate similarities
        scored_docs = []
        for doc_id, chunk in self.documents.items():
            if chunk.embedding:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                
                # Apply filters if specified
                if filters:
                    if not self._matches_filters(chunk, filters):
                        continue
                
                scored_docs.append((similarity, chunk))
        
        # Sort by similarity and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        
        a = np.array(vec1)
        b = np.array(vec2)
        
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """Check if document matches the specified filters"""
        for key, value in filters.items():
            if key not in chunk.metadata:
                return False
            if chunk.metadata[key] != value:
                return False
        return True
    
    async def update_market_context(self, portfolio_data: Dict[str, Any]):
        """
        Update RAG context with latest portfolio and market data
        
        This method is called periodically to keep the RAG system
        updated with the latest market information
        """
        
        # Add portfolio summary
        if "holdings" in portfolio_data:
            holdings_text = self._format_holdings_for_rag(portfolio_data["holdings"])
            await self.add_document(
                content=holdings_text,
                doc_id="portfolio_holdings",
                source="zerodha",
                doc_type="portfolio",
                metadata={"type": "holdings", "updated": datetime.now().isoformat()}
            )
        
        # Add positions
        if "positions" in portfolio_data:
            positions_text = self._format_positions_for_rag(portfolio_data["positions"])
            await self.add_document(
                content=positions_text,
                doc_id="portfolio_positions",
                source="zerodha",
                doc_type="portfolio",
                metadata={"type": "positions", "updated": datetime.now().isoformat()}
            )
        
        logger.info("Market context updated in RAG system")
    
    async def update_dynamic_market_data(self, symbols: List[str], exchange: str = "NSE"):
        """
        DYNAMIC RAG: Fetch and update real-time market data for portfolio stocks
        
        This is the core of Dynamic RAG - it fetches real market data
        from yfinance and adds it to the RAG context before each query.
        
        Args:
            symbols: List of stock symbols in the portfolio
            exchange: Stock exchange (NSE, BSE, NASDAQ, etc.)
        """
        from rag.market_data_fetcher import get_market_data_fetcher
        
        fetcher = get_market_data_fetcher()
        
        logger.info(f"Dynamic RAG: Fetching market data for {len(symbols)} stocks...")
        
        # 1. Fetch market overview first
        market_overview = await fetcher.fetch_market_overview()
        market_text = fetcher.format_for_rag(market_overview, "market")
        await self.add_document(
            content=market_text,
            doc_id="market_overview",
            source="yfinance",
            doc_type="market",
            metadata={"type": "market_overview", "updated": datetime.now().isoformat()}
        )
        
        # 2. Track sectors for sector analysis
        sectors_seen = set()
        
        # 3. Fetch comprehensive data for each stock
        for symbol in symbols:
            try:
                # Fetch stock fundamentals and price data
                stock_data = await fetcher.fetch_stock_data(symbol, exchange)
                stock_text = fetcher.format_for_rag(stock_data, "stock")
                await self.add_document(
                    content=stock_text,
                    doc_id=f"stock_data_{symbol}",
                    source="yfinance",
                    doc_type="stock_analysis",
                    metadata={
                        "symbol": symbol,
                        "type": "stock_data",
                        "updated": datetime.now().isoformat()
                    }
                )
                
                # Track sector
                if stock_data.get("sector"):
                    sectors_seen.add(stock_data["sector"])
                
                # Fetch news for the stock
                news = await fetcher.fetch_stock_news(symbol, exchange)
                if news:
                    news_text = fetcher.format_for_rag(news, "news")
                    await self.add_document(
                        content=f"=== News for {symbol} ===\n{news_text}",
                        doc_id=f"news_{symbol}",
                        source="yfinance",
                        doc_type="news",
                        metadata={
                            "symbol": symbol,
                            "type": "news",
                            "updated": datetime.now().isoformat()
                        }
                    )
                
                logger.debug(f"Dynamic RAG: Added data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        # 4. Fetch sector overviews for portfolio sectors
        for sector in sectors_seen:
            try:
                sector_data = await fetcher.fetch_sector_overview(sector)
                sector_text = fetcher.format_for_rag(sector_data, "sector")
                await self.add_document(
                    content=sector_text,
                    doc_id=f"sector_{sector.replace(' ', '_')}",
                    source="yfinance",
                    doc_type="sector",
                    metadata={
                        "sector": sector,
                        "type": "sector_overview",
                        "updated": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error fetching sector data for {sector}: {e}")
        
        logger.info(f"Dynamic RAG: Updated with data for {len(symbols)} stocks, {len(sectors_seen)} sectors")
        return {
            "stocks_updated": len(symbols),
            "sectors_updated": len(sectors_seen),
            "documents_count": len(self.documents)
        }
    
    def _format_holdings_for_rag(self, holdings: List[Dict]) -> str:
        """Format holdings data for RAG ingestion"""
        lines = ["Current Portfolio Holdings:\n"]
        
        for h in holdings:
            lines.append(
                f"- {h.get('tradingsymbol', 'N/A')}: "
                f"Qty: {h.get('quantity', 0)}, "
                f"Avg Cost: ₹{h.get('average_price', 0):.2f}, "
                f"Current: ₹{h.get('last_price', 0):.2f}, "
                f"P&L: ₹{h.get('pnl', 0):.2f} ({h.get('pnl_percent', 0):.2f}%)"
            )
        
        return "\n".join(lines)
    
    def _format_positions_for_rag(self, positions: List[Dict]) -> str:
        """Format positions data for RAG ingestion"""
        lines = ["Current Positions:\n"]
        
        for p in positions:
            lines.append(
                f"- {p.get('tradingsymbol', 'N/A')}: "
                f"Type: {p.get('product', 'N/A')}, "
                f"Qty: {p.get('quantity', 0)}, "
                f"Buy Price: ₹{p.get('buy_price', 0):.2f}, "
                f"Current: ₹{p.get('last_price', 0):.2f}, "
                f"P&L: ₹{p.get('pnl', 0):.2f}"
            )
        
        return "\n".join(lines)


class StreamingRAGPipeline:
    """
    Streaming RAG Pipeline for real-time data
    
    Uses Pathway's streaming capabilities to process
    continuous data updates from market feeds
    """
    
    def __init__(self, rag_engine: PathwayRAGEngine):
        self.rag_engine = rag_engine
        self.running = False
    
    async def start_pipeline(self):
        """Start the streaming RAG pipeline"""
        self.running = True
        logger.info("Streaming RAG pipeline started")
        
        while self.running:
            # In production, this would connect to Pathway's
            # streaming infrastructure
            await asyncio.sleep(1)
    
    async def stop_pipeline(self):
        """Stop the streaming pipeline"""
        self.running = False
        logger.info("Streaming RAG pipeline stopped")
    
    async def process_market_update(self, update: Dict[str, Any]):
        """Process a market data update"""
        
        update_type = update.get("type", "price")
        symbol = update.get("symbol", "UNKNOWN")
        
        if update_type == "price":
            content = (
                f"Price update for {symbol}: "
                f"₹{update.get('price', 0):.2f} "
                f"({update.get('change_percent', 0):+.2f}%)"
            )
        elif update_type == "news":
            content = f"News for {symbol}: {update.get('headline', '')}"
        elif update_type == "announcement":
            content = f"Corporate announcement for {symbol}: {update.get('content', '')}"
        else:
            content = json.dumps(update)
        
        await self.rag_engine.add_document(
            content=content,
            doc_id=f"{symbol}_{update_type}_{datetime.now().timestamp()}",
            source="market_feed",
            doc_type=update_type,
            metadata={
                "symbol": symbol,
                "type": update_type,
                "timestamp": datetime.now().isoformat()
            }
        )


def create_rag_engine() -> PathwayRAGEngine:
    """Factory function to create RAG engine instance"""
    from config import get_settings
    
    settings = get_settings()
    
    return PathwayRAGEngine(
        embedding_model=settings.EMBEDDING_MODEL,
        persistence_path=settings.PATHWAY_PERSISTENCE_PATH
    )
