"""
Portabull - Vector Store Implementation
Manages vector embeddings for efficient semantic search
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import numpy as np
from datetime import datetime
import json


@dataclass
class VectorSearchResult:
    """Result from vector search"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class VectorStore:
    """
    Vector Store for semantic search using ChromaDB
    
    Provides:
    - Fast similarity search
    - Metadata filtering
    - Persistent storage
    - Real-time updates
    """
    
    def __init__(
        self,
        collection_name: str = "portabull_docs",
        persist_directory: str = "data/vector_store"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(
            ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Portabull document store"}
        )
        
        logger.info(f"VectorStore initialized with collection: {collection_name}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to the vector store"""
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas or [{}] * len(documents)
        )
        
        logger.debug(f"Added {len(documents)} documents to vector store")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar documents"""
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(
                VectorSearchResult(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i] if results['documents'] else "",
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    score=results['distances'][0][i] if results['distances'] else 0.0
                )
            )
        
        return search_results
    
    def update_document(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None
    ):
        """Update an existing document"""
        
        self.collection.update(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )
        
        logger.debug(f"Updated document: {doc_id}")
    
    def delete_document(self, doc_id: str):
        """Delete a document from the store"""
        
        self.collection.delete(ids=[doc_id])
        logger.debug(f"Deleted document: {doc_id}")
    
    def get_document(self, doc_id: str) -> Optional[VectorSearchResult]:
        """Get a specific document by ID"""
        
        result = self.collection.get(ids=[doc_id])
        
        if result['ids']:
            return VectorSearchResult(
                id=result['ids'][0],
                content=result['documents'][0] if result['documents'] else "",
                metadata=result['metadatas'][0] if result['metadatas'] else {},
                score=1.0
            )
        
        return None
    
    def clear(self):
        """Clear all documents from the store"""
        
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        
        logger.info("Vector store cleared")
    
    def count(self) -> int:
        """Get the number of documents in the store"""
        return self.collection.count()


class MarketKnowledgeBase:
    """
    Specialized knowledge base for market data
    
    Maintains separate collections for:
    - Stock fundamentals
    - News articles
    - Technical indicators
    - Analyst reports
    - User portfolio history
    """
    
    def __init__(self, persist_directory: str = "data/market_kb"):
        self.persist_directory = persist_directory
        
        # Initialize specialized stores
        self.fundamentals_store = VectorStore(
            collection_name="fundamentals",
            persist_directory=persist_directory
        )
        
        self.news_store = VectorStore(
            collection_name="news",
            persist_directory=persist_directory
        )
        
        self.technicals_store = VectorStore(
            collection_name="technicals",
            persist_directory=persist_directory
        )
        
        self.reports_store = VectorStore(
            collection_name="reports",
            persist_directory=persist_directory
        )
        
        self.portfolio_history_store = VectorStore(
            collection_name="portfolio_history",
            persist_directory=persist_directory
        )
        
        logger.info("MarketKnowledgeBase initialized")
    
    def get_store_for_type(self, data_type: str) -> VectorStore:
        """Get the appropriate store for a data type"""
        
        stores = {
            "fundamentals": self.fundamentals_store,
            "news": self.news_store,
            "technicals": self.technicals_store,
            "reports": self.reports_store,
            "portfolio": self.portfolio_history_store
        }
        
        return stores.get(data_type, self.fundamentals_store)
    
    def search_all(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ) -> Dict[str, List[VectorSearchResult]]:
        """Search across all knowledge stores"""
        
        results = {}
        
        for store_name, store in [
            ("fundamentals", self.fundamentals_store),
            ("news", self.news_store),
            ("technicals", self.technicals_store),
            ("reports", self.reports_store),
            ("portfolio", self.portfolio_history_store)
        ]:
            if store.count() > 0:
                results[store_name] = store.search(query_embedding, top_k)
        
        return results
