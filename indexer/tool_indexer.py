"""
Layer 2: Vectorized Tool Indexer
Semantic search for tools using ChromaDB
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from config import vector_db_config
import json

class ToolIndexer:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=vector_db_config.PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=vector_db_config.COLLECTION_NAME
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            vector_db_config.EMBEDDING_MODEL
        )
    
    def index_tools(self, tools: List[Dict[str, Any]]):
        """
        Index all tools into vector database
        
        Args:
            tools: List of tool dictionaries with metadata
        """
        if self.collection.count() > 0:
            print("Tools already indexed. Skipping...")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, tool in enumerate(tools):
            # Create searchable document
            doc = self._create_tool_document(tool)
            documents.append(doc)
            
            # Store metadata
            metadatas.append({
                "tool_id": tool["id"],
                "domain": tool["domain"],
                "method": tool["method"],
                "url": tool["url"],
                "name": tool["name"]
            })
            
            ids.append(f"tool_{idx}")
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Indexed {len(tools)} tools")
    
    def retrieve_tools(self, query: str, domain: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant tools for a query within a specific domain
        
        Args:
            query: User query
            domain: Domain to search within
            top_k: Number of tools to retrieve
            
        Returns:
            List of relevant tools
        """
        if top_k is None:
            top_k = vector_db_config.TOP_K_RETRIEVAL
        
        # Query with domain filter
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"domain": domain}
        )
        
        # Format results
        tools = []
        if results["metadatas"] and len(results["metadatas"][0]) > 0:
            for metadata, document, distance in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0]
            ):
                tools.append({
                    "tool_id": metadata["tool_id"],
                    "name": metadata["name"],
                    "domain": metadata["domain"],
                    "method": metadata["method"],
                    "url": metadata["url"],
                    "relevance_score": 1 - distance,
                    "description": document
                })
        
        return tools
    
    def _create_tool_document(self, tool: Dict[str, Any]) -> str:
        """Create searchable text representation of tool"""
        parts = [
            f"Name: {tool['name']}",
            f"Description: {tool.get('description', '')}",
            f"Method: {tool['method']}",
            f"URL: {tool['url']}",
            f"Domain: {tool['domain']}"
        ]
        
        if tool.get("parameters"):
            params = ", ".join([p["name"] for p in tool["parameters"]])
            parts.append(f"Parameters: {params}")
        
        return " | ".join(parts)