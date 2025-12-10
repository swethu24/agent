"""
Vector Database Utilities for ChromaDB
Provides helper functions for managing the tool embedding collection
"""
import chromadb
from chromadb.config import Settings
from config import vector_db_config
from typing import Dict, List, Any

def get_client():
    """Get ChromaDB client"""
    return chromadb.PersistentClient(
        path=vector_db_config.PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )

def get_collection_info() -> Dict[str, Any]:
    """
    Get information about the vector collection
    
    Returns:
        {
            "name": str,
            "count": int,
            "metadata": dict
        }
    """
    try:
        client = get_client()
        collection = client.get_collection(name=vector_db_config.COLLECTION_NAME)
        
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata or {}
        }
    except Exception as e:
        return {
            "error": str(e),
            "name": vector_db_config.COLLECTION_NAME,
            "count": 0,
            "metadata": {}
        }

def reset_collection():
    """
    Reset the vector collection (delete and recreate)
    
    Returns:
        New collection object
    """
    client = get_client()
    
    # Delete existing collection
    try:
        client.delete_collection(name=vector_db_config.COLLECTION_NAME)
        print(f" Deleted existing collection: {vector_db_config.COLLECTION_NAME}")
    except Exception:
        print(f"  No existing collection to delete")
    
    # Create new collection
    collection = client.create_collection(
        name=vector_db_config.COLLECTION_NAME,
        metadata={"description": "Tool embeddings for agentic system"}
    )
    print(f" Created new collection: {vector_db_config.COLLECTION_NAME}")
    
    return collection

def list_all_tools() -> List[Dict[str, Any]]:
    """
    List all tools in the collection
    
    Returns:
        List of tool metadata
    """
    try:
        client = get_client()
        collection = client.get_collection(name=vector_db_config.COLLECTION_NAME)
        
        results = collection.get()
        
        tools = []
        for i, metadata in enumerate(results["metadatas"]):
            tools.append({
                "id": results["ids"][i],
                "tool_id": metadata.get("tool_id"),
                "name": metadata.get("name"),
                "domain": metadata.get("domain"),
                "method": metadata.get("method"),
                "url": metadata.get("url")
            })
        
        return tools
    except Exception as e:
        print(f"Error listing tools: {e}")
        return []

def search_tools_by_domain(domain: str) -> List[Dict[str, Any]]:
    """
    Get all tools in a specific domain
    
    Args:
        domain: Domain name (e.g., "INVOICING")
        
    Returns:
        List of tools in that domain
    """
    try:
        client = get_client()
        collection = client.get_collection(name=vector_db_config.COLLECTION_NAME)
        
        results = collection.get(where={"domain": domain})
        
        tools = []
        for i, metadata in enumerate(results["metadatas"]):
            tools.append({
                "id": results["ids"][i],
                "tool_id": metadata.get("tool_id"),
                "name": metadata.get("name"),
                "domain": metadata.get("domain"),
                "method": metadata.get("method")
            })
        
        return tools
    except Exception as e:
        print(f"Error searching by domain: {e}")
        return []
