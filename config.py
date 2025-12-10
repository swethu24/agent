from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
class LLMConfig:
    # Layer 1: Global Router (Fast, cheap model)
    ROUTER_MODEL = "gpt-3.5-turbo"
    ROUTER_TEMPERATURE = 0.0
    
    # Layer 3: Specialized Agent (Capable model)
    AGENT_MODEL = "claude-3-5-sonnet-20241022"
    AGENT_TEMPERATURE = 0.1
    AGENT_MAX_TOKENS = 4096
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Domain Configuration
class DomainConfig:
    """Define high-level domains for Layer 1 routing"""
    DOMAINS: List[str] = [
        "INVOICING",
        "PAYMENTS",
        "REPORTING",
        "DISPUTES",
        "USER_MANAGEMENT",
        "RAG_QUERY",
        "SYSTEM_SEARCH",
        "GENERAL"
    ]
    
    DOMAIN_DESCRIPTIONS: Dict[str, str] = {
        "INVOICING": "Creating, updating, or managing invoices and billing",
        "PAYMENTS": "Processing payments, refunds, and payment methods",
        "REPORTING": "Generating reports, analytics, and data exports",
        "DISPUTES": "Handling chargebacks, disputes, and claims",
        "USER_MANAGEMENT": "Managing users, accounts, and permissions",
        "RAG_QUERY": "Answering questions from knowledge base",
        "SYSTEM_SEARCH": "Searching for request status or available tools",
        "GENERAL": "General queries that don't fit other categories"
    }

# Vector Database Configuration
class VectorDBConfig:
    COLLECTION_NAME = "tool_embeddings"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    PERSIST_DIRECTORY = "./chroma_db"
    TOP_K_RETRIEVAL = 10

# Tool Configuration
class ToolConfig:
    POSTMAN_COLLECTIONS_DIR = "./postman_collections"
    MAX_TOOLS_PER_DOMAIN = 15
    TOOL_TIMEOUT_SECONDS = 30

# LangSmith Configuration
class ObservabilityConfig:
    LANGSMITH_PROJECT = "agentic-system-prod"
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    
# Export configurations
llm_config = LLMConfig()
domain_config = DomainConfig()
vector_db_config = VectorDBConfig()
tool_config = ToolConfig()
observability_config = ObservabilityConfig()
# Error message templates
ERROR_TEMPLATES = {
        "TIMEOUT": "The request took too long to complete. The service might be slow right now. Please try again in a moment.",
        "AUTH": "There was an authentication issue. Please check that your API credentials are configured correctly.",
        "NOT_FOUND": "I couldn't find the resource you're looking for. Please check that the ID or name is correct.",
        "INVALID_INPUT": "Some of the information provided appears to be invalid. Please check your input and try again.",
        "RATE_LIMIT": "You've made too many requests recently. Please wait a moment before trying again.",
        "FORBIDDEN": "You don't have permission to perform this action. Please check your access rights.",
        "SERVER_ERROR": "The service is experiencing issues. Please try again in a few minutes.",
        "CONNECTION": "I couldn't connect to the service. Please check your internet connection.",
        "UNKNOWN": "Something unexpected happened. Please try again or rephrase your request."
    }
