"""
Layer 1: Global Router for Domain Classification
Uses a fast, cheap LLM to classify user queries into high-level domains
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any
from config import llm_config, domain_config

class GlobalRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=llm_config.ROUTER_MODEL,
            temperature=llm_config.ROUTER_TEMPERATURE,
            api_key=llm_config.OPENAI_API_KEY
        )
        
        # Create routing prompt
        domain_list = "\\n".join([
            f"- {domain}: {desc}" 
            for domain, desc in domain_config.DOMAIN_DESCRIPTIONS.items()
        ])
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a domain classification router. Classify the user's query into ONE of these domains:

            {domain_list}

            Respond with ONLY the domain name (e.g., INVOICING, PAYMENTS, etc.). No explanation."""),
            ("user", "{query}")
        ])
        
        self.chain = self.prompt | self.llm
    
    def route(self, query: str) -> str:
        """
        Classify user query into a domain
        Args:
            query: User's input query    
        Returns:
            Domain name (e.g., "INVOICING")
        """
        try:
            response = self.chain.invoke({"query": query})
            domain = response.content.strip().upper()
            
            # Validate domain
            if domain not in domain_config.DOMAINS:
                return "GENERAL"
                
            return domain
            
        except Exception as e:
            print(f"Router error: {e}")
            return "GENERAL"
    
    def route_with_confidence(self, query: str) -> Dict[str, Any]:
        """Route with additional metadata"""
        domain = self.route(query)
        return {
            "domain": domain,
            "query": query,
            "description": domain_config.DOMAIN_DESCRIPTIONS.get(domain, "")
        }