"""
Error Handler: Generates user-friendly error messages
Analyzes technical errors and converts them to helpful user messages
"""
from langchain_openai import ChatOpenAI
from config import llm_config,ERROR_TEMPLATES
from typing import Dict, Any, Optional


class ErrorHandler:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=llm_config.ROUTER_MODEL,
            temperature=0.3,
            api_key=llm_config.OPENAI_API_KEY
        )
    
    def generate_error_message(
        self, 
        user_query: str, 
        error: str,
        tool_call: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a user-friendly error message using LLM
        
        Args:
            user_query: Original user query
            error: Technical error message
            tool_call: The attempted tool call (if any)
            
        Returns:
            User-friendly error message
        """
        error_category = self.categorize_error(error)
        
        # For common errors, use templates
        if error_category in ERROR_TEMPLATES:
            return self._use_template(error_category, user_query, error)
        
        # For complex errors, use LLM
        return self._generate_with_llm(user_query, error, tool_call)
    
    def _generate_with_llm(self, user_query: str, error: str, 
                           tool_call: Optional[Dict[str, Any]]) -> str:
        """Generate error message using LLM"""
        tool_info = ""
        if tool_call:
            tool_info = f"\nTool attempted: {tool_call.get('tool_id', 'unknown')}\nParameters: {tool_call.get('parameters', {})}"
        
        prompt = f"""The user asked: "{user_query}"

        We attempted to fulfill their request but encountered an error:
        Error: {error}{tool_info}

        Generate a clear, helpful error message for the user that:
        1. Explains what went wrong in simple, non-technical terms
        2. Suggests what they should check or do differently
        3. Maintain a friendly, supportive tone
        4. Does NOT include technical details or codes

        Keep it concise (2-3 sentences)."""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception:
            return self._fallback_message(error)
    
    def _use_template(self, category: str, user_query: str, error: str) -> str:
        """Use predefined template for common errors"""
        template = ERROR_TEMPLATES[category]
        
        # Extract specific details if possible
        if category == "INVALID_INPUT":
            # Try to find what was invalid
            if "email" in error.lower():
                return "The email address you provided appears to be invalid. Please check and try again."
            elif "amount" in error.lower() or "number" in error.lower():
                return "The amount or number you provided is invalid. Please check the format and try again."
        
        return template
    
    def categorize_error(self, error: str) -> str:
        """
        Categorize error type for appropriate handling
        
        Returns:
            Error category code
        """
        error_lower = error.lower()
        
        if "timeout" in error_lower or "408" in error_lower:
            return "TIMEOUT"
        elif any(word in error_lower for word in ["authentication", "unauthorized", "401", "api key", "token"]):
            return "AUTH"
        elif "not found" in error_lower or "404" in error_lower:
            return "NOT_FOUND"
        elif any(word in error_lower for word in ["invalid", "bad request", "400", "validation"]):
            return "INVALID_INPUT"
        elif "rate limit" in error_lower or "429" in error_lower:
            return "RATE_LIMIT"
        elif "forbidden" in error_lower or "403" in error_lower:
            return "FORBIDDEN"
        elif any(word in error_lower for word in ["500", "internal server", "503", "service unavailable"]):
            return "SERVER_ERROR"
        elif "connection" in error_lower or "network" in error_lower:
            return "CONNECTION"
        else:
            return "UNKNOWN"
    
    def _fallback_message(self, error: str) -> str:
        """Fallback message when LLM fails"""
        category = self.categorize_error(error)
        
        if category in ERROR_TEMPLATES:
            return ERROR_TEMPLATES[category]
        
        return "I encountered an issue while processing your request. Please try again or rephrase your query."
    
    

