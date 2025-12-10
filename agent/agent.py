"""
Layer 3: Specialized Agent
Performs reasoning, parameter extraction, and tool execution
"""
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from typing import Dict, Any, List
from config import llm_config
import json

class SpecializedAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            model=llm_config.AGENT_MODEL,
            temperature=llm_config.AGENT_TEMPERATURE,
            max_tokens=llm_config.AGENT_MAX_TOKENS,
            api_key=llm_config.ANTHROPIC_API_KEY
        )
    
    def decide_action(
        self, 
        query: str, 
        tools: List[Dict[str, Any]], 
        chat_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Decide whether to use a tool or respond directly
        
        Returns:
            {
                "action": "use_tool" | "respond",
                "tool_call": {...} | None,
                "response": str | None
            }
        """
        # Format tools for prompt
        tool_descriptions = self._format_tools(tools)
        
        # Build prompt
        prompt = f"""You are a helpful API assistant. Given the user's query and available tools, decide whether to:
        1. Call an API tool to fulfill the request
        2. Respond directly if no tool is needed or available

        User Query: {query}

        Available Tools:
        {tool_descriptions}

        If you need to use a tool, respond with JSON:
        {{"action": "use_tool", "tool_id": "...", "parameters": {{...}}}}

        If you can respond directly, respond with JSON:
        {{"action": "respond", "response": "..."}}

        Be precise with parameter extraction."""

        # Add chat history
        messages = []
        for msg in chat_history[-4:]:  # Last 4 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=prompt))
        
        # Get decision
        response = self.llm.invoke(messages)
        
        try:
            decision = json.loads(response.content)
            return decision
        except json.JSONDecodeError:
            # Fallback: try to respond directly
            return {
                "action": "respond",
                "response": response.content
            }
    
    def synthesize_response(
        self, 
        query: str, 
        tool_result: Dict[str, Any],
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Synthesize a natural language response from tool result
        """
        prompt = f"""The user asked: "{query}"

        The API returned:
        {json.dumps(tool_result, indent=2)}

        Provide a clear, natural language response to the user based on this result.
        Be concise and helpful."""

        messages = []
        for msg in chat_history[-2:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=prompt))
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for LLM prompt"""
        formatted = []
        for tool in tools[:10]:  # Limit to top 10
            formatted.append(
                f"- {tool['name']} (ID: {tool['tool_id']}): "
                f"{tool.get('description', '')} "
                f"[{tool['method']} {tool['url']}]"
            )
        return "\\n".join(formatted)
