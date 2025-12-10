"""
LangGraph State Machine Workflow
Implements the complete 3-layer hierarchical routing system with proper state management
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Annotated, Literal
from typing_extensions import TypedDict
import operator

# Import components
from router.global_router import GlobalRouter
from indexer.tool_indexer import ToolIndexer
from agent.agent import SpecializedAgent
from tools.api_executor import APIExecutor
from utils.error_handler import ErrorHandler
from config import domain_config

import json

class AgentState(TypedDict):
    """
    Complete state for the agentic workflow
    All information flows through this state object
    """
    # User input
    user_query: str
    chat_history: List[Dict[str, str]]
    
    # Layer 1: Router output
    domain: str
    domain_confidence: float
    
    # Layer 2: Tool retrieval
    retrieved_tools: List[Dict[str, Any]]
    retrieval_count: int
    
    # Layer 3: Agent decision
    agent_decision: Dict[str, Any]  # {"action": "use_tool"|"respond", ...}
    selected_tool: Dict[str, Any]
    
    # Tool execution
    tool_result: Dict[str, Any]
    execution_attempts: int
    
    # Error handling
    error_message: str
    error_category: str
    
    # Final output
    final_response: str
    
    # Metadata
    workflow_path: List[str]  # Track which nodes were visited


# Global instances (created once)
router = GlobalRouter()
indexer = ToolIndexer()
agent = SpecializedAgent()
executor = APIExecutor()
error_handler = ErrorHandler()


def route_query(state: AgentState) -> AgentState:
    """
    Layer 1: Route user query to appropriate domain
    
    This is the entry point - classifies intent into high-level domains
    """
    print(f"\n{'='*60}")
    print(f" LAYER 1: GLOBAL ROUTER")
    print(f"{'='*60}")
    print(f"Query: {state['user_query']}")
    
    # Classify domain
    routing_result = router.route_with_confidence(state["user_query"])
    
    state["domain"] = routing_result["domain"]
    state["domain_confidence"] = routing_result.get("confidence", 1.0)
    state["workflow_path"].append(f"route:{routing_result['domain']}")
    
    print(f" Routed to domain: {routing_result['domain']}")
    print(f"   Description: {routing_result.get('description', '')}")
    
    return state


def retrieve_tools(state: AgentState) -> AgentState:
    """
    Layer 2: Retrieve relevant tools using vector search
    
    Queries the vector database for tools matching the user's intent
    within the classified domain
    """
    print(f"\n{'='*60}")
    print(f" LAYER 2: TOOL INDEXER")
    print(f"{'='*60}")
    
    # Retrieve tools from vector DB
    tools = indexer.retrieve_tools(
        query=state["user_query"],
        domain=state["domain"],
        top_k=10
    )
    
    state["retrieved_tools"] = tools
    state["retrieval_count"] = len(tools)
    state["workflow_path"].append(f"retrieve:{len(tools)}_tools")
    
    print(f" Retrieved {len(tools)} relevant tools")
    if tools:
        print(f"Top tools:")
        for i, tool in enumerate(tools[:3], 1):
            print(f"   {i}. {tool['name']} (score: {tool.get('relevance_score', 0):.2f})")
    
    return state



def agent_decide(state: AgentState) -> AgentState:
    """
    Layer 3: Agent decides whether to use a tool or respond directly
    
    The specialized agent analyzes the query and available tools
    """
    print(f"\n{'='*60}")
    print(f" LAYER 3: SPECIALIZED AGENT - DECISION")
    print(f"{'='*60}")
    
    # Get agent's decision
    decision = agent.decide_action(
        query=state["user_query"],
        tools=state["retrieved_tools"],
        chat_history=state["chat_history"]
    )
    
    state["agent_decision"] = decision
    state["workflow_path"].append(f"decide:{decision.get('action', 'unknown')}")
    
    print(f" Agent decision: {decision.get('action', 'unknown').upper()}")
    
    if decision.get("action") == "use_tool":
        # Find the selected tool details
        tool_id = decision.get("tool_id")
        selected = next((t for t in state["retrieved_tools"] if t["tool_id"] == tool_id), None)
        state["selected_tool"] = selected or {}
        print(f"   Tool: {decision.get('tool_id')}")
        print(f"   Parameters: {json.dumps(decision.get('parameters', {}), indent=2)}")
    else:
        print(f"   Response: {decision.get('response', '')[:100]}...")
    
    return state


def call_tool(state: AgentState) -> AgentState:
    """
    Execute the selected API tool
    
    Makes the actual HTTP request using the API executor
    """
    print(f"\n{'='*60}")
    print(f"TOOL EXECUTION")
    print(f"{'='*60}")
    
    tool_call = state["agent_decision"]
    attempts = state.get("execution_attempts", 0) + 1
    state["execution_attempts"] = attempts
    
    print(f"Executing: {tool_call.get('tool_id')}")
    print(f"   Attempt: {attempts}")
    
    try:
        # Execute the tool
        result = executor.execute(
            tool_id=tool_call["tool_id"],
            parameters=tool_call.get("parameters", {}),
            tool_definition=state["selected_tool"]
        )
        
        state["tool_result"] = result
        state["workflow_path"].append(f"execute:{result.get('status_code', 0)}")
        
        if result.get("success"):
            print(f"  Execution successful (Status: {result.get('status_code')})")
            print(f"   Response: {json.dumps(result.get('data', {}), indent=2)[:200]}...")
        else:
            print(f" Execution failed (Status: {result.get('status_code')})")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            state["error_message"] = result.get("error", "Tool execution failed")
            state["error_category"] = error_handler.categorize_error(state["error_message"])
        
    except Exception as e:
        print(f" Exception during execution: {str(e)}")
        state["error_message"] = str(e)
        state["error_category"] = "EXECUTION_ERROR"
        state["tool_result"] = {"success": False, "error": str(e)}
        state["workflow_path"].append("execute:exception")
    
    return state


# ============================================================================
# NODE FUNCTIONS (Response Synthesis)
# ============================================================================

def synthesize_response(state: AgentState) -> AgentState:
    """
    Synthesize final natural language response
    
    Converts tool results or direct responses into user-friendly messages
    """
    print(f"\n{'='*60}")
    print(f"RESPONSE SYNTHESIS")
    print(f"{'='*60}")
    
    if state["agent_decision"]["action"] == "respond":
        # Direct response (no tool used)
        state["final_response"] = state["agent_decision"]["response"]
        state["workflow_path"].append("synthesize:direct")
        print(f" Direct response generated")
        
    else:
        # Synthesize from tool result
        response = agent.synthesize_response(
            query=state["user_query"],
            tool_result=state["tool_result"],
            chat_history=state["chat_history"]
        )
        state["final_response"] = response
        state["workflow_path"].append("synthesize:from_tool")
        print(f" Response synthesized from tool result")
    
    print(f"   Response: {state['final_response'][:150]}...")
    
    return state


# ============================================================================
# NODE FUNCTIONS (Error Handling)
# ============================================================================

def handle_error(state: AgentState) -> AgentState:
    """
    Handle errors gracefully with user-friendly messages
    
    Converts technical errors into helpful explanations
    """
    print(f"\n{'='*60}")
    print(f"  ERROR HANDLER")
    print(f"{'='*60}")
    print(f"Error category: {state.get('error_category', 'UNKNOWN')}")
    print(f"Error message: {state.get('error_message', 'Unknown error')}")
    
    # Generate user-friendly error message
    response = error_handler.generate_error_message(
        user_query=state["user_query"],
        error=state["error_message"],
        tool_call=state.get("agent_decision")
    )
    
    state["final_response"] = response
    state["workflow_path"].append(f"error:{state.get('error_category', 'unknown')}")
    
    print(f" User-friendly error message generated")
    print(f"   Message: {response}")
    
    return state


# ============================================================================
# SPECIAL DOMAIN HANDLERS
# ============================================================================

def handle_rag_query(state: AgentState) -> AgentState:
    """
    Handle RAG (Retrieval Augmented Generation) queries
    
    Searches internal knowledge base instead of calling APIs
    """
    print(f"\n{'='*60}")
    print(f" RAG QUERY HANDLER")
    print(f"{'='*60}")
    
    # TODO: Implement actual RAG pipeline with Required dsocuments LlamaIndex
    
    
    state["final_response"] = (
        "I can help you search our knowledge base. "
        "However, the RAG system is not yet fully configured. "
        "Please provide your question and I'll do my best to assist."
    )
    state["workflow_path"].append("rag:placeholder")
    
    print(f" RAG system not yet configured ")
    
    return state


def handle_system_search(state: AgentState) -> AgentState:
    """
    Handle system search queries
    
    Searches for request status, available tools, or system metadata
    """
    print(f"\n{'='*60}")
    print(f" SYSTEM SEARCH HANDLER")
    print(f"{'='*60}")
    
    query_lower = state["user_query"].lower()
    
    # Search for available tools
    if any(word in query_lower for word in ["available tools", "what can you do", "capabilities"]):
        from utils.vector_db import list_all_tools
        tools = list_all_tools()
        
        tool_summary = f"I have access to {len(tools)} tools across these domains:\n"
        domain_counts = {}
        for tool in tools:
            domain = tool.get("domain", "GENERAL")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        for domain, count in sorted(domain_counts.items()):
            tool_summary += f"- {domain}: {count} tools\n"
        
        state["final_response"] = tool_summary
        state["workflow_path"].append("system_search:list_tools")
        
    # Search for request status (placeholder)
    elif any(word in query_lower for word in ["status", "request", "history"]):
        state["final_response"] = (
            "To check request status, I'll need more details. "
            "Could you provide a request ID or describe which request you're asking about?"
        )
        state["workflow_path"].append("system_search:status_query")
    
    else:
        state["final_response"] = "I can help you search for available tools or check request status. What would you like to know?"
        state["workflow_path"].append("system_search:general")
    
    print(f" System search completed")
    
    return state


# ============================================================================
# CONDITIONAL EDGE FUNCTIONS
# ============================================================================

def should_use_special_handler(state: AgentState) -> Literal["rag", "system_search", "continue"]:
    """
    Determine if query needs special handling
    
    Routes RAG and SYSTEM_SEARCH queries to dedicated handlers
    """
    domain = state["domain"]
    
    if domain == "RAG_QUERY":
        return "rag"
    elif domain == "SYSTEM_SEARCH":
        return "system_search"
    else:
        return "continue"


def should_call_tool(state: AgentState) -> Literal["call_tool", "synthesize"]:
    """
    Decide whether to execute a tool or respond directly
    """
    action = state["agent_decision"].get("action")
    
    if action == "use_tool":
        return "call_tool"
    else:
        return "synthesize"


def check_tool_result(state: AgentState) -> Literal["error", "success"]:
    """
    Check if tool execution was successful
    """
    if state.get("error_message"):
        return "error"
    
    tool_result = state.get("tool_result", {})
    if not tool_result.get("success"):
        # Set error message if not already set
        if not state.get("error_message"):
            state["error_message"] = tool_result.get("error", "Tool execution failed")
            state["error_category"] = error_handler.categorize_error(state["error_message"])
        return "error"
    
    return "success"


def should_retry_tool(state: AgentState) -> Literal["retry", "give_up"]:
    """
    Decide whether to retry failed tool execution
    """
    attempts = state.get("execution_attempts", 0)
    max_attempts = 2
    
    # Only retry on timeout or connection errors
    error_category = state.get("error_category", "")
    retryable = error_category in ["TIMEOUT", "CONNECTION", "SERVER_ERROR"]
    
    if attempts < max_attempts and retryable:
        print(f" Retrying tool execution (attempt {attempts + 1}/{max_attempts})")
        return "retry"
    else:
        return "give_up"


# ============================================================================
# WORKFLOW CONSTRUCTION
# ============================================================================

def create_workflow() -> StateGraph:
    """
    Create and compile the complete LangGraph workflow
    
    Returns:
        Compiled workflow graph ready for execution
    """
    # Initialize workflow
    workflow = StateGraph(AgentState)
    
    # ========================================================================
    # ADD NODES
    # ========================================================================
    
    # Layer 1: Routing
    workflow.add_node("route", route_query)
    
    # Layer 2: Tool Retrieval
    workflow.add_node("retrieve", retrieve_tools)
    
    # Layer 3: Agent Decision & Execution
    workflow.add_node("decide", agent_decide)
    workflow.add_node("call_tool", call_tool)
    workflow.add_node("synthesize", synthesize_response)
    
    # Error Handling
    workflow.add_node("handle_error", handle_error)
    
    # Special Handlers
    workflow.add_node("rag", handle_rag_query)
    workflow.add_node("system_search", handle_system_search)
    
    # ========================================================================
    # SET ENTRY POINT
    # ========================================================================
    
    workflow.set_entry_point("route")
    
    # ========================================================================
    # ADD EDGES
    # ========================================================================
    
    # From router: check if special handling needed
    workflow.add_conditional_edges(
        "route",
        should_use_special_handler,
        {
            "rag": "rag",
            "system_search": "system_search",
            "continue": "retrieve"
        }
    )
    
    # Standard flow: retrieve -> decide
    workflow.add_edge("retrieve", "decide")
    
    # From decision: call tool or respond directly
    workflow.add_conditional_edges(
        "decide",
        should_call_tool,
        {
            "call_tool": "call_tool",
            "synthesize": "synthesize"
        }
    )
    
    # After tool execution: check result
    workflow.add_conditional_edges(
        "call_tool",
        check_tool_result,
        {
            "error": "handle_error",
            "success": "synthesize"
        }
    )
    
    # ========================================================================
    # TERMINAL EDGES (all paths lead to END)
    # ========================================================================
    
    workflow.add_edge("synthesize", END)
    workflow.add_edge("handle_error", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("system_search", END)
    
    # ========================================================================
    # COMPILE AND RETURN
    # ========================================================================
    
    print("\n" + "="*60)
    print(" LangGraph Workflow Compiled Successfully")
    print("="*60)
    print("\nWorkflow Structure:")
    print("  Entry → route → [rag|system_search|retrieve]")
    print("  retrieve → decide → [call_tool|synthesize]")
    print("  call_tool → [handle_error|synthesize]")
    print("  All paths → END")
    print("="*60 + "\n")
    
    return workflow.compile()


# ============================================================================
# HELPER FUNCTION FOR INITIALIZATION
# ============================================================================

def initialize_state(user_query: str, chat_history: List[Dict[str, str]] = None) -> AgentState:
    """
    Initialize a clean state for a new query
    
    Args:
        user_query: The user's input query
        chat_history: Previous conversation history
        
    Returns:
        Initialized state dictionary
    """
    return {
        "user_query": user_query,
        "chat_history": chat_history or [],
        "domain": "",
        "domain_confidence": 0.0,
        "retrieved_tools": [],
        "retrieval_count": 0,
        "agent_decision": {},
        "selected_tool": {},
        "tool_result": {},
        "execution_attempts": 0,
        "error_message": "",
        "error_category": "",
        "final_response": "",
        "workflow_path": []
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create workflow
    workflow = create_workflow()
    
    # Example query
    test_query = "Send an invoice for $50 to user@example.com"
    
    # Initialize state
    initial_state = initialize_state(test_query)
    
    # Run workflow
    print(f"\n{'='*60}")
    print(f"RUNNING TEST QUERY: {test_query}")
    print(f"{'='*60}\n")
    
    result = workflow.invoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Response: {result['final_response']}")
    print(f"Workflow path: {' → '.join(result['workflow_path'])}")
    print(f"{'='*60}\n")