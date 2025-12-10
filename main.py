"""
Main entry point for the Hierarchical Agentic System
"""
import os
from dotenv import load_dotenv
from graph.lang_graph import create_workflow
from tools.tool_parser import ToolParser
from indexer.tool_indexer import ToolIndexer
from config import tool_config, observability_config

load_dotenv()

def setup_system():
    """Initialize the system: parse tools and build vector index"""
    print(" Initializing Hierarchical Agentic System...")
    
    # Step 1: Parse Postman collections
    print("Parsing Postman collections...")
    parser = ToolParser(tool_config.POSTMAN_COLLECTIONS_DIR)
    tools = parser.parse_all_collections()
    print(f"Parsed {len(tools)} tools from Postman collections")
    
    # Step 2: Build vector index
    print("Building vector index...")
    indexer = ToolIndexer()
    indexer.index_tools(tools)
    print("Vector index built successfully")
    
    return tools

def run_interactive():
    """Run interactive chat loop"""
    workflow = create_workflow()
    chat_history = []
    
    print("\\n" + "="*60)
    print("🤖 Agentic System Ready!")
    print("="*60)
    print("Type your queries below (or 'quit' to exit)\\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
        
        try:
            result = workflow.invoke({
                "user_query": user_input,
                "chat_history": chat_history,
                "error_message": None
            })
            
            response = result.get("final_response", "I apologize, but I couldn't process your request.")
            print(f"\\nAssistant: {response}\\n")
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print(f"\\n Error: {str(e)}\\n")

def main():
    # Setup system
    tools = setup_system()
    
    # Run interactive mode
    run_interactive()

if __name__ == "__main__":
    main()