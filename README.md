## Architecture Overview

This system implements a 3-layer hierarchical architecture for managing 50+ API tools efficiently:

**Layer 1: Global Router** - Domain classification (5-10 domains)
**Layer 2: Vectorized Tool Indexer** - Semantic tool retrieval
**Layer 3: Specialized Agent** - Tool execution and response synthesis

## Quick Start

\`\`\`bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Run setup script
bash setup.sh

# 4. Start the system
python main.py
\`\`\`

## Project Structure

\`\`\`
├── router/
│   └── global_router.py      # Layer 1: Domain classification
├── indexer/
│   └── tool_indexer.py       # Layer 2: Vector-based tool retrieval
├── agent/
│   └── specialized_agent.py  # Layer 3: Tool execution agent
├── graph/
│   └── langgraph_workflow.py # State machine workflow
├── tools/
│   ├── tool_parser.py        # Postman collection parser
│   └── api_executor.py       # API call executor
├── utils/
│   ├── vector_db.py          # ChromaDB integration
│   └── error_handler.py      # Error handling utilities
├── config.py                 # Configuration
└── main.py                   # Entry point
\`\`\`

## Configuration

1. Place your Postman collections in \`./postman_collections/\`
2. Configure domains in \`config.py\`
3. Set up your LLM API keys in \`.env\`

## Features

- ✅ Hierarchical routing for 1000+ tools
- ✅ Vector-based semantic tool search
- ✅ Multi-turn conversation support
- ✅ Comprehensive error handling
- ✅ LangSmith observability
- ✅ RAG pipeline integration
- ✅ System search capabilities
