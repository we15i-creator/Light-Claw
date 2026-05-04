ClawLight 🦞

<div align="center">

https://img.shields.io/badge/ClawLight-v1.0-blue
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/python-3.10+-blue.svg
https://img.shields.io/badge/code%20style-black-000000.svg

Lightweight Multi-Agent AI Framework with Intelligent Routing

Fast • Flexible • Open Source

Features • Quick Start • Documentation • Examples • Contributing

</div>

---

📖 Overview

ClawLight is a production-ready, lightweight multi-agent AI framework that intelligently routes queries to specialized agents. Inspired by systems like Claude's tool use and ChatGPT's function calling, ClawLight brings agent-based architecture to any LLM provider.

Why ClawLight?

· 🎯 Smart Routing: Automatically sends queries to the right specialist
· 🔌 Provider Agnostic: Works with Groq, OpenAI, Anthropic, Ollama, or any OpenAI-compatible API
· 🛠️ Extensible Tools: Built-in search, calculator, file operations + custom tools
· ⚡ Blazing Fast: Async architecture with streaming support
· 🪶 Lightweight: Single file, minimal dependencies, ~500 lines of clean code
· 🔒 Privacy First: Full support for local models via Ollama

✨ Features

🧠 Intelligent Agent System

· Router Agent: Automatically classifies queries and routes to specialists
· Research Agent: Web search, fact-finding, current events
· Code Agent: Programming, debugging, code generation
· Writer Agent: Content creation, editing, communication
· Analyst Agent: Data analysis, calculations, problem-solving

🔧 Tool Ecosystem

Tool Description Default
web_search DuckDuckGo search integration ✅
calculator Safe math expression evaluation ✅
read_file File content reader ✅
custom_tools Register your own tools ✅

🌐 Provider Support

Provider Type Speed Setup
Groq Cloud 🚀 Fastest API Key
OpenAI Cloud 🏃 Fast API Key
Ollama Local 💻 Variable Local Install
Custom Any - OpenAI-compatible endpoint

🚀 Quick Start

Prerequisites

· Python 3.10 or higher
· pip package manager

Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clawlight.git
cd clawlight

# Install dependencies
pip install -r requirements.txt
```

Basic Usage

```python
import asyncio
from clawlight import ClawLight

async def main():
    # Initialize with Groq (free tier available)
    claw = ClawLight(
        provider_name="groq",
        api_key="your-groq-api-key"  # or set GROQ_API_KEY env variable
    )
    
    # Simple query - automatically routed to best agent
    response = await claw.chat("What's the latest in AI research?")
    print(response)
    
    # Force specific agent
    code = await claw.chat("Write a sorting algorithm", force_agent="coder")
    print(code)

asyncio.run(main())
```

Environment Setup

```bash
# For Groq (recommended - fastest)
export GROQ_API_KEY="gsk_your_api_key_here"

# For OpenAI
export OPENAI_API_KEY="sk_your_api_key_here"

# For Ollama (local - no API key needed)
ollama pull llama3.1:8b
```

📚 Documentation

Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Router    │ ← Classifies intent
└──────┬──────┘
       │
       ├──→ Researcher ──→ Web Search
       ├──→ Coder      ──→ File Read, Calculator
       ├──→ Writer     ──→ Web Search
       └──→ Analyst    ──→ Calculator, Web Search
           │
           ▼
      Final Response
```

Agent Configuration

```python
from clawlight import ClawLight, AgentConfig, AgentRole

# Custom agent configuration
claw = ClawLight(provider_name="groq")

# Add a custom specialist
custom_agent = AgentConfig(
    name="MedicalExpert",
    role=AgentRole.ANALYST,
    system_prompt="You are a medical expert...",
    model="llama-3.1-70b-versatile",  # Use larger model for complex tasks
    temperature=0.3,                   # Lower temperature for factual accuracy
    tools=["web_search", "calculator"]
)
```

Streaming Responses

```python
# Stream responses in real-time
async for chunk in claw.chat_stream("Write a poem about AI"):
    if chunk["type"] == "content":
        print(chunk["data"], end="", flush=True)
    elif chunk["type"] == "tool_result":
        print(f"\n🔍 Using {chunk['name']}: {chunk['data']}")
```

Custom Tools

```python
# Register your own tools
async def fetch_api_data(endpoint: str, params: dict = None):
    """Custom API fetcher"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as resp:
            return await resp.json()

claw.register_tool(
    name="api_fetcher",
    description="Fetch data from external APIs",
    parameters={
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": "API endpoint URL"
            },
            "params": {
                "type": "object",
                "description": "Query parameters"
            }
        },
        "required": ["endpoint"]
    },
    handler=fetch_api_data
)
```

Local Models with Ollama

```python
# Completely offline, private AI
claw = ClawLight(
    provider_name="ollama",
    base_url="http://localhost:11434"  # default
)

# Use any Ollama model
response = await claw.chat(
    "Summarize this document",
    # Model can be set per agent in config
)
```

🔥 Examples

Example 1: Research Assistant

```python
claw = ClawLight(provider_name="groq")

# Multi-tool research
result = await claw.chat(
    "Research quantum computing breakthroughs in 2024, "
    "calculate the potential speedup over classical computers, "
    "and write a summary"
)
```

Example 2: Code Review Bot

```python
# Automated code review
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

review = await claw.chat(
    f"Review this code and suggest improvements:\n{code}",
    force_agent="coder"
)
```

Example 3: Content Pipeline

```python
# Research → Write → Analyze pipeline
topic = "sustainable energy innovations"

# Step 1: Research
research = await claw.chat(f"Research latest in {topic}", 
                           force_agent="researcher")

# Step 2: Write
article = await claw.chat(f"Write article based on: {research}", 
                          force_agent="writer")

# Step 3: Analyze
metrics = await claw.chat(f"Analyze readability and SEO: {article}", 
                          force_agent="analyst")
```

🏗️ Project Structure

```
clawlight/
├── clawlight.py          # Main framework (single file!)
├── requirements.txt      # Dependencies
├── examples/
│   ├── basic_chat.py     # Simple chat examples
│   ├── streaming.py      # Streaming demos
│   ├── custom_tools.py   # Tool creation examples
│   └── pipeline.py       # Multi-agent pipelines
├── tests/
│   └── test_clawlight.py # Unit tests
└── README.md
```

🤝 Contributing

We welcome contributions! Here's how you can help:

Getting Started

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

Development Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/yourusername/clawlight.git
cd clawlight
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black clawlight.py
```

What We Need Help With

· 🐛 Bug fixes and error handling improvements
· 🔌 Additional LLM provider integrations
· 🛠️ New built-in tools (databases, APIs, file operations)
· 📚 Documentation and examples
· 🧪 Test coverage
· 🎨 Web UI for agent visualization

🗺️ Roadmap

v1.1 (Coming Soon)

· Memory persistence (conversation history)
· Parallel agent execution
· Rate limiting and retry logic
· Anthropic Claude provider

v1.2

· Web UI dashboard
· Agent team collaboration
· Tool marketplace
· Docker deployment

v2.0

· Autonomous agent loops
· Visual workflow builder
· Enterprise features (RBAC, audit logs)
· gRPC API server

⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

· Inspired by OpenAI's function calling
· Built with aiohttp for async operations
· Search powered by DuckDuckGo
· Special thanks to all contributors

💬 Community

· Discord Server - Chat with the community
· Twitter - Updates and announcements
· GitHub Discussions - Questions and ideas

⭐ Support the Project

If you find ClawLight useful, please consider:

· Starring the repository ⭐
· Sharing with your network
· Contributing to the codebase
· Sponsoring development

---

<div align="center">

Built with ❤️ by the open source community

ClawLight - Light enough to run anywhere, powerful enough to handle anything

</div>

---

📊 Performance Benchmarks

Provider Model Latency (first token) Throughput Cost/1M tokens
Groq Llama 3.1 8B ~50ms ~800 t/s Free tier
Groq Llama 3.1 70B ~100ms ~300 t/s $0.59/$0.79
OpenAI GPT-4o ~500ms ~100 t/s $5/$15
Ollama Llama 3.1 8B ~200ms ~50 t/s Free (local)

Benchmarks may vary based on hardware and network conditions

🔐 Security

· API Keys: Never hardcode API keys. Use environment variables or secure vaults
· Tool Safety: Calculator uses restricted eval() with math functions only
· File Access: File read tool respects OS-level permissions
· Local Mode: Ollama provider ensures complete data privacy

❓ FAQ

Q: Can I use this in production?
A: Yes! ClawLight is designed for production use with error handling and async support.

Q: How is this different from LangChain?
A: ClawLight is significantly lighter (~500 lines vs 100k+), faster to implement, and easier to understand.

Q: Does it support vision/multimodal?
A: Text-only currently. Vision support planned for v1.2.

Q: Can agents communicate with each other?
A: Currently sequential. Agent-to-agent communication planned for v1.2.

---

<div align="center">

Get Started • Documentation • Examples • Contribute

</div>
