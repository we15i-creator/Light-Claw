"""
ClawLight - Lightweight Multi-Agent System with Provider Flexibility
Supports: Groq, OpenAI, Anthropic, Ollama
Features: Agent routing, web search, tool use, streaming
"""

import os
import json
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Data Models ==============

class AgentRole(Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    WRITER = "writer"
    ANALYST = "analyst"
    ROUTER = "router"


@dataclass
class Message:
    role: str
    content: str
    name: Optional[str] = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentConfig:
    name: str
    role: AgentRole
    system_prompt: str
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    tools: List[str] = field(default_factory=list)


# ============== Provider Abstraction ==============

class LLMProvider:
    """Base provider class"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv(f"{self.provider_name.upper()}_API_KEY")
        self.base_url = base_url

    @property
    def provider_name(self) -> str:
        raise NotImplementedError

    async def generate(self, 
                      messages: List[Dict], 
                      model: str,
                      temperature: float = 0.7,
                      max_tokens: int = 4096,
                      tools: List[Dict] = None,
                      stream: bool = False) -> Dict | AsyncGenerator:
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq provider - fastest inference"""
    
    @property
    def provider_name(self) -> str:
        return "groq"

    async def generate(self, messages, model, temperature=0.7, max_tokens=4096, tools=None, stream=False):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if stream:
                    return self._stream_response(response)
                return await response.json()

    async def _stream_response(self, response):
        async for line in response.content:
            if line.startswith(b'data: '):
                data = line[6:]
                if data == b'[DONE]':
                    break
                yield json.loads(data)


class OpenAIProvider(LLMProvider):
    """OpenAI provider"""
    
    @property
    def provider_name(self) -> str:
        return "openai"

    async def generate(self, messages, model, temperature=0.7, max_tokens=4096, tools=None, stream=False):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if stream:
                    return self._stream_response(response)
                return await response.json()

    async def _stream_response(self, response):
        async for line in response.content:
            if line.startswith(b'data: '):
                data = line[6:]
                if data == b'[DONE]':
                    break
                yield json.loads(data)


class OllamaProvider(LLMProvider):
    """Local Ollama provider"""
    
    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(self, messages, model, temperature=0.7, max_tokens=4096, tools=None, stream=False):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url or 'http://localhost:11434'}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            ) as response:
                if stream:
                    return self._stream_response(response)
                return await response.json()

    async def _stream_response(self, response):
        async for line in response.content:
            if line:
                yield json.loads(line)


# ============== Tool System ==============

class ToolRegistry:
    """Register and manage tools"""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools"""
        self.register_tool(
            name="web_search",
            description="Search the web for current information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            handler=self._web_search
        )
        
        self.register_tool(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
            handler=self._calculator
        )
        
        self.register_tool(
            name="read_file",
            description="Read content from a file",
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to file"
                    }
                },
                "required": ["filepath"]
            },
            handler=self._read_file
        )

    def register_tool(self, name: str, description: str, parameters: Dict, handler: Callable):
        """Register a custom tool"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
    
    def get_tool_definitions(self, tool_names: List[str] = None) -> List[Dict]:
        """Get tool definitions for LLM"""
        if tool_names is None:
            return [{"type": "function", "function": {k: v} for k, v in t.items() if k != 'handler'} 
                   for t in self.tools.values()]
        
        tools = []
        for name in tool_names:
            if name in self.tools:
                tool = self.tools[name]
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                })
        return tools

    async def execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a tool and return result"""
        if tool_name not in self.tools:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})
        
        try:
            handler = self.tools[tool_name]["handler"]
            result = await handler(**arguments) if asyncio.iscoroutinefunction(handler) else handler(**arguments)
            return json.dumps(result) if isinstance(result, dict) else str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _web_search(self, query: str, num_results: int = 5) -> Dict:
        """Perform web search using DuckDuckGo"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1}
                ) as response:
                    data = await response.json()
                    
                    results = []
                    if data.get("Abstract"):
                        results.append({
                            "title": "Abstract",
                            "snippet": data["Abstract"],
                            "url": data.get("AbstractURL", "")
                        })
                    
                    for topic in data.get("RelatedTopics", [])[:num_results]:
                        if isinstance(topic, dict) and "Text" in topic:
                            results.append({
                                "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                "snippet": topic["Text"],
                                "url": topic.get("FirstURL", "")
                            })
                    
                    return {"query": query, "results": results[:num_results]}
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    def _calculator(self, expression: str) -> Dict:
        """Safe calculator"""
        try:
            # Safe eval with math functions
            import math
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e)}

    def _read_file(self, filepath: str) -> Dict:
        """Read file contents"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            return {"filepath": filepath, "content": content, "size": len(content)}
        except Exception as e:
            return {"error": str(e)}


# ============== Agent System ==============

class Agent:
    """Individual agent with specific role and capabilities"""
    
    def __init__(self, 
                 config: AgentConfig, 
                 provider: LLMProvider, 
                 tool_registry: ToolRegistry,
                 max_iterations: int = 5):
        self.config = config
        self.provider = provider
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.conversation_history = []
    
    async def run(self, user_message: str, stream: bool = False) -> AsyncGenerator | str:
        """Run agent with tool use loop"""
        self.conversation_history = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        if stream:
            return self._stream_run()
        
        return await self._normal_run()

    async def _normal_run(self) -> str:
        """Run without streaming"""
        for _ in range(self.max_iterations):
            response = await self.provider.generate(
                messages=self.conversation_history,
                model=self.config.model,
                temperature=self.config.temperature,
                tools=self.tool_registry.get_tool_definitions(self.config.tools)
            )
            
            assistant_message = response["choices"][0]["message"]
            self.conversation_history.append(assistant_message)
            
            # Check for tool calls
            if "tool_calls" in assistant_message:
                for tool_call in assistant_message["tool_calls"]:
                    function = tool_call["function"]
                    result = await self.tool_registry.execute_tool(
                        function["name"],
                        json.loads(function["arguments"])
                    )
                    self.conversation_history.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call["id"]
                    })
            else:
                return assistant_message["content"]
        
        return "Max iterations reached without final response."

    async def _stream_run(self):
        """Run with streaming"""
        for iteration in range(self.max_iterations):
            stream = await self.provider.generate(
                messages=self.conversation_history,
                model=self.config.model,
                temperature=self.config.temperature,
                tools=self.tool_registry.get_tool_definitions(self.config.tools),
                stream=True
            )
            
            full_message = {"role": "assistant", "content": ""}
            tool_calls = []
            
            async for chunk in stream:
                if "choices" not in chunk or not chunk["choices"]:
                    continue
                    
                delta = chunk["choices"][0].get("delta", {})
                
                if "content" in delta and delta["content"]:
                    full_message["content"] += delta["content"]
                    yield {"type": "content", "data": delta["content"]}
                
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        
                        if "id" in tc:
                            tool_calls[idx]["id"] = tc["id"]
                        if "function" in tc:
                            if "name" in tc["function"]:
                                tool_calls[idx]["function"]["name"] += tc["function"]["name"]
                            if "arguments" in tc["function"]:
                                tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
            
            if tool_calls:
                full_message["tool_calls"] = tool_calls
                self.conversation_history.append(full_message)
                
                for tool_call in tool_calls:
                    function = tool_call["function"]
                    result = await self.tool_registry.execute_tool(
                        function["name"],
                        json.loads(function["arguments"])
                    )
                    yield {"type": "tool_result", "name": function["name"], "data": result}
                    self.conversation_history.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call["id"]
                    })
            else:
                self.conversation_history.append(full_message)
                break
        
        yield {"type": "done", "data": None}


class AgentRouter:
    """Routes queries to appropriate specialized agents"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        
        self.router_prompt = """You are an intelligent router. Analyze the user's query and determine which specialized agent would be best suited to handle it.

Available agents:
- researcher: For research, fact-finding, current events, and web searches
- coder: For programming, debugging, code generation, and technical questions
- writer: For creative writing, content creation, editing, and communication
- analyst: For data analysis, calculations, logic, and complex problem-solving
- router: For general conversation and unclear queries

Respond ONLY with the agent name that best matches the query."""

    async def route(self, query: str) -> str:
        """Determine which agent to use"""
        response = await self.provider.generate(
            messages=[
                {"role": "system", "content": self.router_prompt},
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=10
        )
        
        agent_name = response["choices"][0]["message"]["content"].strip().lower()
        valid_agents = ["researcher", "coder", "writer", "analyst", "router"]
        
        return agent_name if agent_name in valid_agents else "router"


# ============== Main System ==============

class ClawLight:
    """Main multi-agent system with routing"""
    
    def __init__(self, provider_name: str = "groq", api_key: str = None, base_url: str = None):
        # Initialize provider
        self.provider = self._get_provider(provider_name, api_key, base_url)
        self.tool_registry = ToolRegistry()
        self.router = AgentRouter(self.provider)
        
        # Create specialized agents
        self.agents = {
            "researcher": Agent(
                AgentConfig(
                    name="Researcher",
                    role=AgentRole.RESEARCHER,
                    system_prompt="""You are a research specialist. Your role is to:
- Find accurate and current information
- Use web search when needed for up-to-date facts
- Synthesize information from multiple sources
- Cite sources clearly
- Be thorough but concise in your findings""",
                    tools=["web_search"]
                ),
                self.provider,
                self.tool_registry
            ),
            "coder": Agent(
                AgentConfig(
                    name="Coder",
                    role=AgentRole.CODER,
                    system_prompt="""You are a programming expert. Your role is to:
- Write clean, efficient, well-documented code
- Debug and explain technical issues
- Use best practices and design patterns
- Provide complete solutions with explanations
- Read files when needed for context""",
                    tools=["read_file", "calculator"]
                ),
                self.provider,
                self.tool_registry
            ),
            "writer": Agent(
                AgentConfig(
                    name="Writer",
                    role=AgentRole.WRITER,
                    system_prompt="""You are a professional writer. Your role is to:
- Create engaging, well-structured content
- Adapt tone and style to the audience
- Edit and improve text clarity
- Be creative and original
- Consider SEO and readability""",
                    tools=["web_search"]
                ),
                self.provider,
                self.tool_registry
            ),
            "analyst": Agent(
                AgentConfig(
                    name="Analyst",
                    role=AgentRole.ANALYST,
                    system_prompt="""You are a data analyst. Your role is to:
- Break down complex problems logically
- Perform calculations and data analysis
- Identify patterns and insights
- Present findings clearly
- Think step-by-step for accuracy""",
                    tools=["calculator", "web_search"]
                ),
                self.provider,
                self.tool_registry
            )
        }
    
    def _get_provider(self, provider_name: str, api_key: str, base_url: str) -> LLMProvider:
        """Factory for creating providers"""
        providers = {
            "groq": GroqProvider,
            "openai": OpenAIProvider,
            "ollama": OllamaProvider
        }
        
        provider_class = providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        return provider_class(api_key=api_key, base_url=base_url)

    async def chat(self, query: str, stream: bool = False, force_agent: str = None):
        """Process query with intelligent routing"""
        # Route to appropriate agent
        agent_name = force_agent or await self.router.route(query)
        logger.info(f"Routing to {agent_name} agent")
        
        agent = self.agents.get(agent_name, self.agents["router"])
        return await agent.run(query, stream=stream)

    async def chat_stream(self, query: str, force_agent: str = None):
        """Stream chat with intelligent routing"""
        agent_name = force_agent or await self.router.route(query)
        logger.info(f"Routing to {agent_name} agent")
        
        agent = self.agents.get(agent_name, self.agents["router"])
        async for chunk in await agent.run(query, stream=True):
            yield chunk

    def register_tool(self, name: str, description: str, parameters: Dict, handler: Callable):
        """Register custom tool"""
        self.tool_registry.register_tool(name, description, parameters, handler)


# ============== Example Usage ==============

async def main():
    # Initialize with Groq (fastest)
    # Set GROQ_API_KEY in environment or pass directly
    claw = ClawLight(provider_name="groq")
    
    # Or use local Ollama
    # claw = ClawLight(provider_name="ollama")
    
    # Example 1: Basic chat with routing
    print("\n=== Example 1: Research Query ===")
    response = await claw.chat("What are the latest developments in quantum computing?")
    print(f"Response: {response}\n")
    
    # Example 2: Code query
    print("=== Example 2: Code Query ===")
    response = await claw.chat("Write a Python function to find prime numbers using Sieve of Eratosthenes")
    print(f"Response: {response}\n")
    
    # Example 3: Streaming with forced agent
    print("=== Example 3: Stream with Writer Agent ===")
    async for chunk in await claw.chat("Write a haiku about artificial intelligence", force_agent="writer"):
        if chunk["type"] == "content":
            print(chunk["data"], end="", flush=True)
    print("\n")
    
    # Example 4: Custom tool registration
    def custom_translator(text: str, target_lang: str = "es") -> Dict:
        """Simple mock translator"""
        return {"original": text, "translated": f"[{target_lang} translation of '{text}']", "target_lang": target_lang}
    
    claw.register_tool(
        name="translate",
        description="Translate text to target language",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_lang": {"type": "string", "description": "Target language code"}
            },
            "required": ["text"]
        },
        handler=custom_translator
    )


if __name__ == "__main__":
    asyncio.run(main())
