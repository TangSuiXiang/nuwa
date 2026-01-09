# Nuwa - AI Agent Framework Documentation

Nuwa is a powerful framework for creating AI agents with advanced capabilities including ReAct (Reasoning and Acting) patterns, tool integration, and conversation management. This documentation provides comprehensive guidance on using the Nuwa library.

## Table of Contents

- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Basic Usage](#basic-usage)
  - [Simple LLM Interaction](#simple-llm-interaction)
  - [Chat with Memory](#chat-with-memory)
  - [ReAct Agent](#react-agent)
- [Advanced Features](#advanced-features)
  - [Tool Integration](#tool-integration)
  - [MCP (Model Context Protocol) Support](#mcp-model-context-protocol-support)
  - [Custom System Prompts](#custom-system-prompts)
  - [Streaming Responses](#streaming-responses)
  - [Time Awareness](#time-awareness)
- [Configuration Options](#configuration-options)
- [Examples](#examples)

## Installation

Install Nuwa using pip:

```bash
pip install nuwa
```

Or install from source:

```bash
git clone https://github.com/your-repo/nuwa.git
cd nuwa
pip install .
```

## Core Concepts

### Node Architecture
Nuwa uses a node-based architecture where each component inherits from the `Node` base class. This allows for flexible composition and chaining of different AI capabilities.

### Key Components

- **OpenAI**: Basic LLM interface with OpenAI-compatible APIs
- **ChatLLM**: Enhanced chat functionality with message history management
- **ReActAgent**: Advanced agent implementing the ReAct (Reasoning + Acting) pattern
- **ToolsManager**: Tool registration and execution system
- **MessagesManager**: Message persistence and retrieval system

## Basic Usage

### Simple LLM Interaction

For basic LLM interactions without chat history or tools:

```python
from src.nuwa.llm import OpenAI
import asyncio

async def simple_llm_example():
    llm = OpenAI(
        model="gpt-4",
        system_prompt="You are a helpful assistant.",
        api_key="your-api-key"
    )
    
    async for chunk in llm.run({"user": "Hello, how are you?"}):
        print(chunk.content, end="", flush=True)

# Run the example
asyncio.run(simple_llm_example())
```

### Chat with Memory

For conversational AI with persistent message history:

```python
from src.nuwa.chat import ChatLLM
from src.nuwa.base import MessagesManager
import asyncio

# Custom MessagesManager implementation (or use QdrantMessagesManager)
class SimpleMessagesManager(MessagesManager):
    def __init__(self):
        self.messages = {}
    
    async def get_messages(self, session_id: str, user_input: str = ""):
        return self.messages.get(session_id, [])
    
    async def save_messages(self, session_id: str, messages):
        if session_id not in self.messages:
            self.messages[session_id] = []
        self.messages[session_id].extend(messages)

async def chat_example():
    messages_manager = SimpleMessagesManager()
    chat_llm = ChatLLM(
        model="gpt-4",
        system_prompt="You are a friendly chatbot.",
        api_key="your-api-key",
        messages_manager=messages_manager,
        session_id="user123"
    )
    
    # First message
    async for chunk in chat_llm.run({"user": "What's your name?"}):
        print(chunk.content, end="", flush=True)
    
    print("\n")
    
    # Follow-up message (with context from previous conversation)
    async for chunk in chat_llm.run({"user": "Nice to meet you!"}):
        print(chunk.content, end="", flush=True)

asyncio.run(chat_example())
```

### ReAct Agent

The ReAct agent combines reasoning and tool usage in a structured format:

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def react_agent_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="You are a helpful assistant that can reason and act.",
        api_key="your-api-key",
        max_loop=3  # Maximum reasoning-action loops
    )
    
    async for chunk in agent.run({"user": "What's the weather like today?"}):
        if chunk.state == "END":
            print(f"\nFinal answer: {chunk.content}")
        else:
            print(chunk.content, end="", flush=True)

asyncio.run(react_agent_example())
```

## Advanced Features

### Tool Integration

Nuwa provides a robust tool system for extending agent capabilities:

```python
from src.nuwa.re_act import ReActAgent
from src.nuwa.tool import ToolsManager, ToolObjectParameter, ToolParameter
import asyncio

# Create a tools manager
tools = ToolsManager()

# Register a weather tool
@tools.tool(
    name="get_weather",
    description="Get the weather for a specific city.",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "city": ToolParameter(type="string", description="City name")
        },
    ),
)
async def get_weather(city: str):
    # Simulate API call
    return {"temperature": "22°C", "condition": "Sunny"}

async def tool_integration_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="You are a weather assistant.",
        api_key="your-api-key",
        tools_manager=tools,
        with_time=True  # Include current time in system prompt
    )
    
    async for chunk in agent.run({"user": "What's the weather in Beijing?"}):
        print(chunk.content, end="", flush=True)

asyncio.run(tool_integration_example())
```

### MCP (Model Context Protocol) Support

Nuwa supports MCP for integrating external tools and services:

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def mcp_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="You are a creative assistant.",
        api_key="your-api-key",
        mcp="http://localhost:12345/mcp",  # MCP endpoint
        mcp_timeout=300
    )
    
    async for chunk in agent.run({"user": "Generate an image of a cat"}):
        print(chunk.content, end="", flush=True)

asyncio.run(mcp_example())
```

### Custom System Prompts

System prompts can be dynamically formatted with user-provided variables:

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def custom_prompt_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="You are a {role} with personality: {personality}",
        api_key="your-api-key"
    )
    
    # Provide system variables
    input_data = {
        "user": "Tell me about yourself",
        "system": {
            "role": "catgirl maid",
            "personality": "shy but caring"
        }
    }
    
    async for chunk in agent.run(input_data):
        print(chunk.content, end="", flush=True)

asyncio.run(custom_prompt_example())
```

### Streaming Responses

Enable streaming for real-time response generation:

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def streaming_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="You are a helpful assistant.",
        api_key="your-api-key",
        stream=True,
        extra_body={"enable_thinking": True}  # Enable thinking mode if supported
    )
    
    async for chunk in agent.run({"user": "Explain quantum computing"}):
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(streaming_example())
```

### Time Awareness

Include current timestamp in system prompts for time-sensitive queries:

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def time_aware_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="You are a time-aware assistant.",
        api_key="your-api-key",
        with_time=True  # Automatically adds current time to system prompt
    )
    
    async for chunk in agent.run({"user": "What time is it?"}):
        print(chunk.content, end="", flush=True)

asyncio.run(time_aware_example())
```

## Configuration Options

### OpenAI/ChatLLM/ReActAgent Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | - | LLM model name |
| `system_prompt` | str | - | System prompt template |
| `api_key` | str | - | API key for the LLM service |
| `base_url` | str | "https://api.openai.com/v1" | Base URL for the API |
| `temperature` | float | 0.6 | Sampling temperature |
| `stream` | bool | False | Enable streaming responses |
| `with_time` | bool | False | Include current time in system prompt |
| `with_others` | str | "" | Additional context to include |
| `stop` | List[str] | None | Stop sequences |
| `extra_body` | Dict | None | Additional parameters for the API |

### ReActAgent Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | str | auto-generated | Session identifier |
| `messages_manager` | MessagesManager | None | Message persistence manager |
| `tools_manager` | ToolsManager | None | Tool registry and executor |
| `mcp` | str | None | MCP endpoint URL |
| `mcp_timeout` | int | 300 | MCP request timeout |
| `max_loop` | int | 3 | Maximum ReAct reasoning loops |
| `enable_chat_history` | bool | True | Enable conversation history |
| `enable_selection` | bool | False | Enable user choice requests |
| `hook_tool_call` | Callable | None | Callback function for tool calls |

### ToolsManager

The `ToolsManager` provides decorator-based tool registration:

```python
tools = ToolsManager()

@tools.tool(
    name="calculator",
    description="Perform mathematical calculations",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "expression": ToolParameter(type="string", description="Math expression")
        }
    )
)
def calculator(expression: str):
    return eval(expression)
```

## Examples

### Complete ReAct Agent with Multiple Tools

```python
from src.nuwa.re_act import ReActAgent
from src.nuwa.tool import ToolsManager, ToolObjectParameter, ToolParameter
import asyncio
import json

# Initialize tools manager
tools = ToolsManager()

# Weather tool
@tools.tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "city": ToolParameter(type="string", description="City name")
        }
    )
)
async def get_weather(city: str):
    return {"city": city, "temperature": "25°C", "condition": "Sunny"}

# Calculator tool
@tools.tool(
    name="calculate",
    description="Perform mathematical calculations",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "expression": ToolParameter(type="string", description="Math expression")
        }
    )
)
def calculate(expression: str):
    try:
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

async def complete_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="You are a helpful assistant with access to weather and calculation tools.",
        api_key="your-api-key",
        tools_manager=tools,
        with_time=True,
        stream=True,
        enable_chat_history=True,
        max_loop=5
    )
    
    # Multi-turn conversation
    questions = [
        "What's the weather in Tokyo?",
        "Calculate 15 * 24",
        "Based on the weather and my calculation, should I go outside?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        print("Agent: ", end="")
        async for chunk in agent.run({"user": question}):
            if chunk.state == "END":
                print(f"\nFinal: {chunk.content}")
            else:
                print(chunk.content, end="", flush=True)

asyncio.run(complete_example())
```

### Using with Qdrant for Message Storage

Nuwa includes built-in support for Qdrant vector database for message storage:

```python
from src.nuwa.re_act import ReActAgent
from src.nuwa.qdrant import QdrantMessagesManager
import asyncio

async def qdrant_example():
    # Initialize Qdrant message manager
    messages_manager = QdrantMessagesManager(
        host="localhost",
        port=6333,
        collection_name="chat_history"
    )
    
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="You are a memory-enhanced assistant.",
        api_key="your-api-key",
        messages_manager=messages_manager,
        session_id="user_session_123"
    )
    
    async for chunk in agent.run({"user": "Remember that I like chocolate ice cream"}):
        print(chunk.content, end="", flush=True)
    
    print("\n")
    
    async for chunk in agent.run({"user": "What's my favorite ice cream?"}):
        print(chunk.content, end="", flush=True)

# asyncio.run(qdrant_example())
```

## Best Practices

1. **Session Management**: Always provide a consistent `session_id` for maintaining conversation context
2. **Error Handling**: Implement proper error handling for tool calls and API requests
3. **Rate Limiting**: Be mindful of API rate limits when making multiple requests
4. **Security**: Never expose API keys in client-side code or public repositories
5. **Testing**: Use the provided test suite as a reference for implementation patterns

## Troubleshooting

### Common Issues

- **Tool Not Found**: Ensure tool names match exactly between registration and usage
- **JSON Parsing Errors**: Use `json-repair` library for robust JSON parsing (already included)
- **Streaming Issues**: Set `stream=True` and handle chunks appropriately
- **Memory Leaks**: Clear historical messages periodically or disable chat history when not needed

### Debugging

Enable logging to see detailed information about agent operations:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

This will show tool calls, message flows, and internal agent states.

---

For more examples and advanced usage patterns, refer to the test files in the `tests/` directory of the repository.
