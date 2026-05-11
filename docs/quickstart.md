# Nuwa Quickstart Guide

Get up and running with Nuwa in minutes! This guide covers the essential steps to create your first AI agent.

## Installation

### Prerequisites
- Python 3.10 or higher
- An API key from an OpenAI-compatible LLM provider (OpenAI, Anthropic, DeepSeek, etc.)

### Install Nuwa
```bash
pip install nuwa
```

Or install from source:
```bash
git clone https://github.com/your-repo/nuwa.git
cd nuwa
pip install .
```

## Your First Agent

### 1. Simple LLM Interaction

Create a basic agent that responds to user input:

```python
# simple_agent.py
import asyncio
from src.nuwa.llm import LLMNode

async def main():
    # Initialize the LLM
    llm = LLMNode(
        model="gpt-4",
        system_prompt="You are a helpful assistant.",
        api_key="your-api-key-here"
    )
    
    # Get user input
    user_input = "Hello! What can you do?"
    
    # Generate response
    print("Agent: ", end="")
    async for chunk in llm.run({"user": user_input}):
        print(chunk.content, end="", flush=True)
    print()  # New line at the end

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python simple_agent.py
```

### 2. Chat Agent with Memory

Create a conversational agent that remembers previous interactions:

```python
# chat_agent.py
import asyncio
from src.nuwa.chat import ConversationAgent
from src.nuwa.storages import ConversationStorage

class SimpleMemory(ConversationStorage):
    def __init__(self):
        self.messages = {}
    
    async def get_messages(self, session_id: str, user_input: str = ""):
        return self.messages.get(session_id, [])
    
    async def save_messages(self, session_id: str, messages):
        if session_id not in self.messages:
            self.messages[session_id] = []
        self.messages[session_id].extend(messages)

async def main():
    # Initialize with memory
    memory = SimpleMemory()
    chat_agent = ConversationAgent(
        model="gpt-4",
        system_prompt="You are a friendly conversational AI.",
        api_key="your-api-key-here",
        messages_manager=memory,
        session_id="demo_session"
    )
    
    # First interaction
    print("User: What's your name?")
    print("Agent: ", end="")
    async for chunk in chat_agent.run({"user": "What's your name?"}):
        print(chunk.content, end="", flush=True)
    print("\n")
    
    # Second interaction (with context)
    print("User: Nice to meet you!")
    print("Agent: ", end="")
    async for chunk in chat_agent.run({"user": "Nice to meet you!"}):
        print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Reasoning and Acting Agent with Tools

Create an advanced agent that can reason and use tools:

```python
# reasoning_agent.py
import asyncio
from src.nuwa.react_agent import ReasoningActingAgent
from src.nuwa.tool import ToolRegistry, ToolObjectParameter, ToolParameter

# Create tools registry
tools = ToolRegistry()

# Register a calculator tool
@tools.tool(
    name="calculate",
    description="Perform mathematical calculations",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "expression": ToolParameter(
                type="string", 
                description="Mathematical expression to evaluate"
            )
        }
    )
)
def calculate(expression: str):
    try:
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

async def main():
    # Initialize ReasoningActing agent with tools
    agent = ReasoningActingAgent(
        model="gpt-4",
        system_prompt="You are a helpful assistant with calculation capabilities.",
        api_key="your-api-key-here",
        tools_manager=tools,
        with_time=True,  # Include current time
        stream=True,     # Enable streaming
        max_loop=5       # Allow up to 5 reasoning loops
    )
    
    # Ask a question that requires tool usage
    question = "What is 123 * 456 + 789?"
    print(f"User: {question}")
    print("Agent: ", end="")
    
    async for chunk in agent.run({"user": question}):
        if chunk.state == "END":
            print(f"\nFinal Answer: {chunk.content}")
        else:
            print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Tips

### API Keys and Providers

Nuwa works with any OpenAI-compatible API. Configure your provider:

```python
# OpenAI
agent = ReasoningActingAgent(
    model="gpt-4",
    api_key="sk-your-openai-key",
    base_url="https://api.openai.com/v1"
)

# Anthropic (via OpenAI compatibility layer)
agent = ReasoningActingAgent(
    model="claude-3-opus-20240229",
    api_key="your-anthropic-key",
    base_url="https://api.anthropic.com/v1/messages"  # Adjust as needed
)

# DeepSeek
agent = ReasoningActingAgent(
    model="deepseek-v3.2",
    api_key="your-deepseek-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

### Environment Variables

Store your API key securely using environment variables:

```python
import os
from src.nuwa.react_agent import ReasoningActingAgent

agent = ReasoningActingAgent(
    model="gpt-4",
    system_prompt="You are a helpful assistant.",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
```

Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
python your_agent.py
```

## Common Patterns

### Pattern 1: Multi-turn Conversation
```python
async def multi_turn_conversation():
    agent = ReasoningActingAgent(
        model="gpt-4",
        system_prompt="You are a helpful assistant.",
        api_key="your-key",
        enable_chat_history=True
    )
    
    conversation = [
        "What's the capital of France?",
        "How many people live there?",
        "What are some famous landmarks?"
    ]
    
    for message in conversation:
        print(f"User: {message}")
        print("Agent: ", end="")
        async for chunk in agent.run({"user": message}):
            print(chunk.content, end="", flush=True)
        print("\n")
```

### Pattern 2: Custom Tool with Async Function
```python
@tools.tool(
    name="fetch_weather",
    description="Get current weather information",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "city": ToolParameter(type="string", description="City name")
        }
    )
)
async def fetch_weather(city: str):
    # Simulate async API call
    await asyncio.sleep(1)
    return {"city": city, "temperature": "22°C", "condition": "Sunny"}
```

### Pattern 3: Streaming with Thinking Mode
```python
agent = ReasoningActingAgent(
    model="deepseek-v3.2",
    system_prompt="Think step by step before answering.",
    api_key="your-key",
    extra_body={"enable_thinking": True},
    stream=True,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

async for chunk in agent.run({"user": "Explain quantum entanglement"}):
    print(chunk.content, end="", flush=True)
```

## Troubleshooting

### Common Issues and Solutions

**Issue**: `RateLimitError`
- **Solution**: Add retry logic or reduce request frequency
```python
import asyncio
from openai import RateLimitError

async def safe_run(agent, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            async for chunk in agent.run(input_data):
                yield chunk
            break
        except RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

**Issue**: Tool not found
- **Solution**: Ensure tool names match exactly and are registered before agent initialization

**Issue**: JSON parsing errors
- **Solution**: The framework uses `json-repair` internally, but ensure your tool returns valid JSON-like structures

### Debugging

Enable debug logging to see internal operations:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now you'll see detailed logs of tool calls, message flows, etc.
```

## Next Steps

1. **Explore the Examples**: Check the `tests/` directory for more comprehensive examples
2. **Custom ConversationStorage**: Implement persistent storage with databases like Qdrant
3. **Advanced Tools**: Create tools that integrate with external APIs and services
4. **MCP Integration**: Connect to Model Context Protocol endpoints for extended capabilities
5. **Performance Optimization**: Tune parameters like `max_loop`, `temperature`, and streaming settings

## Resources

- [Full Documentation](README.md)
- [API Reference](api_reference.md)
- [Test Examples](../../tests/)
- [Source Code](../../src/nuwa/)

Happy building with Nuwa! 🎉
