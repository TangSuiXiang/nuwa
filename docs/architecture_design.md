# Nuwa Architecture and Design Patterns

Nuwa is built on a sophisticated architecture that combines modern design patterns with practical AI agent development principles. This document provides a comprehensive overview of the system's architecture, design patterns, and component relationships.

## Overall Architecture

Nuwa follows a **layered architecture** combined with a **Directed Acyclic Graph (DAG) processing pipeline**. The core philosophy is that everything in Nuwa is a `Node` that can be chained together to create complex AI agent workflows.

### Key Architectural Principles

1. **Composability**: Components can be easily combined and extended
2. **Separation of Concerns**: Each layer has distinct responsibilities
3. **Extensibility**: Easy to add new components without modifying existing code
4. **Streaming Support**: Native support for real-time response generation
5. **Pluggable Storage**: Message persistence can use different backends

## Core Layers

### 1. Base Layer (`src/nuwa/base.py`)

The foundation of Nuwa's architecture, defining abstract interfaces:

- **`Node`**: Abstract base class for all processing components
  - Implements dependency management via `deps` list
  - Supports chaining with `>` operator (`__gt__` method)
  - Defines abstract `run()` method for processing logic
  
- **`MessagesManager`**: Abstract interface for message persistence
  - `get_messages()`: Retrieve conversation history
  - `save_messages()`: Persist messages to storage
  
- **`InputChunk`**: Data structure for streaming input/output
  - Contains `state` field (`DOING`, `DONE`, `END`)
  - Contains `content` field for actual data

### 2. LLM Layer (`src/nuwa/llm.py`)

Provides basic Large Language Model interaction capabilities:

- **`OpenAI`**: Concrete implementation of `Node`
  - Handles API communication with OpenAI-compatible endpoints
  - Supports both streaming and non-streaming modes
  - Manages system prompts with dynamic templating
  - Includes time awareness and custom context injection

### 3. Chat Layer (`src/nuwa/chat.py`)

Extends LLM capabilities with conversation management:

- **`ChatLLM`**: Inherits from `OpenAI`
  - Integrates with `MessagesManager` for conversation history
  - Manages session IDs for multi-user support
  - Handles tool calling through `ToolsManager`
  - Supports MCP (Model Context Protocol) integration
  - Provides hook mechanism for custom tool call handling

### 4. ReAct Layer (`src/nuwa/re_act.py`)

Implements the ReAct (Reasoning + Acting) pattern:

- **`ReActAgent`**: Inherits from `ChatLLM`
  - Parses structured output format: `<Q>question</Q><T>thought</T><A>action</A><O>observation</O>`
  - Supports multiple reasoning-action cycles (`max_loop` parameter)
  - Built-in `answer` tool for final responses
  - Optional `request_user_choice` tool for user interaction
  - Sophisticated JSON parsing with error recovery using `json-repair`

### 5. Tool Layer (`src/nuwa/tool.py`)

Manages external function integration:

- **`ToolsManager`**: Central registry for tools
  - Decorator-based tool registration (`@tools.tool()`)
  - Automatic parameter inference from function annotations
  - Supports both synchronous and asynchronous functions
  - Handles MCP tool discovery and integration
  
- **Tool Entities**: Structured tool definitions
  - `ToolEntity`: Main tool definition with name, parameters, description
  - `ToolParameter`: Parameter specifications with type and description
  - `ToolObjectParameter`: Complex object parameters
  - `ToolArrayParameter`: Array parameters

### 6. Storage Layer (`src/nuwa/qdrant.py`)

Provides concrete message persistence implementation:

- **`QdrantMessagesManager`**: Implements `MessagesManager`
  - Uses Qdrant vector database for message storage
  - Supports efficient retrieval and persistence
  - Can be replaced with other storage backends

### 7. Agent Group Layer (`src/nuwa/agent_group.py`)

Specialized agent for multi-agent scenarios:

- **`GroupAgent`**: Inherits from `ReActAgent`
  - Supports role-based agent specialization
  - Includes role blacklisting to prevent self-reference
  - Maintains group context awareness

## Design Patterns

### 1. Abstract Factory Pattern

The `MessagesManager` abstract base class allows pluggable storage implementations:

```python
# Abstract interface
class MessagesManager(ABC):
    @abstractmethod
    async def get_messages(self, session_id: str, user_input: str = ""):
        pass
    
    @abstractmethod
    async def save_messages(self, session_id: str, messages):
        pass

# Concrete implementations
class QdrantMessagesManager(MessagesManager):
    # Qdrant-specific implementation
    
class SimpleMessagesManager(MessagesManager):
    # In-memory implementation
```

### 2. Strategy Pattern

Different LLM providers can be implemented as strategies:

```python
class OpenAI(Node):
    # OpenAI-compatible implementation
    
class Anthropic(Node):
    # Could be added for Anthropic support
```

### 3. Decorator Pattern

Tool registration uses decorator pattern for clean syntax:

```python
tools = ToolsManager()

@tools.tool(
    name="get_weather",
    description="Get weather for a city",
    parameters=ToolObjectParameter(...)
)
def get_weather(city: str):
    return {"weather": "sunny"}
```

### 4. Observer Pattern

The `hook_tool_call` callback mechanism allows external observation:

```python
async def my_hook(agent, function):
    print(f"Tool called: {function.name}")
    return await original_tool_call(function)

agent = ReActAgent(hook_tool_call=my_hook)
```

### 5. Template Method Pattern

The `Node.run()` method defines the algorithm skeleton:

```python
class Node(ABC):
    @abstractmethod
    async def run(self, input_chunks):
        # Subclasses implement specific processing logic
        pass

class OpenAI(Node):
    async def run(self, input_chunks):
        # OpenAI-specific implementation
        pass
```

### 6. Chain of Responsibility Pattern

Nodes can be chained together forming a processing pipeline:

```python
# Create nodes
node1 = OpenAI(...)
node2 = ReActAgent(...)

# Chain them: node1 > node2 means node1 feeds into node2
pipeline = node1 > node2

# The __gt__ method establishes dependencies
def __gt__(self, node: "Node") -> "Node":
    if self not in node.deps:
        node.deps.append(self)
    return self
```

### 7. DAG (Directed Acyclic Graph) Pattern

The node dependency system creates a DAG structure:

- Each node maintains a list of dependencies (`deps`)
- Processing flows from leaf nodes to root nodes
- Cycles are prevented by the acyclic nature
- Enables complex workflows with branching and merging

## Component Relationships

### Inheritance Hierarchy

```
Node (base.py)
└── OpenAI (llm.py)
    └── ChatLLM (chat.py)
        └── ReActAgent (re_act.py)
            └── GroupAgent (agent_group.py)
```

### Composition Relationships

- `ChatLLM` → `MessagesManager`: For message persistence
- `ChatLLM` → `ToolsManager`: For tool execution  
- `ReActAgent` → `ChatLLM`: Inherits chat capabilities
- `ToolsManager` → `Tool`: Manages registered tools

### Data Flow

1. **Input**: User provides input as dictionary or stream of `InputChunk`
2. **Processing**: 
   - System prompt is formatted with user variables
   - Conversation history is retrieved (if enabled)
   - Messages are sent to LLM API
   - Response is parsed for tool calls or final answers
   - Tool calls are executed and results incorporated
3. **Output**: Stream of `InputChunk` objects with final response

### Dependency Injection

Components are loosely coupled through constructor injection:

```python
agent = ReActAgent(
    model="gpt-4",
    system_prompt="You are helpful",
    api_key="secret",
    messages_manager=qdrant_manager,  # Injected dependency
    tools_manager=tools_manager,      # Injected dependency
    mcp="http://mcp-endpoint"         # Injected dependency
)
```

## Extension Points

### Adding New Storage Backends

Implement the `MessagesManager` interface:

```python
class PostgreSQLMessagesManager(MessagesManager):
    async def get_messages(self, session_id: str, user_input: str = ""):
        # PostgreSQL implementation
        
    async def save_messages(self, session_id: str, messages):
        # PostgreSQL implementation
```

### Adding New LLM Providers

Extend the `Node` class:

```python
class Anthropic(Node):
    async def run(self, input_chunks):
        # Anthropic API implementation
```

### Custom Tool Processing

Use the `hook_tool_call` parameter:

```python
async def custom_tool_handler(agent, function):
    # Add logging, validation, or transformation
    result = await agent.call_tool(function)
    # Post-process result
    return result
```

## Best Practices

### 1. Session Management
- Always provide consistent `session_id` for maintaining conversation context
- Consider session cleanup for long-running applications

### 2. Error Handling
- Implement robust error handling in custom tools
- Use `json-repair` for resilient JSON parsing (already integrated)

### 3. Performance Optimization
- Limit conversation history length to prevent token overflow
- Use appropriate `max_loop` values to prevent infinite reasoning cycles

### 4. Security
- Never expose API keys in client-side code
- Validate tool inputs to prevent injection attacks

### 5. Testing
- Use the provided test suite as reference for implementation patterns
- Test both streaming and non-streaming modes

## Example Architecture Usage

### Simple Pipeline
```python
# Basic LLM interaction
llm = OpenAI(model="gpt-4", system_prompt="Helpful assistant", api_key="key")
```

### Enhanced Chat Agent
```python
# Chat with memory and tools
messages_manager = QdrantMessagesManager()
tools_manager = ToolsManager()

@tools_manager.tool(name="calculator", ...)
def calculator(expr): return eval(expr)

agent = ChatLLM(
    model="gpt-4",
    system_prompt="Math helper",
    api_key="key",
    messages_manager=messages_manager,
    tools_manager=tools_manager
)
```

### Full ReAct Agent
```python
# Complete reasoning and action agent
react_agent = ReActAgent(
    model="gpt-4",
    system_prompt="Expert assistant",
    api_key="key",
    messages_manager=messages_manager,
    tools_manager=tools_manager,
    with_time=True,
    max_loop=5,
    stream=True
)
```

This architecture enables building sophisticated AI agents while maintaining code clarity, testability, and extensibility.
