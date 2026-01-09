# Nuwa API Reference

This document provides detailed API reference for all classes, methods, and parameters in the Nuwa framework.

## Table of Contents

- [Base Classes](#base-classes)
  - [Node](#node)
  - [MessagesManager](#messagesmanager)
  - [InputChunk](#inputchunk)
- [LLM Module](#llm-module)
  - [OpenAI](#openai)
- [Chat Module](#chat-module)
  - [ChatLLM](#chatllm)
- [ReAct Module](#react-module)
  - [ReActAgent](#reactagent)
- [Tool Module](#tool-module)
  - [ToolsManager](#toolsmanager)
  - [ToolEntity](#toolentity)
  - [ToolParameter](#toolparameter)
  - [ToolObjectParameter](#toolobjectparameter)
  - [ToolArrayParameter](#toolarrayparameter)
- [Utility Functions](#utility-functions)

## Base Classes

### Node

Abstract base class for all processing nodes in the Nuwa pipeline.

**Methods:**
- `__gt__(self, node: "Node") -> "Node"`: Creates a dependency relationship between nodes
- `get_dep(self) -> "Node"`: Returns the root dependency node
- `run(self, input_chunks: Union[AsyncGenerator[InputChunk, Any], Dict]) -> AsyncGenerator[InputChunk, Any]`: Abstract method that must be implemented by subclasses

### MessagesManager

Abstract base class for message persistence and retrieval.

**Abstract Methods:**
- `get_messages(self, session_id: str, user_input: str = "") -> List[ChatCompletionMessageParam]`: Retrieve conversation history for a session
- `save_messages(self, session_id: str, messages: List[ChatCompletionMessageParam])`: Save messages to persistent storage

### InputChunk

Pydantic model representing a chunk of data in the processing pipeline.

**Fields:**
- `state: Literal["DOING", "DONE", "END"]`: Current state of the chunk
- `content: Union[str, Dict]`: Content of the chunk

## LLM Module

### OpenAI

Basic LLM interface supporting OpenAI-compatible APIs.

**Constructor:**
```python
OpenAI(
    model: str,
    system_prompt: str,
    api_key: str,
    temperature: float = 0.6,
    extra_body: Optional[Dict[str, Any]] = None,
    stop: Optional[List[str]] = None,
    with_time: bool = False,
    with_others: str = "",
    base_url: str = "https://api.openai.com/v1",
    stream: bool = False
)
```

**Parameters:**
- `model`: LLM model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
- `system_prompt`: System prompt template (supports formatting with `{key}` placeholders)
- `api_key`: API key for authentication
- `temperature`: Sampling temperature (0.0 to 1.0)
- `extra_body`: Additional parameters to pass to the API
- `stop`: Stop sequences for generation
- `with_time`: Include current timestamp in system prompt
- `with_others`: Additional context string to prepend to system prompt
- `base_url`: Base URL for the API endpoint
- `stream`: Enable streaming responses

**Methods:**
- `parse_input(self, input_chunks: Union[AsyncGenerator[InputChunk, None], Dict[str, Any]]) -> Dict[str, Any]`: Parse input chunks into dictionary format
- `generate_messages(self, input_dict: Dict[str, Any]) -> List[ChatCompletionMessageParam]`: Generate chat completion messages from input dictionary
- `run(self, input_chunks: Union[AsyncGenerator[InputChunk, None], Dict[str, Any]]) -> AsyncGenerator[InputChunk, None]`: Execute the LLM and yield response chunks

## Chat Module

### ChatLLM

Enhanced chat functionality extending OpenAI with message history management and tool support.

**Constructor:**
```python
ChatLLM(
    model: str,
    system_prompt: str,
    api_key: str,
    messages_manager: Optional[MessagesManager] = None,
    tools_manager: ToolsManager = None,
    mcp: Optional[str] = None,
    mcp_timeout: int = 300,
    session_id: Optional[str] = None,
    stream: bool = False,
    temperature: float = 0.6,
    extra_body: Optional[Dict[str, Any]] = None,
    with_time: bool = False,
    with_others: str = "",
    stop: Optional[list] = None,
    base_url: str = "https://api.openai.com/v1",
    hook_tool_call: Optional[Callable[["ChatLLM", Function], Awaitable[Any]]] = None
)
```

**Additional Parameters:**
- `messages_manager`: Message persistence manager instance
- `tools_manager`: Tool registry and executor
- `mcp`: MCP (Model Context Protocol) endpoint URL
- `mcp_timeout`: MCP request timeout in seconds
- `session_id`: Unique session identifier (auto-generated if not provided)
- `hook_tool_call`: Callback function called before executing tool calls

**Methods:**
- `call_tool(self, func: Function)`: Execute a tool call
- `save_messages(self)`: Save conversation history to the messages manager
- All inherited methods from OpenAI

## ReAct Module

### ReActAgent

Advanced agent implementing the ReAct (Reasoning + Acting) pattern with structured output parsing.

**Constructor:**
```python
ReActAgent(
    model: str,
    system_prompt: str,
    api_key: str,
    session_id: str = str(uuid4()),
    stream: bool = False,
    messages_manager: Optional[MessagesManager] = None,
    tools_manager: ToolsManager = None,
    mcp: Optional[str] = None,
    mcp_timeout: int = 300,
    temperature: float = 0.6,
    extra_body: Dict[str, Any] = None,
    max_loop: int = 3,
    with_time: bool = False,
    with_others: str = "",
    stop: List[str] = None,
    base_url: str = "https://api.openai.com/v1",
    enable_chat_history: bool = True,
    enable_selection: bool = False,
    hook_tool_call: Optional[Callable[[ChatLLM, Function], Awaitable[Any]]] = None
)
```

**Additional Parameters:**
- `max_loop`: Maximum number of reasoning-action loops allowed
- `enable_chat_history`: Enable conversation history persistence
- `enable_selection`: Enable user choice requests (adds `request_user_choice` tool)

**Special Tags:**
The ReActAgent uses specific XML-style tags to structure its output:
- `<T>`: Thought/Reasoning content
- `<A>`: Action/Tool call in JSON format
- `<O>`: Observation/Tool result
- `<Q>`: Question to user (when `enable_selection=True`)

**Methods:**
- `parse_action(self, json_str: str) -> List[Function]`: Parse action JSON into function calls
- `parse_labels(self, chunk: InputChunk) -> List[Function]`: Parse streaming output for ReAct tags
- `parse_system_prompt(self, instruction: str) -> str`: Generate system prompt with tool descriptions
- All inherited methods from ChatLLM

## Tool Module

### ToolsManager

Tool registration and execution manager.

**Constructor:**
```python
ToolsManager(init_tools: Dict[str, Tool] = {})
```

**Methods:**
- `tool(self, name: Optional[str] = None, parameters: Union[ToolObjectParameter, ToolParameter] = ToolObjectParameter(type="object", properties={}, required=[]), description: Optional[str] = None)`: Decorator for registering functions as tools
- `add_tool(self, tool: Tool)`: Add a tool instance to the manager
- `get_tool(self, tool_name: str) -> Optional[Tool]`: Retrieve a tool by name
- `call_tool(self, func: Function) -> Any`: Execute a tool call
- `list_tools(self) -> List[str]`: Get list of registered tool names
- `has_tool(self, tool_name: str) -> bool`: Check if a tool is registered
- `clear_tools(self)`: Remove all registered tools

### ToolEntity

Pydantic model representing a tool definition.

**Fields:**
- `name: str`: Tool name
- `parameters: Union[ToolObjectParameter, ToolParameter]`: Tool parameters schema
- `description: Optional[str]`: Tool description

### ToolParameter

Base parameter type for tools.

**Fields:**
- `type: Literal["object", "string", "number", "boolean", "array"]`: Parameter type
- `description: Optional[str]`: Parameter description
- `enum: Optional[List[str]]`: Allowed values (for string parameters)

### ToolObjectParameter

Object parameter type (extends ToolParameter).

**Additional Fields:**
- `properties: Dict[str, Union["ToolObjectParameter", ToolArrayParameter, ToolParameter]]`: Object properties
- `required: List[str]`: Required property names

### ToolArrayParameter

Array parameter type (extends ToolParameter).

**Additional Fields:**
- `items: Union["ToolObjectParameter", "ToolArrayParameter", ToolParameter]`: Array item type

## Utility Functions

### get_tool_entity

Convert MCP tool definition to Nuwa ToolEntity.

**Signature:**
```python
get_tool_entity(tool: MCPTool) -> ToolEntity
```

**Parameters:**
- `tool`: MCP tool definition

**Returns:**
- `ToolEntity`: Converted Nuwa tool entity

---

## Type Definitions

### Function

Alias for OpenAI's function call type:
```python
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
```

### ChatCompletionMessageParam

OpenAI chat message types:
- `ChatCompletionSystemMessageParam`
- `ChatCompletionUserMessageParam`
- `ChatCompletionAssistantMessageParam`
- `ChatCompletionToolMessageParam`

## Error Handling

Common exceptions to handle:

- `RateLimitError`: When API rate limits are exceeded
- `ValueError`: When invalid inputs or tool names are provided
- `json.JSONDecodeError`: When JSON parsing fails (though `json-repair` is used internally)

## Async/Await Patterns

All main methods in Nuwa are asynchronous. Use proper async/await patterns:

```python
# Correct usage
async def main():
    agent = ReActAgent(...)
    async for chunk in agent.run(input_data):
        process_chunk(chunk)

# Run with asyncio
import asyncio
asyncio.run(main())
```

For synchronous contexts, you can use:
```python
import asyncio

def sync_wrapper():
    async def async_func():
        # Your async code here
        pass
    
    return asyncio.run(async_func())
```

## Performance Considerations

- **Memory Management**: The framework automatically manages message history, but you can disable it with `enable_chat_history=False`
- **Streaming**: Enable streaming (`stream=True`) for better user experience with long responses
- **Tool Caching**: Tools are cached in the ToolsManager, so registration overhead is minimal after initial setup
- **Connection Reuse**: The OpenAI client reuses connections for better performance

This API reference covers all public interfaces in the Nuwa framework. For internal implementation details, refer to the source code.
