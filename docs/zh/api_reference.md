# Nuwa API 参考

本文档提供了 Nuwa 框架中所有类、方法和参数的详细 API 参考。

## 目录

- [基础类](#基础类)
  - [Node](#node)
  - [MessagesManager](#messagesmanager)
  - [InputChunk](#inputchunk)
- [LLM 模块](#llm-模块)
  - [OpenAI](#openai)
- [聊天模块](#聊天模块)
  - [ChatLLM](#chatllm)
- [ReAct 模块](#react-模块)
  - [ReActAgent](#reactagent)
- [工具模块](#工具模块)
  - [ToolsManager](#toolsmanager)
  - [ToolEntity](#toolentity)
  - [ToolParameter](#toolparameter)
  - [ToolObjectParameter](#toolobjectparameter)
  - [ToolArrayParameter](#toolarrayparameter)
- [工具函数](#工具函数)

## 基础类

### Node

Nuwa 流水线中所有处理节点的抽象基类。

**方法:**
- `__gt__(self, node: "Node") -> "Node"`: 在节点之间创建依赖关系
- `get_dep(self) -> "Node"`: 返回根依赖节点
- `run(self, input_chunks: Union[AsyncGenerator[InputChunk, Any], Dict]) -> AsyncGenerator[InputChunk, Any]`: 抽象方法，必须由子类实现

### MessagesManager

消息持久化和检索的抽象基类。

**抽象方法:**
- `get_messages(self, session_id: str, user_input: str = "") -> List[ChatCompletionMessageParam]`: 检索会话的对话历史
- `save_messages(self, session_id: str, messages: List[ChatCompletionMessageParam])`: 将消息保存到持久化存储

### InputChunk

表示流水线中数据块的 Pydantic 模型。

**字段:**
- `state: Literal["DOING", "DONE", "END"]`: 数据块的当前状态
- `content: Union[str, Dict]`: 数据块的内容

## LLM 模块

### OpenAI

支持 OpenAI 兼容 API 的基础 LLM 接口。

**构造函数:**
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

**参数:**
- `model`: LLM 模型标识符（例如 "gpt-4", "gpt-3.5-turbo"）
- `system_prompt`: 系统提示模板（支持使用 `{key}` 占位符进行格式化）
- `api_key`: 用于身份验证的 API 密钥
- `temperature`: 采样温度（0.0 到 1.0）
- `extra_body`: 要传递给 API 的额外参数
- `stop`: 生成的停止序列
- `with_time`: 在系统提示中包含当前时间戳
- `with_others`: 要添加到系统提示前面的额外上下文字符串
- `base_url`: API 端点的基础 URL
- `stream`: 启用流式响应

**方法:**
- `parse_input(self, input_chunks: Union[AsyncGenerator[InputChunk, None], Dict[str, Any]]) -> Dict[str, Any]`: 将输入数据块解析为字典格式
- `generate_messages(self, input_dict: Dict[str, Any]) -> List[ChatCompletionMessageParam]`: 从输入字典生成聊天完成消息
- `run(self, input_chunks: Union[AsyncGenerator[InputChunk, None], Dict[str, Any]]) -> AsyncGenerator[InputChunk, None]`: 执行 LLM 并生成响应数据块

## 聊天模块

### ChatLLM

扩展 OpenAI 的增强聊天功能，具有消息历史管理和工具支持。

**构造函数:**
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

**额外参数:**
- `messages_manager`: 消息持久化管理器实例
- `tools_manager`: 工具注册和执行器
- `mcp`: MCP（模型上下文协议）端点 URL
- `mcp_timeout`: MCP 请求超时时间（秒）
- `session_id`: 唯一会话标识符（如未提供则自动生成）
- `hook_tool_call`: 在执行工具调用之前调用的回调函数

**方法:**
- `call_tool(self, func: Function)`: 执行工具调用
- `save_messages(self)`: 将对话历史保存到消息管理器
- 继承自 OpenAI 的所有方法

## ReAct 模块

### ReActAgent

实现 ReAct（推理 + 行动）模式的高级 Agent，具有结构化输出解析。

**构造函数:**
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

**额外参数:**
- `max_loop`: 允许的最大推理-行动循环次数
- `enable_chat_history`: 启用对话历史持久化
- `enable_selection`: 启用用户选择请求（添加 `request_user_choice` 工具）

**特殊标签:**
ReActAgent 使用特定的 XML 样式标签来构建其输出：
- `<T>`: 思考/推理内容
- `<A>`: 行动/工具调用（JSON 格式）
- `<O>`: 观察/工具结果
- `<Q>`: 向用户提问（当 `enable_selection=True` 时）

**方法:**
- `parse_action(self, json_str: str) -> List[Function]`: 将行动 JSON 解析为函数调用
- `parse_labels(self, chunk: InputChunk) -> List[Function]`: 解析流式输出中的 ReAct 标签
- `parse_system_prompt(self, instruction: str) -> str`: 使用工具描述生成系统提示
- 继承自 ChatLLM 的所有方法

## 工具模块

### ToolsManager

工具注册和执行管理器。

**构造函数:**
```python
ToolsManager(init_tools: Dict[str, Tool] = {})
```

**方法:**
- `tool(self, name: Optional[str] = None, parameters: Union[ToolObjectParameter, ToolParameter] = ToolObjectParameter(type="object", properties={}, required=[]), description: Optional[str] = None)`: 用于将函数注册为工具的装饰器
- `add_tool(self, tool: Tool)`: 将工具实例添加到管理器
- `get_tool(self, tool_name: str) -> Optional[Tool]`: 按名称检索工具
- `call_tool(self, func: Function) -> Any`: 执行工具调用
- `list_tools(self) -> List[str]`: 获取已注册工具名称列表
- `has_tool(self, tool_name: str) -> bool`: 检查工具是否已注册
- `clear_tools(self)`: 移除所有已注册的工具

### ToolEntity

表示工具定义的 Pydantic 模型。

**字段:**
- `name: str`: 工具名称
- `parameters: Union[ToolObjectParameter, ToolParameter]`: 工具参数模式
- `description: Optional[str]`: 工具描述

### ToolParameter

工具的基础参数类型。

**字段:**
- `type: Literal["object", "string", "number", "boolean", "array"]`: 参数类型
- `description: Optional[str]`: 参数描述
- `enum: Optional[List[str]]`: 允许的值（用于字符串参数）

### ToolObjectParameter

对象参数类型（继承自 ToolParameter）。

**额外字段:**
- `properties: Dict[str, Union["ToolObjectParameter", ToolArrayParameter, ToolParameter]]`: 对象属性
- `required: List[str]`: 必需的属性名称

### ToolArrayParameter

数组参数类型（继承自 ToolParameter）。

**额外字段:**
- `items: Union["ToolObjectParameter", "ToolArrayParameter", ToolParameter]`: 数组项类型

## 工具函数

### get_tool_entity

将 MCP 工具定义转换为 Nuwa ToolEntity。

**签名:**
```python
get_tool_entity(tool: MCPTool) -> ToolEntity
```

**参数:**
- `tool`: MCP 工具定义

**返回:**
- `ToolEntity`: 转换后的 Nuwa 工具实体

---

## 类型定义

### Function

OpenAI 函数调用类型的别名：
```python
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
```

### ChatCompletionMessageParam

OpenAI 聊天消息类型：
- `ChatCompletionSystemMessageParam`
- `ChatCompletionUserMessageParam`
- `ChatCompletionAssistantMessageParam`
- `ChatCompletionToolMessageParam`

## 错误处理

需要处理的常见异常：

- `RateLimitError`: 当超过 API 速率限制时
- `ValueError`: 当提供无效输入或工具名称时
- `json.JSONDecodeError`: 当 JSON 解析失败时（尽管内部使用了 `json-repair`）

## 异步/等待模式

Nuwa 中的所有主要方法都是异步的。使用正确的 async/await 模式：

```python
# 正确用法
async def main():
    agent = ReActAgent(...)
    async for chunk in agent.run(input_data):
        process_chunk(chunk)

# 使用 asyncio 运行
import asyncio
asyncio.run(main())
```

对于同步上下文，您可以使用：
```python
import asyncio

def sync_wrapper():
    async def async_func():
        # 您的异步代码在这里
        pass
    
    return asyncio.run(async_func())
```

## 性能考虑

- **内存管理**: 框架自动管理消息历史，但您可以通过 `enable_chat_history=False` 禁用它
- **流式传输**: 启用流式传输（`stream=True`）以获得更好的长响应用户体验
- **工具缓存**: 工具在 ToolsManager 中被缓存，因此初始设置后的注册开销很小
- **连接复用**: OpenAI 客户端复用连接以获得更好的性能

本 API 参考涵盖了 Nuwa 框架中的所有公共接口。有关内部实现细节，请参考源代码。
