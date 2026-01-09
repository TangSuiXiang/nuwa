# Nuwa - AI Agent 框架文档

Nuwa 是一个功能强大的框架，用于创建具有高级功能的 AI Agent，包括 ReAct（推理与行动）模式、工具集成和对话管理。本文档提供了使用 Nuwa 库的全面指导。

## 目录

- [安装](#安装)
- [核心概念](#核心概念)
- [基本用法](#基本用法)
  - [简单的 LLM 交互](#简单的-llm-交互)
  - [带记忆的聊天](#带记忆的聊天)
  - [ReAct Agent](#react-agent)
- [高级特性](#高级特性)
  - [工具集成](#工具集成)
  - [MCP（模型上下文协议）支持](#mcp-模型上下文协议支持)
  - [自定义系统提示](#自定义系统提示)
  - [流式响应](#流式响应)
  - [时间感知](#时间感知)
- [配置选项](#配置选项)
- [示例](#示例)

## 安装

使用 pip 安装 Nuwa：

```bash
pip install nuwa
```

或者从源码安装：

```bash
git clone https://github.com/your-repo/nuwa.git
cd nuwa
pip install .
```

## 核心概念

### 节点架构 (Node Architecture)
Nuwa 使用基于节点的架构，每个组件都继承自 `Node` 基类。这允许灵活地组合和链接不同的 AI 功能。

### 关键组件

- **OpenAI**: 基础 LLM 接口，支持 OpenAI 兼容 API
- **ChatLLM**: 增强的聊天功能，具有消息历史管理
- **ReActAgent**: 实现 ReAct（推理 + 行动）模式的高级 Agent
- **ToolsManager**: 工具注册和执行系统
- **MessagesManager**: 消息持久化和检索系统

## 基本用法

### 简单的 LLM 交互

用于没有聊天历史或工具的基本 LLM 交互：

```python
from src.nuwa.llm import OpenAI
import asyncio

async def simple_llm_example():
    llm = OpenAI(
        model="gpt-4",
        system_prompt="您是一个乐于助人的助手。",
        api_key="your-api-key"
    )
    
    async for chunk in llm.run({"user": "您好，您好吗？"}):
        print(chunk.content, end="", flush=True)

# 运行示例
asyncio.run(simple_llm_example())
```

### 带记忆的聊天

用于具有持久化消息历史的对话式 AI：

```python
from src.nuwa.chat import ChatLLM
from src.nuwa.base import MessagesManager
import asyncio

# 自定义 MessagesManager 实现（或使用 QdrantMessagesManager）
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
        system_prompt="您是一个友好的聊天机器人。",
        api_key="your-api-key",
        messages_manager=messages_manager,
        session_id="user123"
    )
    
    # 第一条消息
    async for chunk in chat_llm.run({"user": "您叫什么名字？"}):
        print(chunk.content, end="", flush=True)
    
    print("\n")
    
    # 后续消息（带有之前对话的上下文）
    async for chunk in chat_llm.run({"user": "很高兴认识您！"}):
        print(chunk.content, end="", flush=True)

asyncio.run(chat_example())
```

### ReAct Agent

ReAct Agent 将推理和工具使用结合在结构化格式中：

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def react_agent_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个能够推理和行动的乐于助人的助手。",
        api_key="your-api-key",
        max_loop=3  # 最大推理-行动循环次数
    )
    
    async for chunk in agent.run({"user": "今天天气怎么样？"}):
        if chunk.state == "END":
            print(f"\n最终答案: {chunk.content}")
        else:
            print(chunk.content, end="", flush=True)

asyncio.run(react_agent_example())
```

## 高级特性

### 工具集成

Nuwa 提供了强大的工具系统来扩展 Agent 功能：

```python
from src.nuwa.re_act import ReActAgent
from src.nuwa.tool import ToolsManager, ToolObjectParameter, ToolParameter
import asyncio

# 创建工具管理器
tools = ToolsManager()

# 注册天气工具
@tools.tool(
    name="get_weather",
    description="获取特定城市的天气。",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "city": ToolParameter(type="string", description="城市名称")
        },
    ),
)
async def get_weather(city: str):
    # 模拟 API 调用
    return {"temperature": "22°C", "condition": "晴朗"}

async def tool_integration_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个天气助手。",
        api_key="your-api-key",
        tools_manager=tools,
        with_time=True  # 在系统提示中包含当前时间
    )
    
    async for chunk in agent.run({"user": "北京的天气怎么样？"}):
        print(chunk.content, end="", flush=True)

asyncio.run(tool_integration_example())
```

### MCP（模型上下文协议）支持

Nuwa 支持 MCP 以集成外部工具和服务：

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def mcp_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个创意助手。",
        api_key="your-api-key",
        mcp="http://localhost:12345/mcp",  # MCP 端点
        mcp_timeout=300
    )
    
    async for chunk in agent.run({"user": "生成一张猫的图片"}):
        print(chunk.content, end="", flush=True)

asyncio.run(mcp_example())
```

### 自定义系统提示

系统提示可以使用用户提供的变量进行动态格式化：

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def custom_prompt_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个{role}，性格：{personality}",
        api_key="your-api-key"
    )
    
    # 提供系统变量
    input_data = {
        "user": "介绍一下您自己",
        "system": {
            "role": "猫娘女仆",
            "personality": "害羞但很关心人"
        }
    }
    
    async for chunk in agent.run(input_data):
        print(chunk.content, end="", flush=True)

asyncio.run(custom_prompt_example())
```

### 流式响应

启用流式传输以实现实时响应生成：

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def streaming_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个乐于助人的助手。",
        api_key="your-api-key",
        stream=True,
        extra_body={"enable_thinking": True}  # 如果支持，启用思考模式
    )
    
    async for chunk in agent.run({"user": "解释量子计算"}):
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(streaming_example())
```

### 时间感知

在系统提示中包含当前时间戳以处理时间敏感的查询：

```python
from src.nuwa.re_act import ReActAgent
import asyncio

async def time_aware_example():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个时间感知的助手。",
        api_key="your-api-key",
        with_time=True  # 自动在系统提示中添加当前时间
    )
    
    async for chunk in agent.run({"user": "现在几点了？"}):
        print(chunk.content, end="", flush=True)

asyncio.run(time_aware_example())
```

## 配置选项

### OpenAI/ChatLLM/ReActAgent 通用参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `model` | str | - | LLM 模型名称 |
| `system_prompt` | str | - | 系统提示模板 |
| `api_key` | str | - | LLM 服务的 API 密钥 |
| `base_url` | str | "https://api.openai.com/v1" | API 的基础 URL |
| `temperature` | float | 0.6 | 采样温度 |
| `stream` | bool | False | 启用流式响应 |
| `with_time` | bool | False | 在系统提示中包含当前时间 |
| `with_others` | str | "" | 要包含的额外上下文 |
| `stop` | List[str] | None | 停止序列 |
| `extra_body` | Dict | None | API 的额外参数 |

### ReActAgent 特定参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `session_id` | str | 自动生成 | 会话标识符 |
| `messages_manager` | MessagesManager | None | 消息持久化管理器 |
| `tools_manager` | ToolsManager | None | 工具注册和执行器 |
| `mcp` | str | None | MCP 端点 URL |
| `mcp_timeout` | int | 300 | MCP 请求超时 |
| `max_loop` | int | 3 | 最大 ReAct 推理循环次数 |
| `enable_chat_history` | bool | True | 启用对话历史 |
| `enable_selection` | bool | False | 启用用户选择请求 |
| `hook_tool_call` | Callable | None | 工具调用的回调函数 |

### ToolsManager

`ToolsManager` 提供基于装饰器的工具注册：

```python
tools = ToolsManager()

@tools.tool(
    name="calculator",
    description="执行数学计算",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "expression": ToolParameter(type="string", description="数学表达式")
        }
    )
)
def calculator(expression: str):
    return eval(expression)
```

## 示例

### 带多个工具的完整 ReAct Agent

```python
from src.nuwa.re_act import ReActAgent
from src.nuwa.tool import ToolsManager, ToolObjectParameter, ToolParameter
import asyncio
import json

# 初始化工具管理器
tools = ToolsManager()

# 天气工具
@tools.tool(
    name="get_weather",
    description="获取城市的当前天气",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "city": ToolParameter(type="string", description="城市名称")
        }
    )
)
async def get_weather(city: str):
    return {"city": city, "temperature": "25°C", "condition": "晴朗"}

# 计算器工具
@tools.tool(
    name="calculate",
    description="执行数学计算",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "expression": ToolParameter(type="string", description="数学表达式")
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
        system_prompt="您是一个具有天气和计算工具的乐于助人的助手。",
        api_key="your-api-key",
        tools_manager=tools,
        with_time=True,
        stream=True,
        enable_chat_history=True,
        max_loop=5
    )
    
    # 多轮对话
    questions = [
        "东京的天气怎么样？",
        "计算 15 * 24",
        "根据天气和我的计算，我应该出门吗？"
    ]
    
    for question in questions:
        print(f"\n用户: {question}")
        print("Agent: ", end="")
        async for chunk in agent.run({"user": question}):
            if chunk.state == "END":
                print(f"\n最终: {chunk.content}")
            else:
                print(chunk.content, end="", flush=True)

asyncio.run(complete_example())
```

### 使用 Qdrant 进行消息存储

Nuwa 内置支持 Qdrant 向量数据库进行消息存储：

```python
from src.nuwa.re_act import ReActAgent
from src.nuwa.qdrant import QdrantMessagesManager
import asyncio

async def qdrant_example():
    # 初始化 Qdrant 消息管理器
    messages_manager = QdrantMessagesManager(
        host="localhost",
        port=6333,
        collection_name="chat_history"
    )
    
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个具有记忆增强的助手。",
        api_key="your-api-key",
        messages_manager=messages_manager,
        session_id="user_session_123"
    )
    
    async for chunk in agent.run({"user": "记住我喜欢巧克力冰淇淋"}):
        print(chunk.content, end="", flush=True)
    
    print("\n")
    
    async for chunk in agent.run({"user": "我最喜欢的冰淇淋是什么？"}):
        print(chunk.content, end="", flush=True)

# asyncio.run(qdrant_example())
```

## 最佳实践

1. **会话管理**: 始终为维护对话上下文提供一致的 `session_id`
2. **错误处理**: 为工具调用和 API 请求实现适当的错误处理
3. **速率限制**: 在发出多个请求时注意 API 速率限制
4. **安全性**: 切勿在客户端代码或公共仓库中暴露 API 密钥
5. **测试**: 使用提供的测试套件作为实现模式的参考

## 故障排除

### 常见问题

- **工具未找到**: 确保工具名称在注册和使用之间完全匹配
- **JSON 解析错误**: 使用 `json-repair` 库进行健壮的 JSON 解析（已包含）
- **流式问题**: 设置 `stream=True` 并适当地处理数据块
- **内存泄漏**: 定期清除历史消息或在不需要时禁用聊天历史

### 调试

启用日志记录以查看有关 Agent 操作的详细信息：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

这将显示工具调用、消息流和内部 Agent 状态。

---

有关更多示例和高级使用模式，请参阅仓库 `tests/` 目录中的测试文件。
