# Nuwa 架构和设计模式

Nuwa 基于一个复杂的架构构建，该架构结合了现代设计模式与实用的 AI Agent 开发原则。本文档全面概述了系统的架构、设计模式和组件关系。

## 整体架构

Nuwa 遵循**分层架构**，并结合了**有向无环图 (DAG) 处理流水线**。核心理念是 Nuwa 中的所有内容都是一个 `Node`（节点），可以链接在一起创建复杂的 AI Agent 工作流。

### 关键架构原则

1. **可组合性**: 组件可以轻松组合和扩展
2. **关注点分离**: 每个层都有明确的职责
3. **可扩展性**: 可以轻松添加新组件而无需修改现有代码
4. **流式支持**: 原生支持实时响应生成
5. **可插拔存储**: 消息持久化可以使用不同的后端

## 核心层次

### 1. 基础层 (`src/nuwa/base.py`)

Nuwa 架构的基础，定义了抽象接口：

- **`Node`**: 所有处理组件的抽象基类
  - 通过 `deps` 列表实现依赖管理
  - 支持使用 `>` 操作符进行链接 (`__gt__` 方法)
  - 定义抽象的 `run()` 方法用于处理逻辑
  
- **`MessagesManager`**: 消息持久化的抽象接口
  - `get_messages()`: 检索对话历史
  - `save_messages()`: 将消息持久化到存储中
  
- **`InputChunk`**: 用于流式输入/输出的数据结构
  - 包含 `state` 字段 (`DOING`, `DONE`, `END`)
  - 包含 `content` 字段用于实际数据

### 2. LLM 层 (`src/nuwa/llm.py`)

提供基本的大语言模型交互能力：

- **`OpenAI`**: `Node` 的具体实现
  - 处理与 OpenAI 兼容端点的 API 通信
  - 支持流式和非流式模式
  - 管理带有动态模板的系统提示
  - 包括时间感知和自定义上下文注入

### 3. 聊天层 (`src/nuwa/chat.py`)

通过对话管理扩展 LLM 功能：

- **`ChatLLM`**: 继承自 `OpenAI`
  - 与 `MessagesManager` 集成进行对话历史管理
  - 管理会话 ID 以支持多用户
  - 通过 `ToolsManager` 处理工具调用
  - 支持 MCP (Model Context Protocol) 集成
  - 提供钩子机制用于自定义工具调用处理

### 4. ReAct 层 (`src/nuwa/re_act.py`)

实现 ReAct (推理 + 行动) 模式：

- **`ReActAgent`**: 继承自 `ChatLLM`
  - 解析结构化输出格式: `<Q>问题</Q><T>思考</T><A>行动</A><O>观察</O>`
  - 支持多个推理-行动循环 (`max_loop` 参数)
  - 内置 `answer` 工具用于最终响应
  - 可选的 `request_user_choice` 工具用于用户交互
  - 使用 `json-repair` 进行复杂的 JSON 解析和错误恢复

### 5. 工具层 (`src/nuwa/tool.py`)

管理外部函数集成：

- **`ToolsManager`**: 工具的中央注册表
  - 基于装饰器的工具注册 (`@tools.tool()`)
  - 从函数注解自动推断参数
  - 支持同步和异步函数
  - 处理 MCP 工具发现和集成
  
- **工具实体**: 结构化的工具定义
  - `ToolEntity`: 主要工具定义，包含名称、参数、描述
  - `ToolParameter`: 参数规范，包含类型和描述
  - `ToolObjectParameter`: 复杂对象参数
  - `ToolArrayParameter`: 数组参数

### 6. 存储层 (`src/nuwa/qdrant.py`)

提供具体的消息持久化实现：

- **`QdrantMessagesManager`**: 实现 `MessagesManager`
  - 使用 Qdrant 向量数据库进行消息存储
  - 支持高效的检索和持久化
  - 可以替换为其他存储后端

### 7. Agent 组层 (`src/nuwa/agent_group.py`)

用于多 Agent 场景的专用 Agent：

- **`GroupAgent`**: 继承自 `ReActAgent`
  - 支持基于角色的 Agent 专业化
  - 包括角色黑名单以防止自我引用
  - 维护组上下文感知

## 设计模式

### 1. 抽象工厂模式

`MessagesManager` 抽象基类允许可插拔的存储实现：

```python
# 抽象接口
class MessagesManager(ABC):
    @abstractmethod
    async def get_messages(self, session_id: str, user_input: str = ""):
        pass
    
    @abstractmethod
    async def save_messages(self, session_id: str, messages):
        pass

# 具体实现
class QdrantMessagesManager(MessagesManager):
    # Qdrant 特定实现
    
class SimpleMessagesManager(MessagesManager):
    # 内存实现
```

### 2. 策略模式

不同的 LLM 提供商可以作为策略实现：

```python
class OpenAI(Node):
    # OpenAI 兼容实现
    
class Anthropic(Node):
    # 可以为 Anthropic 支持添加
```

### 3. 装饰器模式

工具注册使用装饰器模式以获得简洁的语法：

```python
tools = ToolsManager()

@tools.tool(
    name="get_weather",
    description="获取城市天气",
    parameters=ToolObjectParameter(...)
)
def get_weather(city: str):
    return {"weather": "晴朗"}
```

### 4. 观察者模式

`hook_tool_call` 回调机制允许外部观察：

```python
async def my_hook(agent, function):
    print(f"工具被调用: {function.name}")
    return await original_tool_call(function)

agent = ReActAgent(hook_tool_call=my_hook)
```

### 5. 模板方法模式

`Node.run()` 方法定义了算法骨架：

```python
class Node(ABC):
    @abstractmethod
    async def run(self, input_chunks):
        # 子类实现特定的处理逻辑
        pass

class OpenAI(Node):
    async def run(self, input_chunks):
        # OpenAI 特定实现
        pass
```

### 6. 责任链模式

节点可以链接在一起形成处理流水线：

```python
# 创建节点
node1 = OpenAI(...)
node2 = ReActAgent(...)

# 链接它们: node1 > node2 表示 node1 的输出作为 node2 的输入
pipeline = node1 > node2

# __gt__ 方法建立依赖关系
def __gt__(self, node: "Node") -> "Node":
    if self not in node.deps:
        node.deps.append(self)
    return self
```

### 7. DAG (有向无环图) 模式

节点依赖系统创建 DAG 结构：

- 每个节点维护依赖列表 (`deps`)
- 处理从叶节点流向根节点
- 通过无环性质防止循环
- 支持具有分支和合并的复杂工作流

## 组件关系

### 继承层次

```
Node (base.py)
└── OpenAI (llm.py)
    └── ChatLLM (chat.py)
        └── ReActAgent (re_act.py)
            └── GroupAgent (agent_group.py)
```

### 组合关系

- `ChatLLM` → `MessagesManager`: 用于消息持久化
- `ChatLLM` → `ToolsManager`: 用于工具执行  
- `ReActAgent` → `ChatLLM`: 继承聊天功能
- `ToolsManager` → `Tool`: 管理已注册的工具

### 数据流

1. **输入**: 用户提供字典或 `InputChunk` 流作为输入
2. **处理**: 
   - 系统提示使用用户变量进行格式化
   - 检索对话历史（如果启用）
   - 消息发送到 LLM API
   - 解析响应以查找工具调用或最终答案
   - 执行工具调用并将结果合并
3. **输出**: 包含最终响应的 `InputChunk` 对象流

### 依赖注入

组件通过构造函数注入松散耦合：

```python
agent = ReActAgent(
    model="gpt-4",
    system_prompt="您很有帮助",
    api_key="secret",
    messages_manager=qdrant_manager,  # 注入的依赖
    tools_manager=tools_manager,      # 注入的依赖
    mcp="http://mcp-endpoint"         # 注入的依赖
)
```

## 扩展点

### 添加新的存储后端

实现 `MessagesManager` 接口：

```python
class PostgreSQLMessagesManager(MessagesManager):
    async def get_messages(self, session_id: str, user_input: str = ""):
        # PostgreSQL 实现
        
    async def save_messages(self, session_id: str, messages):
        # PostgreSQL 实现
```

### 添加新的 LLM 提供商

扩展 `Node` 类：

```python
class Anthropic(Node):
    async def run(self, input_chunks):
        # Anthropic API 实现
```

### 自定义工具处理

使用 `hook_tool_call` 参数：

```python
async def custom_tool_handler(agent, function):
    # 添加日志记录、验证或转换
    result = await agent.call_tool(function)
    # 后处理结果
    return result
```

## 最佳实践

### 1. 会话管理
- 始终提供一致的 `session_id` 以维护对话上下文
- 考虑为长时间运行的应用程序进行会话清理

### 2. 错误处理
- 在自定义工具中实现健壮的错误处理
- 使用 `json-repair` 进行弹性 JSON 解析（已集成）

### 3. 性能优化
- 限制对话历史长度以防止令牌溢出
- 使用适当的 `max_loop` 值以防止无限推理循环

### 4. 安全性
- 切勿在客户端代码中暴露 API 密钥
- 验证工具输入以防止注入攻击

### 5. 测试
- 使用提供的测试套件作为实现模式的参考
- 测试流式和非流式模式

## 架构使用示例

### 简单流水线
```python
# 基本 LLM 交互
llm = OpenAI(model="gpt-4", system_prompt="乐于助人的助手", api_key="key")
```

### 增强聊天 Agent
```python
# 带记忆和工具的聊天
messages_manager = QdrantMessagesManager()
tools_manager = ToolsManager()

@tools_manager.tool(name="calculator", ...)
def calculator(expr): return eval(expr)

agent = ChatLLM(
    model="gpt-4",
    system_prompt="数学助手",
    api_key="key",
    messages_manager=messages_manager,
    tools_manager=tools_manager
)
```

### 完整的 ReAct Agent
```python
# 完整的推理和行动 Agent
react_agent = ReActAgent(
    model="gpt-4",
    system_prompt="专家助手",
    api_key="key",
    messages_manager=messages_manager,
    tools_manager=tools_manager,
    with_time=True,
    max_loop=5,
    stream=True
)
```

这种架构使得能够构建复杂的 AI Agent，同时保持代码清晰、可测试性和可扩展性。
