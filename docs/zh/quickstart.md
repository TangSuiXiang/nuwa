# Nuwa 快速入门指南

几分钟内快速上手 Nuwa！本指南涵盖创建第一个 AI Agent 的基本步骤。

## 安装

### 先决条件
- Python 3.10 或更高版本
- 来自 OpenAI 兼容 LLM 提供商的 API 密钥（OpenAI、Anthropic、DeepSeek 等）

### 安装 Nuwa
```bash
pip install nuwa
```

或者从源码安装：
```bash
git clone https://github.com/your-repo/nuwa.git
cd nuwa
pip install .
```

## 您的第一个 Agent

### 1. 简单的 LLM 交互

创建一个对用户输入做出响应的基本 Agent：

```python
# simple_agent.py
import asyncio
from src.nuwa.llm import OpenAI

async def main():
    # 初始化 LLM
    llm = OpenAI(
        model="gpt-4",
        system_prompt="您是一个乐于助人的助手。",
        api_key="your-api-key-here"
    )
    
    # 获取用户输入
    user_input = "您好！您能做什么？"
    
    # 生成响应
    print("Agent: ", end="")
    async for chunk in llm.run({"user": user_input}):
        print(chunk.content, end="", flush=True)
    print()  # 结尾换行

if __name__ == "__main__":
    asyncio.run(main())
```

运行它：
```bash
python simple_agent.py
```

### 2. 带记忆的聊天 Agent

创建一个能记住之前交互的对话 Agent：

```python
# chat_agent.py
import asyncio
from src.nuwa.chat import ChatLLM
from src.nuwa.base import MessagesManager

class SimpleMemory(MessagesManager):
    def __init__(self):
        self.messages = {}
    
    async def get_messages(self, session_id: str, user_input: str = ""):
        return self.messages.get(session_id, [])
    
    async def save_messages(self, session_id: str, messages):
        if session_id not in self.messages:
            self.messages[session_id] = []
        self.messages[session_id].extend(messages)

async def main():
    # 使用记忆初始化
    memory = SimpleMemory()
    chat_agent = ChatLLM(
        model="gpt-4",
        system_prompt="您是一个友好的对话式 AI。",
        api_key="your-api-key-here",
        messages_manager=memory,
        session_id="demo_session"
    )
    
    # 第一次交互
    print("用户: 您叫什么名字？")
    print("Agent: ", end="")
    async for chunk in chat_agent.run({"user": "您叫什么名字？"}):
        print(chunk.content, end="", flush=True)
    print("\n")
    
    # 第二次交互（带有上下文）
    print("用户: 很高兴认识您！")
    print("Agent: ", end="")
    async for chunk in chat_agent.run({"user": "很高兴认识您！"}):
        print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. 带工具的 ReAct Agent

创建一个能够推理并使用工具的高级 Agent：

```python
# react_agent.py
import asyncio
from src.nuwa.re_act import ReActAgent
from src.nuwa.tool import ToolsManager, ToolObjectParameter, ToolParameter

# 创建工具管理器
tools = ToolsManager()

# 注册计算器工具
@tools.tool(
    name="calculate",
    description="执行数学计算",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "expression": ToolParameter(
                type="string", 
                description="要计算的数学表达式"
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
    # 使用工具初始化 ReAct Agent
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个具有计算能力的乐于助人的助手。",
        api_key="your-api-key-here",
        tools_manager=tools,
        with_time=True,  # 包含当前时间
        stream=True,     # 启用流式响应
        max_loop=5       # 允许最多 5 次推理循环
    )
    
    # 提出需要工具使用的问题
    question = "123 * 456 + 789 等于多少？"
    print(f"用户: {question}")
    print("Agent: ", end="")
    
    async for chunk in agent.run({"user": question}):
        if chunk.state == "END":
            print(f"\n最终答案: {chunk.content}")
        else:
            print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

## 配置技巧

### API 密钥和提供商

Nuwa 可与任何 OpenAI 兼容的 API 配合使用。配置您的提供商：

```python
# OpenAI
agent = ReActAgent(
    model="gpt-4",
    api_key="sk-your-openai-key",
    base_url="https://api.openai.com/v1"
)

# Anthropic（通过 OpenAI 兼容层）
agent = ReActAgent(
    model="claude-3-opus-20240229",
    api_key="your-anthropic-key",
    base_url="https://api.anthropic.com/v1/messages"  # 根据需要调整
)

# DeepSeek
agent = ReActAgent(
    model="deepseek-v3.2",
    api_key="your-deepseek-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

### 环境变量

使用环境变量安全地存储您的 API 密钥：

```python
import os
from src.nuwa.re_act import ReActAgent

agent = ReActAgent(
    model="gpt-4",
    system_prompt="您是一个乐于助人的助手。",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
```

设置环境变量：
```bash
export OPENAI_API_KEY="your-api-key-here"
python your_agent.py
```

## 常见模式

### 模式 1：多轮对话
```python
async def multi_turn_conversation():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个乐于助人的助手。",
        api_key="your-key",
        enable_chat_history=True
    )
    
    conversation = [
        "法国的首都是哪里？",
        "那里有多少人口？",
        "有哪些著名地标？"
    ]
    
    for message in conversation:
        print(f"用户: {message}")
        print("Agent: ", end="")
        async for chunk in agent.run({"user": message}):
            print(chunk.content, end="", flush=True)
        print("\n")
```

### 模式 2：带异步函数的自定义工具
```python
@tools.tool(
    name="fetch_weather",
    description="获取当前天气信息",
    parameters=ToolObjectParameter(
        type="object",
        properties={
            "city": ToolParameter(type="string", description="城市名称")
        }
    )
)
async def fetch_weather(city: str):
    # 模拟异步 API 调用
    await asyncio.sleep(1)
    return {"city": city, "temperature": "22°C", "condition": "晴朗"}
```

### 模式 3：带思考模式的流式响应
```python
agent = ReActAgent(
    model="deepseek-v3.2",
    system_prompt="在回答前逐步思考。",
    api_key="your-key",
    extra_body={"enable_thinking": True},
    stream=True,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

async for chunk in agent.run({"user": "解释量子纠缠"}):
    print(chunk.content, end="", flush=True)
```

## 故障排除

### 常见问题和解决方案

**问题**: `RateLimitError`
- **解决方案**: 添加重试逻辑或减少请求频率
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
                await asyncio.sleep(2 ** attempt)  # 指数退避
            else:
                raise
```

**问题**: 工具未找到
- **解决方案**: 确保工具名称完全匹配，并在 Agent 初始化之前注册

**问题**: JSON 解析错误
- **解决方案**: 框架内部使用 `json-repair`，但确保您的工具返回有效的类 JSON 结构

### 调试

启用调试日志以查看内部操作：
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 现在您将看到工具调用、消息流等的详细日志
```

## 后续步骤

1. **探索示例**：查看 `tests/` 目录中的更多综合示例
2. **自定义 MessagesManager**：使用 Qdrant 等数据库实现持久化存储
3. **高级工具**：创建与外部 API 和服务集成的工具
4. **MCP 集成**：连接到模型上下文协议端点以扩展功能
5. **性能优化**：调整 `max_loop`、`temperature` 和流式设置等参数

## 资源

- [完整文档](README.md)
- [API 参考](api_reference.md)
- [测试示例](../../tests/)
- [源代码](../../src/nuwa/)

祝您使用 Nuwa 开发愉快！🎉
