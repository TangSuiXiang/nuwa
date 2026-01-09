# Nuwa

## 一位能够创造 AI 生命的女神

Nuwa 是一个功能强大的 AI Agent 框架，支持 ReAct（推理与行动）模式、工具集成和对话管理等高级功能。

## 文档

- [中文文档](docs/zh/) - 完整的中文使用指南
- [英文文档](docs/) - 英文文档（默认）

## 快速开始

### 安装
```bash
pip install nuwa
```

### 基本使用
```python
import asyncio
from src.nuwa.re_act import ReActAgent

async def main():
    agent = ReActAgent(
        model="gpt-4",
        system_prompt="您是一个乐于助人的助手。",
        api_key="your-api-key"
    )
    
    async for chunk in agent.run({"user": "您好！"}):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

## 主要特性

- **ReAct 模式**: 支持推理与行动的结构化交互
- **工具集成**: 轻松注册和使用自定义工具
- **多轮对话**: 内置对话历史管理
- **MCP 支持**: 集成模型上下文协议
- **流式响应**: 实时生成响应内容
- **时间感知**: 自动包含时间戳信息

## 学习资源

- [快速入门指南](docs/zh/quickstart.md)
- [完整使用文档](docs/zh/README.md)
- [API 参考](docs/zh/api_reference.md)
- [测试示例](tests/)

## 贡献

欢迎提交 Issue 和 Pull Request！
