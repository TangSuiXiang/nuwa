"""Nuwa 聊天层模块，扩展 LLM 能力，增加对话管理和工具调用。

本模块提供 ChatLLM 类，继承自 OpenAI 节点，集成消息存储、工具管理器、
MCP（Model Context Protocol）支持，并支持观察者模式（P-004）的工具调用钩子。
遵循组合优于继承原则（C-007），通过依赖注入实现灵活配置。
设计规范：模块化（C-001）、接口设计（C-008）、类型提示（D-014）。
"""

from ast import arguments
import json
import logging

from uuid import uuid4
from json_repair import loads
from typing import Optional, AsyncGenerator, Dict, Any, List, Callable, Awaitable
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
)

from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function as ToolInvocation,
)
from fastmcp import Client

from .llm import LLMNode
from .base import ConversationStorage, StreamChunk
from .tool import ToolRegistry, get_tool_entity

logger = logging.getLogger()


class ConversationAgent(LLMNode):
    """对话代理节点，扩展基础 LLM 功能，增加对话历史和工具调用。

    继承自 LLMNode 类，遵循继承层次（C-005），同时通过组合引入 ConversationStorage
    和 ToolsManager，体现组合优于继承原则（C-007）。
    支持 MCP 协议工具发现，提供工具调用钩子（观察者模式 P-004）。
    设计规范：类名大驼峰（D-007）、单一职责（C-006）、错误处理（C-012）。

    Attributes:
        session_id: 会话标识符，用于隔离不同用户的对话历史。
        hook_tool_call: 工具调用钩子函数，允许外部观察和自定义处理。
        messages_manager: 消息管理器实例，负责消息持久化。
        tools_manager: 工具管理器实例，负责工具注册和调用。
        mcp: MCP 服务器地址，用于动态发现工具。
        mcp_timeout: MCP 请求超时时间（秒）。
        tool_names: 可用工具名称列表。
        new_messages_idx: 新消息起始索引，用于增量保存。
        tools: 工具实体列表，用于构造系统提示。
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: str,
        messages_manager: Optional[ConversationStorage] = None,
        tools_manager: Optional[ToolRegistry] = None,
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
        hook_tool_call: Optional[
            Callable[["ConversationAgent", ToolInvocation], Awaitable[Any]]
        ] = None,
    ):
        """初始化 ConversationAgent 节点。

        Args:
            model: LLM 模型名称。
            system_prompt: 系统提示模板。
            api_key: API 密钥。
            messages_manager: 消息管理器实例，默认为 None（无持久化）。
            tools_manager: 工具管理器实例，默认为 None（无工具）。
            mcp: MCP 服务器地址，默认为 None（禁用 MCP）。
            mcp_timeout: MCP 请求超时时间，默认为 300 秒。
            session_id: 会话 ID，默认为自动生成的 UUID。
            stream: 是否启用流式输出，默认为 False。
            temperature: 生成温度，默认为 0.6。
            extra_body: 额外的 API 请求体参数，默认为空字典。
            with_time: 是否在系统提示中加入当前时间，默认为 False。
            with_others: 其他自定义信息，默认为空字符串。
            stop: 停止词列表，默认为空列表。
            base_url: API 基础 URL，默认为 OpenAI 官方端点。
            hook_tool_call: 工具调用钩子函数，默认为 None。

        设计规范：参数类型提示（D-014）、默认参数合理（C-011）。
        """
        self.session_id = session_id or str(uuid4())
        self.hook_tool_call = hook_tool_call
        self.messages_manager = messages_manager
        self.tools_manager = tools_manager
        self.mcp = mcp
        self.mcp_timeout = mcp_timeout
        logger.debug("mcp %s", mcp)
        self.tool_names = (
            (tools_manager.list_tools()) if tools_manager else []
        )  # Avoid modifying the original list
        self.new_messages_idx = -1
        self.tools = (
            [
                self.tools_manager.get_tool(name).entity  # type: ignore
                for name in self.tool_names
                if self.tools_manager.get_tool(name)
            ]
            if self.tools_manager
            else []
        )
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            api_key=api_key,
            stream=stream,
            temperature=temperature,
            extra_body=extra_body or {},
            with_time=with_time,
            with_others=with_others,
            stop=stop or [],
            base_url=base_url,
        )

    async def _async_initialize(self):
        """异步初始化，动态加载 MCP 工具。

        如果配置了 MCP 服务器，则连接并获取远程工具列表，将其加入可用工具集。
        遵循错误处理规范（C-012），异常由调用者处理。
        """
        if self.mcp:
            async with Client(transport=self.mcp, timeout=self.mcp_timeout) as client:
                tools = await client.list_tools()
                logger.debug("init_mcp_tools %s", tools)
                for tool in tools:
                    self.tool_names.append(tool.name)
                    self.tools.append(get_tool_entity(tool=tool))

    async def call_tool(self, func: ToolInvocation):
        """调用工具函数。

        优先使用本地注册的工具（通过 tools_manager），若未找到则尝试通过 MCP 调用远程工具。
        遵循错误处理规范（C-012），当工具不存在时抛出明确异常。

        Args:
            func: 工具调用规范，包含名称和参数。

        Returns:
            工具执行结果。

        Raises:
            ValueError: 当工具不存在时。
        """
        if self.tools_manager and self.tools_manager.has_tool(func.get("name")):
            return await self.tools_manager.call_tool(func)
        elif self.mcp:
            async with Client(transport=self.mcp, timeout=self.mcp_timeout) as client:
                _arguments = loads(func.get("arguments") or "{}")
                if not isinstance(_arguments, dict):
                    raise
                ret = await client.call_tool(
                    name=func.get("name"),
                    arguments=_arguments,
                )
                return ret.data
        else:
            raise ValueError(f"function not exists {func}")

    async def _prepare_conversation_messages(
        self, input_dict: Dict[str, Any] | None = None
    ) -> List[ChatCompletionMessageParam]:
        """生成发送给 LLM 的消息列表，集成消息存储。

        如果配置了消息管理器且历史消息为空，则从存储中检索历史消息。
        重写父类方法以支持对话历史持久化，遵循模板方法模式（P-005）。
        设计规范：方法重写清晰、类型提示（D-014）。

        Args:
            input_dict: 输入字典，包含用户输入等字段。

        Returns:
            符合 OpenAI ChatCompletion 接口要求的消息列表。
        """
        input_dict = input_dict or {}

        if self.messages_manager and not self.historical_messages:
            self.historical_messages = await self.messages_manager.get_messages(
                self.session_id, input_dict.get("user", "")
            )
        messages = await super()._prepare_conversation_messages(input_dict)
        if self.new_messages_idx == -1:
            self.new_messages_idx = max(len(self.historical_messages) - 1, 0)
        return messages

    async def save_messages(self):
        """保存当前会话的消息到持久化存储。

        如果配置了消息管理器，则保存新消息（从 new_messages_idx 到末尾）并清空历史缓存；
        否则，当历史消息超过 10 条时，清理最早的非用户消息以避免内存膨胀。
        遵循日志记录规范（C-016）和错误处理规范（C-012）。
        """
        logger.debug(
            "Save messages %s",
            json.dumps(self.historical_messages, indent=2, ensure_ascii=False),
        )
        if self.messages_manager is not None:
            await self.messages_manager.save_messages(
                self.session_id,
                self.historical_messages[self.new_messages_idx :],
            )
            self.new_messages_idx = -1
            self.historical_messages.clear()
        elif len(self.historical_messages) > 10:
            while self.historical_messages[0].get("role") != "user":
                del self.historical_messages[0]

    async def clear_chat_history(self):
        """清空当前会话的历史聊天消息。

        如果配置了消息管理器，则调用其 clear_messages 方法；同时清空内存中的历史消息。
        遵循错误处理规范（C-012），异常由调用者处理。
        """
        if self.messages_manager:
            await self.messages_manager.clear_messages(self.session_id)
        self.historical_messages.clear()
        self.new_messages_idx = -1
        logger.debug(f"Cleared chat history for session {self.session_id}")

    async def run(
        self, input_chunks: AsyncGenerator[StreamChunk, None] | Dict[str, Any]
    ) -> AsyncGenerator[StreamChunk, None]:
        """执行聊天 LLM 处理，并保存对话历史。

        重写父类 run 方法，在生成所有输出后，将助手回复添加到历史消息并保存。
        遵循模板方法模式（P-005）和流式处理规范（C-010）。

        Args:
            input_chunks: 输入数据，可以是异步生成器或字典。

        Yields:
            StreamChunk: 输出数据块，状态为 "DOING" 或 "END"。
        """
        content = ""
        async for chunk in super().run(input_chunks):
            if isinstance(chunk.content, str):
                content += chunk.content
            yield chunk

        self.historical_messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant", content=content.strip()
            )
        )

        await self.save_messages()
