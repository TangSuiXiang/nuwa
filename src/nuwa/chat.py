import json
import logging

from uuid import uuid4
from json_repair import loads
from typing import Optional, AsyncGenerator, Dict, Any, List, Callable, Awaitable
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
)

from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from fastmcp import Client

from .llm import OpenAI
from .base import MessagesManager, InputChunk
from .tool import ToolsManager, get_tool_entity

logger = logging.getLogger()


class ChatLLM(OpenAI):
    def __init__(
        self,
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
        hook_tool_call: Optional[Callable[["ChatLLM", Function], Awaitable[Any]]] = None,
    ):
        self.session_id = session_id or str(uuid4())
        self.hook_tool_call = hook_tool_call
        self.messages_manager = messages_manager
        self.tools_manager = tools_manager
        self.mcp = mcp
        self.mcp_timeout = mcp_timeout
        logger.info("mcp %s", mcp)
        self.tool_names = (
            (tools_manager.list_tools()) if tools_manager else []
        )  # Avoid modifying the original list
        self.new_messages_idx = 0
        self.tools = [
            self.tools_manager.get_tool(name).entity
            for name in self.tool_names
            if self.tools_manager.get_tool(name)
        ]
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

    async def __ainit__(self):
        if self.mcp:
            async with Client(transport=self.mcp, timeout=self.mcp_timeout) as client:
                tools = await client.list_tools()
                logger.info("init_mcp_tools %s", tools)
                for tool in tools:
                    self.tool_names.append(tool.name)
                    self.tools.append(get_tool_entity(tool=tool))

    async def call_tool(self, func: Function):
        if self.tools_manager and self.tools_manager.has_tool(func.get("name")):
            return await self.tools_manager.call_tool(func)
        elif self.mcp:
            async with Client(transport=self.mcp, timeout=self.mcp_timeout) as client:
                ret = await client.call_tool(
                    name=func.get("name"),
                    arguments=loads(func.get("arguments") or "{}"),
                )
                return ret.data
        else:
            raise ValueError(f"function not exists {func}")

    async def generate_messages(
        self, input_dict: Dict[str, Any] = None
    ) -> List[ChatCompletionMessageParam]:
        input_dict = input_dict or {}

        if self.messages_manager and not self.historical_messages:
            self.historical_messages = await self.messages_manager.get_messages(
                self.session_id, input_dict.get("user", "")
            )
            self.new_messages_idx = len(self.historical_messages)
        return await super().generate_messages(input_dict)

    async def save_messages(self):
        logger.info(
            "Save messages %s",
            json.dumps(self.historical_messages, indent=2, ensure_ascii=False),
        )
        if self.messages_manager is not None:
            await self.messages_manager.save_messages(
                self.session_id,
                self.historical_messages[self.new_messages_idx :],
            )
            self.historical_messages.clear()
        elif len(self.historical_messages) > 10:
            while self.historical_messages[0].get("role") != "user":
                del self.historical_messages[0]

    async def run(
        self, input_chunks: AsyncGenerator[InputChunk, None] | Dict[str, Any]
    ) -> AsyncGenerator[InputChunk, None]:
        content = ""

        async for chunk in super().run(input_chunks):
            content += chunk.content
            yield chunk

        self.historical_messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant", content=content.strip()
            )
        )

        await self.save_messages()
