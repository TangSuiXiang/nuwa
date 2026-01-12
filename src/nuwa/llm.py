"""Nuwa LLM 层模块，实现与大语言模型的基础交互。

本模块提供 OpenAI 兼容的 LLM 节点实现，支持流式和非流式响应、系统提示格式化、
时间感知和自定义上下文注入。遵循策略模式（P-002）和模板方法模式（P-005），
继承自 Node 抽象基类，是 Nuwa 处理流水线的核心组件之一。
设计规范：模块化（C-001）、类设计（C-005）、错误处理（C-012）、日志记录（C-016）。
"""

import os
import json
import asyncio
import logging

from typing import AsyncGenerator, List, Dict, Any, Union, Optional
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletion,
)

from openai import omit

from .base import ProcessingNode, StreamChunk
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger()


class LLMNode(ProcessingNode):
    """LLM 节点实现，兼容 OpenAI 接口。

    遵循策略模式（P-002），作为 ProcessingNode 的具体策略，提供与 OpenAI 兼容 API 的交互能力。
    支持流式输出、推理内容分离、历史消息管理和时间感知。
    设计规范：类名大驼峰（D-007）、类型提示（D-014）、单一职责（C-006）。

    Attributes:
        system_prompt: 系统提示模板，支持格式化变量。
        with_time: 是否在系统提示中加入当前时间。
        with_others: 其他自定义信息，将添加到系统提示前。
        model: 使用的 LLM 模型名称。
        temperature: 生成温度参数。
        extra_body: 额外的 API 请求体参数。
        need_init: 标识是否需要初始化。
        stop: 停止词列表。
        client: AsyncOpenAI 客户端实例。
        think_open: 推理开始标记字符串。
        symbol_mappings: 符号开闭映射表。
        historical_messages: 历史消息缓存。
        stream: 是否启用流式输出。
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: str,
        temperature: float = 0.6,
        extra_body: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        with_time: bool = False,
        with_others: str = "",
        base_url: str = "https://api.openai.com/v1",
        stream: bool = False,
    ):
        """初始化 LLM 节点。

        Args:
            model: LLM 模型名称，如 "gpt-4"。
            system_prompt: 系统提示模板，可包含格式化占位符。
            api_key: OpenAI 兼容 API 的密钥。
            temperature: 生成温度，默认为 0.6。
            extra_body: 额外的 API 请求体参数，默认为空字典。
            stop: 停止词列表，默认为空列表。
            with_time: 是否在系统提示中加入当前时间，默认为 False。
            with_others: 其他自定义信息字符串，将添加到系统提示前，默认为空字符串。
            base_url: API 基础 URL，默认为 OpenAI 官方端点。
            stream: 是否启用流式输出，默认为 False。

        设计规范：参数类型提示（D-014）、默认参数合理（C-011）。
        """
        self.system_prompt = system_prompt
        self.with_time = with_time
        self.with_others = with_others
        self.model = model
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.need_init = True
        self.stop = stop or []
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.think_open = "<think>"
        self.symbol_mappings = {self.think_open: "</think>"}
        self.historical_messages: List[ChatCompletionMessageParam] = []
        self.stream = stream

    async def _async_initialize(self):
        """异步初始化方法，用于执行一次性初始化任务。

        当前仅记录调试日志，可扩展用于连接验证等。
        设计规范：异步方法命名（D-006）、日志记录（C-016）。
        """
        logger.debug("ainit")

    async def _parse_input_data(
        self, input_chunks: Union[AsyncGenerator[StreamChunk, None], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """解析输入数据，转换为字典格式。

        支持两种输入形式：异步生成器（流式）或字典（非流式）。
        若为异步生成器，则收集所有 chunk 的内容并尝试解析为 JSON 字典；
        若为字典，则直接返回。
        遵循错误处理规范（C-012），对 JSON 解析失败抛出明确异常。

        Args:
            input_chunks: 输入数据，可以是异步生成器或字典。

        Returns:
            解析后的字典，包含用户输入、系统变量等。

        Raises:
            ValueError: 当输入为异步生成器且内容不是合法 JSON 时。
        """
        if isinstance(input_chunks, AsyncGenerator):
            input_raw = ""
            async for chunk in input_chunks:
                if isinstance(chunk.content, str):
                    input_raw += chunk.content
                    if chunk.state == "END":
                        break
            try:
                return json.loads(input_raw)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse input as JSON: {e}")
                raise ValueError(f"Input is not valid JSON: {input_raw}") from e
        return input_chunks

    async def _prepare_conversation_messages(
        self, input_dict: Dict[str, Any]
    ) -> List[ChatCompletionMessageParam]:
        """生成发送给 LLM 的消息列表。

        根据输入字典格式化系统提示，加入时间信息和自定义前缀，
        并整合历史消息，确保最后一条消息为用户输入。
        设计规范：函数职责清晰（C-006）、类型提示（D-014）。

        Args:
            input_dict: 输入字典，包含 "system"、"user"、"historical_messages" 等字段。

        Returns:
            符合 OpenAI ChatCompletion 接口要求的消息列表。
        """
        input_dict = input_dict or {}
        system_content = self.system_prompt.format(**input_dict.get("system", {}))
        proset_infos = []
        if self.with_others:
            proset_infos.append(self.with_others)
        if self.with_time:
            proset_infos.append(
                "系统时间：{}".format(
                    datetime.now(
                        tz=ZoneInfo(os.environ.get("TZ") or "Asia/Shanghai")
                    ).isoformat()
                )
            )
        if proset_infos:
            proset_infos.append("\n")
        system_content = "\n".join(proset_infos) + system_content
        user_content = input_dict.get("user", "")
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_content),
        ]
        self.historical_messages = (
            input_dict.get("historical_messages", []) + self.historical_messages
        )
        if (
            not self.historical_messages
            or self.historical_messages[-1].get("role") != "user"
        ):
            self.historical_messages.append(
                ChatCompletionUserMessageParam(role="user", content=user_content)
            )
        messages.extend(self.historical_messages)
        return messages

    async def run(
        self, input_chunks: Union[AsyncGenerator[StreamChunk, None], Dict[str, Any]]
    ) -> AsyncGenerator[StreamChunk, None]:
        """执行 LLM 调用，返回流式或非流式输出。

        遵循模板方法模式（P-005），实现了 ProcessingNode.run 的具体逻辑。
        处理流程：解析输入 → 生成消息 → 调用 API → 处理响应 → 输出 chunk。
        支持流式输出中的推理内容分离，并自动处理速率限制错误（C-013）。
        设计规范：复杂算法注释（D-012）、错误处理（C-012）、日志记录（C-016）。

        Args:
            input_chunks: 输入数据，可以是异步生成器或字典。

        Yields:
            StreamChunk: 输出数据块，状态为 "DOING" 或 "END"。

        Raises:
            RateLimitError: 当 API 速率限制时重试（已内部处理）。
            Exception: 其他未捕获的异常会向上抛出。
        """
        logger.debug("need init %s", self.need_init)
        if self.need_init:
            await self._async_initialize()
            self.need_init = False
        while True:
            try:
                input_dict = await self._parse_input_data(input_chunks)
                logger.debug(
                    f"Model: {self.model}, Extra body: {self.extra_body}, Stop: {self.stop}"
                )
                messages = await self._prepare_conversation_messages(
                    input_dict=input_dict
                )
                logger.debug(
                    "messages %s", json.dumps(messages, ensure_ascii=False, indent=2)
                )
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    extra_body=self.extra_body,
                    stream=self.stream,
                    stream_options={"include_usage": True} if self.stream else omit,
                    stop=self.stop,
                )

                if self.stream and not isinstance(completion, ChatCompletion):
                    content = ""
                    reasoning_content = ""
                    in_reasoning_mode = False
                    async for chunk in completion:
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta
                        logger.debug("delta %s", delta)

                        # 处理普通文本内容
                        if delta.content:
                            if in_reasoning_mode:
                                # 推理模式结束，输出闭合标记
                                yield StreamChunk(
                                    state="DOING",
                                    content=self.symbol_mappings.get(
                                        self.think_open, "</think>"
                                    ),
                                )
                                in_reasoning_mode = False
                            content += delta.content
                            yield StreamChunk(state="DOING", content=delta.content)
                        # 处理推理内容（如 OpenAI o1 模型）
                        elif hasattr(delta, "reasoning_content") and getattr(
                            delta, "reasoning_content"
                        ):
                            if not in_reasoning_mode:
                                # 推理模式开始，输出开始标记
                                yield StreamChunk(
                                    state="DOING", content=self.think_open
                                )
                                in_reasoning_mode = True
                            reasoning_content += getattr(delta, "reasoning_content")
                            yield StreamChunk(
                                state="DOING",
                                content=getattr(delta, "reasoning_content"),
                            )
                    # 处理流结束时的边界情况
                    if content == "" and in_reasoning_mode:
                        yield StreamChunk(
                            state="DOING",
                            content=self.symbol_mappings.get(
                                self.think_open, "</think>"
                            ),
                        )
                        yield StreamChunk(state="END", content=reasoning_content)
                    else:
                        yield StreamChunk(state="END", content="")
                elif isinstance(completion, ChatCompletion):
                    # 非流式输出，直接返回完整内容
                    yield StreamChunk(
                        state="END", content=str(completion.choices[0].message.content)
                    )
                break
            except RateLimitError:
                # 速率限制时等待后重试，遵循错误处理规范（C-013）
                await asyncio.sleep(0.1)
