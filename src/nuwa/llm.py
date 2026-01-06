import json
import asyncio
import logging

from typing import AsyncGenerator, List, Dict, Any, Union, Optional
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from openai import omit

from .base import Node, InputChunk
from datetime import datetime

logger = logging.getLogger()


class OpenAI(Node):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: str,
        temperature: float = 0.6,
        extra_body: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        with_time: bool = False,
        base_url: str = "https://api.openai.com/v1",
        stream: bool = False,
    ):
        self.system_prompt = system_prompt
        self.with_time = with_time
        self.model = model
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.need_init = True
        self.stop = stop or []
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.think_open = "<think>"
        self.symbols_open_close_map = {self.think_open: "</think>"}
        self.historical_messages: List[ChatCompletionMessageParam] = []
        self.stream = stream

    async def __ainit__(self):
        logger.info("ainit")

    async def parse_input(
        self, input_chunks: Union[AsyncGenerator[InputChunk, None], Dict[str, Any]]
    ) -> Dict[str, Any]:
        if isinstance(input_chunks, AsyncGenerator):
            input_raw = ""
            async for chunk in input_chunks:
                input_raw += chunk.content
                if chunk.state == "END":
                    break
            try:
                return json.loads(input_raw)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse input as JSON: {e}")
                raise ValueError(f"Input is not valid JSON: {input_raw}") from e
        return input_chunks

    async def generate_messages(
        self, input_dict: Dict[str, Any]
    ) -> List[ChatCompletionMessageParam]:
        input_dict = input_dict or {}
        system_content = self.system_prompt.format(**input_dict.get("system", {}))
        if self.with_time:
            system_content = (
                "系统时间：{}\n\n".format(datetime.now().isoformat()) + system_content
            )
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
        self, input_chunks: Union[AsyncGenerator[InputChunk, None], Dict[str, Any]]
    ) -> AsyncGenerator[InputChunk, None]:
        logger.info("need init %s", self.need_init)
        if self.need_init:
            await self.__ainit__()
            self.need_init = False
        while True:
            try:
                input_dict = await self.parse_input(input_chunks)
                logger.debug(
                    f"Model: {self.model}, Extra body: {self.extra_body}, Stop: {self.stop}"
                )
                messages = await self.generate_messages(input_dict=input_dict)
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

                if self.stream:
                    content = ""
                    reasoning_content = ""
                    in_reasoning_mode = False
                    async for chunk in completion:
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta
                        logger.info("delta %s", delta)

                        if delta.content:
                            if in_reasoning_mode:
                                yield InputChunk(
                                    state="DOING",
                                    content=self.symbols_open_close_map.get(
                                        self.think_open
                                    ),
                                )
                                in_reasoning_mode = False
                            content += delta.content
                            yield InputChunk(state="DOING", content=delta.content)
                        elif (
                            hasattr(delta, "reasoning_content")
                            and delta.reasoning_content
                        ):
                            if not in_reasoning_mode:
                                yield InputChunk(state="DOING", content=self.think_open)
                                in_reasoning_mode = True
                            reasoning_content += delta.reasoning_content
                            yield InputChunk(
                                state="DOING", content=delta.reasoning_content
                            )
                    if content == "" and in_reasoning_mode:
                        yield InputChunk(
                            state="DOING",
                            content=self.symbols_open_close_map.get(self.think_open),
                        )
                        yield InputChunk(state="END", content=reasoning_content)
                    else:
                        yield InputChunk(state="END", content="")
                else:
                    yield InputChunk(
                        state="END", content=completion.choices[0].message.content
                    )
                break
            except RateLimitError:
                await asyncio.sleep(0.1)
