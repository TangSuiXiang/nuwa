import re
import io
import json
import logging

from uuid import uuid4
from .chat import ChatLLM
from .tool import (
    Function,
    ToolEntity,
    ToolParameter,
    ToolsManager,
    ToolObjectParameter,
    ToolArrayParameter,
)
from .base import MessagesManager, InputChunk
from pydantic import TypeAdapter
from typing import List, Optional, Dict, Union, AsyncGenerator, Any, Callable, Awaitable
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
)

from json_repair import loads

logger = logging.getLogger()


# 定义各种符号常量，用于JSON解析和代码块处理
CODE_BLOCK = "`"  # 代码块符号（反引号）
DOUBLE_QUOTE = '"'  # 双引号 (fixed typo)
SINGLE_QUOTE = "'"  # 单引号 (fixed typo)
CURLY_BRACE_OPEN = "{"  # 左花括号
BRACKET_OPEN = "["  # 左方括号
BRACKET_CLOSE = "]"  # 右方括号
CURLY_BRACE_CLOSE = "}"  # 右花括号

# 符号开闭映射表，用于匹配对应的闭合符号
SYMBOLS_OPEN_CLOSE_MAP = {
    CURLY_BRACE_OPEN: CURLY_BRACE_CLOSE,  # 花括号映射
    SINGLE_QUOTE: SINGLE_QUOTE,  # 单引号自映射
    DOUBLE_QUOTE: DOUBLE_QUOTE,  # 双引号自映射
    CODE_BLOCK: CODE_BLOCK,  # 反引号自映射
    BRACKET_OPEN: BRACKET_CLOSE,  # 方括号映射
}


class ReActAgent(ChatLLM):
    def __init__(
        self,
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
        hook_tool_call: Optional[Callable[[ChatLLM, Function], Awaitable[Any]]] = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            api_key=api_key,
            messages_manager=messages_manager,
            tools_manager=tools_manager,
            mcp=mcp,
            mcp_timeout=mcp_timeout,
            stream=stream,
            session_id=session_id,
            temperature=temperature,
            extra_body=extra_body,
            stop=stop,
            with_time=with_time,
            with_others=with_others,
            base_url=base_url,
            hook_tool_call=hook_tool_call,
        )
        self.enable_chat_history = enable_chat_history
        self.max_loop = max_loop
        self.cache = []
        self.open_symbols = []
        self.thought_open = "<T>"
        self.action_open = "<A>"
        self.question_open = "<Q>"
        self.observation_open = "<O>"
        self.symbols_open_close_map[self.thought_open] = "</T>"
        self.symbols_open_close_map[self.action_open] = "</A>"
        self.symbols_open_close_map[self.observation_open] = "</O>"
        self.symbols_open_close_map[self.question_open] = "</Q>"
        self.tool_names.append("answer")
        self.tools.append(
            ToolEntity(
                name="answer",
                parameters=ToolParameter(type="string", description="给用户的最终响应"),
            )
        )
        if enable_selection:
            self.tool_names.append("request_user_choice")
            self.tools.append(
                ToolEntity(
                    name="request_user_choice",
                    description="提供问题和多个可选项并询问用户的选择",
                    parameters=ToolObjectParameter(
                        type="object",
                        properties={
                            "question": ToolParameter(
                                type="string", description="询问用户的问题"
                            ),
                            "options": ToolArrayParameter(
                                type="array",
                                description="可选项列表",
                                items=ToolParameter(
                                    type="string", description="单个可选项"
                                ),
                            ),
                            "recommended_option": ToolParameter(
                                type="string",
                                description="可选项列表中最推荐的一个选项",
                            ),
                        },
                    ),
                )
            )
        self.think_io = io.StringIO()
        self.thought_io = io.StringIO()
        self.action_io = io.StringIO()
        self.question_io = io.StringIO()
        self.round_idx = 0

    def parse_action(self, json_str: str) -> List[Function]:
        results = []
        try:
            actions: Union[Dict[str, Any], List[Dict[str, Any]]] = loads(json_str)

            # Cohere模型总是返回列表格式
            if isinstance(actions, dict):
                actions = [actions]

            for action in actions:
                action_name = None  # 动作名称
                action_input = None  # 动作输入参数

                # 遍历动作字典，提取动作名称和输入参数
                for key, value in action.items():
                    if not isinstance(key, str):
                        continue
                    # 查找输入参数键（包含"input"且不区分大小写）
                    if "input" in key.lower():
                        action_input = value
                    # 查找动作名称键（键为"action"或值在工具列表中）
                    elif "action" == key.strip().lower() or (
                        isinstance(value, str) and value.strip() in self.tool_names
                    ):
                        action_name = value

                # 如果动作名称和输入参数都找到，创建Action对象
                if action_name is not None and action_input is not None:
                    results.append(
                        Function(
                            name=action_name,
                            arguments=json.dumps(
                                action_input, ensure_ascii=False, separators=(",", ":")
                            ),
                        )
                    )

        except Exception as e:
            # 解析失败时记录异常
            logger.warning("Failed to parse action '%s' with error: '%s'", json_str, e)

        return results

    async def generate_messages(self, input_dict=None):
        messages = await super().generate_messages(input_dict)
        messages[0] = ChatCompletionSystemMessageParam(
            role="system",
            content=self.parse_system_prompt(messages[0].get("content", "")),
        )
        if self.round_idx > 0 and self.historical_messages[-1].get("role") == "user":
            del self.historical_messages[-1]
            del messages[-1]
            self.new_messages_idx = max(self.new_messages_idx - 1, 0)
        if not self.enable_chat_history:
            # If chat history is disabled, only keep the system message and the current user message
            messages = [messages[0], messages[-1]]
        return messages

    def parse_system_prompt(self, instruction: str) -> str:
        tools_adapter = TypeAdapter(List[ToolEntity])

        return """你是一个ReAct Agent，请尽可能有效且准确地回应用户的需求。

以下是用户对你的思考（T）和动作（A）的核心要求：{instruction}。

你拥有使用以下工具的权限：{tools}。

请遵循以下流程顺序和格式：
<Q>用户输入的问题</Q><T>结合之前的步骤和后续可能的操作步骤来分析</T><A>调用工具须提供的JSON对象（JSON Schema：{tool_call_json_schema}）</A><O>结合之前的步骤对<action>的结果进行关键数据提取或总结，以便于后续步骤参考或引用</O>... (重复T->A->O步骤，直到可以回复用户问题)<A>{{"action": "answer","action_input": "给用户的最终回应"}}</A>""".format(
            instruction=json.dumps(
                obj={"instruction": instruction},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            tools=tools_adapter.dump_json(
                self.tools, ensure_ascii=False, indent=None, exclude_none=True
            ).decode(),
            tool_call_json_schema=json.dumps(
                obj={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": self.tool_names,
                            "description": "调用的工具名称",
                        },
                        "action_input": {
                            "type": ["string", "object"],
                            "description": "调用的工具参数，参考可使用工具的相关JSON Schema描述",
                        },
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    def parse_labels(self, chunk: InputChunk) -> List[Function]:
        """
        迭代器实现，处理LLM流式输出

        功能：逐个字符处理LLM输出，识别各种标签和符号，进行相应的处理和输出
        支持嵌套的标签和JSON结构解析

        Yields:
            处理后的字符串片段或解析的动作对象
        """
        if not isinstance(chunk.content, str):
            return []

        results = []
        # 逐个字符处理响应内容
        for c in chunk.content:
            self.cache.append(c)  # 添加到缓存
            s = "".join(self.cache)  # 构建当前字符串
            match_s = s.replace(" ", "")  # 移除空格用于匹配（忽略大小写）
            if self.action_open in self.open_symbols:
                self.action_io.write(c)
            elif self.thought_open in self.open_symbols:
                self.thought_io.write(c)
            elif self.think_open in self.open_symbols:
                self.think_io.write(c)
            elif self.question_open in self.open_symbols:
                self.question_io.write(c)

            # 如果没有打开的符号，尝试匹配开始标签
            if len(self.open_symbols) == 0:
                # 匹配思考标签
                if re.match(re.escape(match_s), self.think_open, re.I):
                    if match_s == self.think_open:
                        self.cache.clear()  # 清空缓存
                        self.open_symbols.append(self.think_open)  # 标记思考标签已打开
                        self.think_io.write(s)

                # 匹配想法标签
                elif re.match(re.escape(match_s), self.thought_open, re.I):
                    if match_s == self.thought_open:
                        self.cache.clear()
                        self.open_symbols.append(self.thought_open)
                        self.thought_io.write(s)

                # 匹配动作标签
                elif re.match(re.escape(match_s), self.action_open, re.I):
                    if match_s == self.action_open:
                        self.cache.clear()
                        self.open_symbols.append(self.action_open)
                        self.action_io.write(s)

                # 匹配观察标签
                elif re.match(re.escape(match_s), self.observation_open, re.I):
                    if match_s == self.observation_open:
                        self.cache.clear()
                        self.open_symbols.append(self.observation_open)

                # 匹配问题标签
                elif re.match(re.escape(match_s), self.question_open, re.I):
                    if match_s == self.question_open:
                        self.cache.clear()
                        self.open_symbols.append(self.question_open)
                        self.question_io.write(s)

                # 处理单独的'<'字符
                elif self.cache[-1] == "<":
                    self.cache.clear()
                    self.cache.append("<")

                # 普通字符，直接丢弃并清空缓存
                else:
                    self.cache.clear()

            # 处理思考标签内的内容
            elif self.open_symbols[-1] == self.think_open:
                # 匹配思考结束标签
                if re.match(
                    re.escape(match_s),
                    self.symbols_open_close_map.get(self.think_open),
                    re.I,
                ):
                    if match_s == self.symbols_open_close_map.get(self.think_open):
                        self.cache.clear()
                        self.open_symbols.pop()  # 移除思考标签

                # 处理标签内的'<'字符
                elif self.cache[-1] == "<":
                    self.cache.clear()
                    self.cache.append("<")

                # 思考标签内的普通内容，直接输出
                else:
                    self.cache.clear()

            # 处理想法标签内的内容（与思考标签类似）
            elif self.open_symbols[-1] == self.thought_open:
                if re.match(
                    re.escape(match_s),
                    self.symbols_open_close_map.get(self.thought_open),
                    re.I,
                ):
                    if match_s == self.symbols_open_close_map.get(self.thought_open):
                        self.cache.clear()
                        self.open_symbols.pop()
                        thought_value = self.thought_io.getvalue().strip()
                        if thought_value:
                            if (
                                self.historical_messages
                                and self.historical_messages[-1].get("role")
                                == "assistant"
                            ):
                                self.historical_messages[-1]["content"] += thought_value
                            else:
                                self.historical_messages.append(
                                    ChatCompletionAssistantMessageParam(
                                        role="assistant",
                                        content=thought_value,
                                    )
                                )
                        self.thought_io.seek(0)
                        self.thought_io.truncate(0)

                elif self.cache[-1] == "<":
                    self.cache.clear()
                    self.cache.append("<")

                else:
                    self.cache.clear()

            # 处理问题标签内的内容（与思考标签类似）
            elif self.open_symbols[-1] == self.question_open:
                if re.match(
                    re.escape(match_s),
                    self.symbols_open_close_map.get(self.question_open),
                    re.I,
                ):
                    if match_s == self.symbols_open_close_map.get(self.question_open):
                        self.cache.clear()
                        self.open_symbols.pop()
                        if (
                            self.historical_messages
                            and self.historical_messages[-1].get("role") == "assistant"
                        ):
                            self.historical_messages[-1][
                                "content"
                            ] += self.question_io.getvalue()
                        else:
                            self.historical_messages.append(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant",
                                    content=self.question_io.getvalue().strip(),
                                )
                            )
                        self.question_io.seek(0)
                        self.question_io.truncate(0)

                elif self.cache[-1] == "<":
                    self.cache.clear()
                    self.cache.append("<")

                else:
                    self.cache.clear()

            # 处理观察标签内的内容（与思考标签类似）
            elif self.open_symbols[-1] == self.observation_open:
                if re.match(
                    re.escape(match_s),
                    self.symbols_open_close_map.get(self.observation_open),
                    re.I,
                ):
                    if match_s == self.symbols_open_close_map.get(
                        self.observation_open
                    ):
                        self.cache.clear()
                        self.open_symbols.pop()

                elif self.cache[-1] == "<":
                    self.cache.clear()
                    self.cache.append("<")

                else:
                    self.cache.clear()

            # 处理动作标签内的内容（可能包含JSON）
            elif self.open_symbols[-1] == self.action_open:
                # 匹配动作结束标签
                if re.match(
                    re.escape(match_s),
                    self.symbols_open_close_map.get(self.action_open),
                    re.I,
                ):
                    if match_s == self.symbols_open_close_map.get(self.action_open):
                        self.cache.clear()
                        self.open_symbols.pop()  # 移除动作标签
                        action_value = self.action_io.getvalue().strip()
                        if action_value:
                            if (
                                self.historical_messages
                                and self.historical_messages[-1].get("role")
                                == "assistant"
                            ):
                                self.historical_messages[-1]["content"] += action_value
                            else:
                                self.historical_messages.append(
                                    ChatCompletionAssistantMessageParam(
                                        role="assistant",
                                        content=action_value,
                                    )
                                )
                        self.action_io.seek(0)
                        self.action_io.truncate(0)

                # 遇到JSON开始符号，压入栈
                elif self.cache[-1] in (CURLY_BRACE_OPEN, BRACKET_OPEN, CODE_BLOCK):
                    self.open_symbols.append(self.cache[-1])
                    self.cache.clear()
                    self.cache.append(s[-1])

                # 处理动作标签内的'<'字符
                elif self.cache[-1] == "<":
                    self.cache.clear()
                    self.cache.append("<")

                # 动作标签内的普通内容，直接输出
                else:
                    self.cache.clear()

            # 处理代码块内的内容（在动作标签内）
            elif self.open_symbols[-1] == CODE_BLOCK:
                # 检查是否遇到动作结束标签（且动作标签在栈中）
                if (
                    match_s.endswith(self.symbols_open_close_map.get(self.action_open))
                    and len(self.open_symbols) >= 2
                    and self.open_symbols[-2] == self.action_open
                ):
                    self.cache.clear()
                    # 解析并输出动作
                    json_str = re.sub(
                        r"^[a-zA-Z0-9\s]+\n",
                        "",
                        s[
                            : -len(self.symbols_open_close_map.get(self.action_open))
                        ].strip("\n`\r\t "),
                        flags=re.MULTILINE,
                    )
                    # 移除代码块和动作标签从栈
                    if len(self.open_symbols) >= 2:
                        self.open_symbols.pop()  # 移除代码块
                        self.open_symbols.pop()  # 移除动作标签
                    results.extend(self.parse_action(json_str))

                # 遇到代码块结束符号
                elif self.cache[-1] == CODE_BLOCK:
                    if (
                        len(self.open_symbols) >= 2
                        and self.open_symbols[-2] == self.action_open
                    ):
                        self.cache.clear()
                        # 解析并输出动作
                        json_str = re.sub(
                            r"^[a-zA-Z0-9\s]+\n",
                            "",
                            s[
                                : -len(
                                    self.symbols_open_close_map.get(self.action_open)
                                )
                            ].strip("\n`\r\t "),
                            flags=re.MULTILINE,
                        )
                        results.extend(self.parse_action(json_str))
                    # 从栈中移除代码块
                    if self.open_symbols:
                        self.open_symbols.pop()

            # 处理JSON对象内的内容（在动作标签内）
            elif self.open_symbols[-1] == CURLY_BRACE_OPEN:
                # 检查是否遇到动作结束标签（且动作标签在栈中）
                if (
                    match_s.endswith(self.symbols_open_close_map.get(self.action_open))
                    and len(self.open_symbols) >= 2
                    and self.open_symbols[-2] == self.action_open
                ):
                    self.cache.clear()
                    # 移除JSON对象和动作标签从栈
                    if len(self.open_symbols) >= 2:
                        self.open_symbols.pop()  # 移除JSON对象
                        self.open_symbols.pop()  # 移除动作标签
                    # 解析并输出动作
                    results.extend(
                        self.parse_action(
                            s[
                                : -len(
                                    self.symbols_open_close_map.get(self.action_open)
                                )
                            ],
                        )
                    )

                # 遇到JSON对象结束符号
                elif self.cache[-1] == CURLY_BRACE_CLOSE:
                    if (
                        len(self.open_symbols) >= 2
                        and self.open_symbols[-2] == self.action_open
                    ):
                        self.cache.clear()
                        # 解析并输出动作
                        results.extend(self.parse_action(s))
                    # 从栈中移除JSON对象
                    if self.open_symbols:
                        self.open_symbols.pop()

                # 遇到嵌套的JSON开始符号，压入栈
                elif self.cache[-1] in (
                    CURLY_BRACE_OPEN,
                    BRACKET_OPEN,
                    DOUBLE_QUOTE,
                ):
                    self.open_symbols.append(self.cache[-1])

            # 处理JSON数组内的内容（在动作标签内）
            elif self.open_symbols[-1] == BRACKET_OPEN:
                if (
                    match_s.endswith(self.symbols_open_close_map.get(self.action_open))
                    and len(self.open_symbols) >= 2
                    and self.open_symbols[-2] == self.action_open
                ):
                    self.cache.clear()
                    # 移除JSON数组和动作标签从栈
                    if len(self.open_symbols) >= 2:
                        self.open_symbols.pop()  # 移除JSON数组
                        self.open_symbols.pop()  # 移除动作标签
                    # 解析并输出动作
                    results.extend(
                        self.parse_action(
                            s[
                                : -len(
                                    self.symbols_open_close_map.get(self.action_open)
                                )
                            ],
                        )
                    )
                elif self.cache[-1] == BRACKET_CLOSE:
                    if (
                        len(self.open_symbols) >= 2
                        and self.open_symbols[-2] == self.action_open
                    ):
                        self.cache.clear()
                        # 解析并输出动作
                        results.extend(self.parse_action(s))
                    # 从栈中移除JSON数组
                    if self.open_symbols:
                        self.open_symbols.pop()
                elif self.cache[-1] in (
                    CURLY_BRACE_OPEN,
                    BRACKET_OPEN,
                    DOUBLE_QUOTE,
                ):
                    self.open_symbols.append(self.cache[-1])
            # 处理双引号闭合
            elif self.open_symbols[-1] == DOUBLE_QUOTE:
                if self.cache[-1] == DOUBLE_QUOTE and s[-2] != "\\":
                    self.open_symbols.pop()

        # Clear cache at the end of processing
        # 不能清空cache
        # self.cache.clear()
        return results

    async def run(
        self, input_chunks: AsyncGenerator[InputChunk, None] | Dict[str, Any]
    ) -> AsyncGenerator[InputChunk, None]:
        input_dict = await self.parse_input(input_chunks)
        action_close = self.symbols_open_close_map.get(self.action_open)
        for round_idx in range(self.max_loop):
            self.round_idx = round_idx
            actions: List[Function] = []
            pre_chunk = InputChunk(state="DOING", content="")
            async for chunk in super(ChatLLM, self).run(input_dict):
                for action in self.parse_labels(chunk):
                    if (
                        action.get("name") != "answer"
                        and action.get("name") != "request_user_choice"
                    ):
                        actions.append(action)
                        continue
                    if len(actions) > 0:
                        if (
                            len(self.open_symbols) > 0
                            and self.open_symbols[-1] != self.action_open
                        ):
                            break
                        continue
                    action_value = self.action_io.getvalue().strip()
                    if self.action_open in self.open_symbols and action_value:
                        if action_value[-1] not in action_close:
                            action_value += action_close
                        elif action_value[-1] == "<":
                            action_value += action_value[1:]
                        else:
                            action_value += action_close[
                                action_close.index(action_value[-2:]) + 2 :
                            ]
                        if (
                            self.historical_messages
                            and self.historical_messages[-1].get("role") == "assistant"
                        ):
                            self.historical_messages[-1]["content"] += action_value
                        else:
                            self.historical_messages.append(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant",
                                    content=action_value,
                                )
                            )
                        self.action_io.seek(0)
                        self.action_io.truncate(0)
                        while self.open_symbols.pop() != self.action_open:
                            pass
                    chunk.content = pre_chunk.content + chunk.content
                    if chunk.content.rfind(action_close) != -1:
                        chunk.content = chunk.content[
                            : chunk.content.rfind(action_close) + len(action_close)
                        ]
                    elif chunk.content[-1] not in action_close:
                        chunk.content += action_close
                    elif chunk.content[-1] == "<":
                        chunk.content += action_value[1:]
                    else:
                        chunk.content += action_close[
                            action_close.index(chunk.content[-2:]) + 2 :
                        ]
                    yield chunk
                    logger.info("open symbols %s", self.open_symbols)
                    self.open_symbols.clear()
                    logger.info("cache %s", self.open_symbols)
                    self.cache.clear()
                    if self.enable_chat_history:
                        await self.save_messages()
                    else:
                        self.historical_messages = []
                        self.round_idx = 0
                        self.new_messages_idx = 0
                    if action.get("name") == "answer":
                        yield InputChunk(
                            state="END", content=json.loads(action.get("arguments"))
                        )
                    self.cache = []
                    self.open_symbols = []
                    return
                if self.open_symbols and self.open_symbols[-1] == self.observation_open:
                    chunk.content = pre_chunk.content + chunk.content
                    chunk.content = re.sub(
                        rf"{re.escape(self.observation_open)}.*$",
                        "",
                        chunk.content,
                        flags=re.M,
                    )
                    yield chunk
                    pre_chunk.content = ""
                    break
                yield pre_chunk
                pre_chunk = chunk
            if pre_chunk.content:
                if pre_chunk.state == "END":
                    pre_chunk.state = "DOING"
                yield pre_chunk
            self.open_symbols.clear()
            self.cache.clear()
            logger.info("actions %s", actions)
            if actions:
                thought_value = self.thought_io.getvalue().strip()
                if thought_value:
                    thought_close = self.symbols_open_close_map.get(self.thought_open)
                    if thought_value[-1] not in thought_close:
                        thought_value += thought_close
                    elif thought_value[-1] == "<":
                        thought_value += thought_value[1:]
                    else:
                        thought_value += thought_close[
                            thought_close.index(thought_value[-2:]) + 2 :
                        ]
                    if (
                        self.historical_messages
                        and self.historical_messages[-1].get("role") == "assistant"
                    ):
                        self.historical_messages[-1]["content"] += thought_value
                    else:
                        self.historical_messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=thought_value,
                            )
                        )
                    self.thought_io.seek(0)
                    self.thought_io.truncate(0)
                    while self.thought_open in self.open_symbols:
                        self.open_symbols.pop()
                action_value = self.action_io.getvalue().strip()
                if action_value:
                    action_close = self.symbols_open_close_map.get(self.action_open)
                    if action_value[-1] not in action_close:
                        action_value += action_close
                    elif action_value[-1] == "<":
                        action_value += action_value[1:]
                    else:
                        action_value += action_close[
                            action_close.index(action_value[-2:]) + 2 :
                        ]
                    if (
                        self.historical_messages
                        and self.historical_messages[-1].get("role") == "assistant"
                    ):
                        self.historical_messages[-1]["content"] += action_value
                    else:
                        self.historical_messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=action_value,
                            )
                        )
                    self.action_io.seek(0)
                    self.action_io.truncate(0)
                    while self.action_open in self.open_symbols:
                        self.open_symbols.pop()
                if (
                    self.historical_messages
                    and self.historical_messages[-1].get("role") == "assistant"
                ):
                    assistant_idx = len(self.historical_messages) - 1
                    self.historical_messages[assistant_idx]["tool_calls"] = []
                    for action in actions:
                        tool_response = {}
                        try:
                            if self.hook_tool_call:
                                tool_response = await self.hook_tool_call(self, action)
                            else:
                                tool_response = await self.call_tool(action)
                        except Exception as e:
                            tool_response["err_msg"] = str(e)
                        tool_call_id = str(uuid4())
                        tool_calls = self.historical_messages[assistant_idx].get(
                            "tool_calls", []
                        )
                        tool_calls.append(
                            ChatCompletionMessageFunctionToolCallParam(
                                id=tool_call_id, function=action, type="function"
                            )
                        )
                        self.historical_messages[assistant_idx][
                            "tool_calls"
                        ] = tool_calls
                        self.historical_messages.append(
                            ChatCompletionToolMessageParam(
                                role="tool",
                                content=f"{self.observation_open}{json.dumps(tool_response, ensure_ascii=False, separators=(',', ':'))}{self.symbols_open_close_map[self.observation_open]}",
                                tool_call_id=tool_call_id,
                            )
                        )
            if self.historical_messages and self.historical_messages[-1].get(
                "role"
            ) not in ["user", "tool"]:
                self.historical_messages.append(
                    ChatCompletionUserMessageParam(role="user", content="继续")
                )
        self.historical_messages = []
        self.round_idx = 0
        self.new_messages_idx = 0
        self.cache = []
        self.open_symbols = []
