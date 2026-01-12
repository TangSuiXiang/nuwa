"""Nuwa ReAct 层模块，实现推理与行动（ReAct）模式。

本模块提供 ReActAgent 类，继承自 ChatLLM，扩展了结构化输出解析、
多轮推理循环、工具调用和观察标签处理。遵循责任链模式（P-006）和
DAG 模式（P-007），通过符号状态机实现流式输出的实时解析。
设计规范：模块化（C-001）、复杂算法注释（D-012）、错误处理（C-012）。
"""

import re
import io
import json
import logging

from uuid import uuid4
from .chat import ConversationAgent
from .tool import (
    ToolInvocation,
    ToolEntity,
    ToolParameter,
    ToolRegistry,
    ToolObjectParameter,
    ToolArrayParameter,
)
from .base import ConversationStorage, StreamChunk
from pydantic import TypeAdapter
from typing import (
    List,
    Optional,
    Dict,
    AsyncGenerator,
    Any,
    Callable,
    Awaitable,
    Literal,
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
)

from json_repair import loads, JSONReturnType

logger = logging.getLogger()


# 定义各种符号常量，用于 JSON 解析和代码块处理
CODE_BLOCK = "`"  # 代码块符号（反引号）
DOUBLE_QUOTE = '"'  # 双引号
SINGLE_QUOTE = "'"  # 单引号
CURLY_BRACE_OPEN = "{"  # 左花括号
BRACKET_OPEN = "["  # 左方括号
BRACKET_CLOSE = "]"  # 右方括号
CURLY_BRACE_CLOSE = "}"  # 右花括号

# 符号开闭映射表，用于匹配对应的闭合符号
symbol_mappings = {
    CURLY_BRACE_OPEN: CURLY_BRACE_CLOSE,  # 花括号映射
    SINGLE_QUOTE: SINGLE_QUOTE,  # 单引号自映射
    DOUBLE_QUOTE: DOUBLE_QUOTE,  # 双引号自映射
    CODE_BLOCK: CODE_BLOCK,  # 反引号自映射
    BRACKET_OPEN: BRACKET_CLOSE,  # 方括号映射
}


class ReasoningActingAgent(ConversationAgent):
    """推理与行动代理，实现 ReAct（Reasoning + Acting）模式。

    继承自 ConversationAgent，扩展了结构化标签解析（<Q>, <T>, <A>, <O>）和多轮推理循环。
    遵循责任链模式（P-006），将 LLM 输出解析、工具调用、历史管理串联成处理链。
    支持 DAG 模式（P-007）的节点依赖，同时集成观察者模式（P-004）的工具调用钩子。
    设计规范：类名大驼峰（D-007）、单一职责（C-006）、类型提示（D-014）。

    Attributes:
        enable_chat_history: 是否启用对话历史持久化。
        _new_messages_idx: 标识是否需要重置新消息索引。
        max_loop: 最大推理循环次数，防止无限循环。
        _character_buffer: 字符缓存，用于累积解析中的临时字符。
        _open_tags_stack: 打开的符号栈，用于处理嵌套标签和 JSON 结构。
        thought_open: 思考开始标签 "<T>"。
        action_open: 动作开始标签 "<A>"。
        question_open: 问题开始标签 "<Q>"。
        observation_open: 观察开始标签 "<O>"。
        _reasoning_buffer: 思考内容缓冲区（StringIO）。
        thought_io: 想法内容缓冲区（StringIO）。
        action_io: 动作内容缓冲区（StringIO）。
        question_io: 问题内容缓冲区（StringIO）。
        round_idx: 当前推理轮次索引。
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: str,
        session_id: str = str(uuid4()),
        stream: bool = False,
        messages_manager: Optional[ConversationStorage] = None,
        tools_manager: Optional[ToolRegistry] = None,
        mcp: Optional[str] = None,
        mcp_timeout: int = 300,
        temperature: float = 0.6,
        extra_body: Optional[Dict[str, Any]] = None,
        max_loop: int = 3,
        with_time: bool = False,
        with_others: str = "",
        stop: Optional[List[str]] = None,
        base_url: str = "https://api.openai.com/v1",
        enable_chat_history: bool = True,
        enable_selection: bool = False,
        hook_tool_call: Optional[
            Callable[[ConversationAgent, ToolInvocation], Awaitable[Any]]
        ] = None,
        answer_format: Literal["normal", "markdown"] = "normal",
    ):
        """初始化 ReasoningActingAgent。

        Args:
            model: LLM 模型名称。
            system_prompt: 系统提示模板。
            api_key: API 密钥。
            session_id: 会话 ID，默认为自动生成的 UUID。
            stream: 是否启用流式输出，默认为 False。
            messages_manager: 消息管理器实例，默认为 None。
            tools_manager: 工具管理器实例，默认为 None。
            mcp: MCP 服务器地址，默认为 None。
            mcp_timeout: MCP 请求超时时间，默认为 300 秒。
            temperature: 生成温度，默认为 0.6。
            extra_body: 额外的 API 请求体参数，默认为空字典。
            max_loop: 最大推理循环次数，默认为 3。
            with_time: 是否在系统提示中加入当前时间，默认为 False。
            with_others: 其他自定义信息，默认为空字符串。
            stop: 停止词列表，默认为空列表。
            base_url: API 基础 URL，默认为 OpenAI 官方端点。
            enable_chat_history: 是否启用对话历史，默认为 True。
            enable_selection: 是否启用用户选择工具，默认为 False。
            hook_tool_call: 工具调用钩子函数，默认为 None。
            answer_format: 最终回答格式，可选 "normal" 或 "markdown"，默认为 "normal"。

        设计规范：参数类型提示（D-014）、默认参数合理（C-011）。
        """
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
        self._new_messages_idx = True
        self.max_loop = max_loop
        self._character_buffer = []
        self._open_tags_stack = []
        self.thought_open = "<T>"
        self.action_open = "<A>"
        self.question_open = "<Q>"
        self.observation_open = "<O>"
        # 扩展符号映射表，加入自定义标签
        self.symbol_mappings[self.thought_open] = "</T>"
        self.symbol_mappings[self.action_open] = "</A>"
        self.symbol_mappings[self.observation_open] = "</O>"
        self.symbol_mappings[self.question_open] = "</Q>"
        # 添加内置答案工具
        self.tool_names.append("answer")
        self.tools.append(
            ToolEntity(
                name="answer",
                parameters=ToolParameter(
                    type="string",
                    description=f"给用户的最终响应{'(Markdown格式)' if answer_format == 'markdown' else ''}",
                ),
            )
        )
        # 可选添加用户选择工具
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
                        required=["question", "options", "recommended_option"],
                    ),
                )
            )
        # 初始化缓冲区
        self._reasoning_buffer = io.StringIO()
        self.thought_io = io.StringIO()
        self.action_io = io.StringIO()
        self.question_io = io.StringIO()
        self.round_idx = 0

    def _extract_tool_calls(self, json_str: str) -> List[ToolInvocation]:
        """解析动作 JSON 字符串，提取工具调用信息。

        支持 Cohere 模型返回的列表格式，以及标准的字典格式。
        从动作字典中提取动作名称和输入参数，构造 Function 对象。
        遵循错误处理规范（C-012），解析失败时记录警告并返回空列表。

        Args:
            json_str: 包含动作信息的 JSON 字符串。

        Returns:
            解析出的 ToolInvocation 对象列表。
        """
        results = []
        try:
            actions: JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]] = (
                loads(json_str)
            )

            # Cohere 模型总是返回列表格式，统一转换为列表处理
            if isinstance(actions, dict):
                actions = [actions]
            if isinstance(actions, list) or isinstance(actions, tuple):
                for action in actions:
                    action_name = None  # 动作名称
                    action_input = None  # 动作输入参数
                    if not isinstance(action, dict):
                        continue
                    # 遍历动作字典，提取动作名称和输入参数
                    for key, value in action.items():
                        if not isinstance(key, str):
                            continue
                        # 查找输入参数键（包含 "input" 且不区分大小写）
                        if "input" in key.lower():
                            action_input = value
                        # 查找动作名称键（键为 "action" 或值在工具列表中）
                        elif "action" == key.strip().lower() or (
                            isinstance(value, str) and value.strip() in self.tool_names
                        ):
                            action_name = value

                    # 如果动作名称和输入参数都找到，创建 ToolInvocation 对象
                    if action_name is not None and action_input is not None:
                        results.append(
                            ToolInvocation(
                                name=action_name,
                                arguments=json.dumps(
                                    action_input,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            )
                        )

        except Exception as e:
            # 解析失败时记录异常，遵循日志记录规范（C-016）
            logger.warning("Failed to parse action '%s' with error: '%s'", json_str, e)

        return results

    async def _prepare_conversation_messages(self, input_dict=None):
        """生成发送给 LLM 的消息列表，适配 ReAct 格式。

        重写父类方法，将系统提示替换为 ReAct 专用的结构化提示。
        根据当前轮次和历史消息状态调整消息列表，确保格式正确。
        遵循模板方法模式（P-005），扩展了父类行为。

        Args:
            input_dict: 输入字典，包含用户输入等字段。

        Returns:
            符合 ReAct 格式的消息列表。
        """
        messages = await super()._prepare_conversation_messages(input_dict)
        instruction = messages[0].get("content", "")
        if not isinstance(instruction, str):
            raise
        messages[0] = ChatCompletionSystemMessageParam(
            role="system",
            content=self.parse_system_prompt(instruction=instruction),
        )
        # 如果当前轮次大于 0 且最后一条消息是用户消息，则删除它（避免重复）
        if self.round_idx > 0 and self.historical_messages[-1].get("role") == "user":
            del self.historical_messages[-1]
            del messages[-1]
        # 如果禁用对话历史，只保留系统消息和当前用户消息
        if not self.enable_chat_history:
            messages = [messages[0], messages[-1]]
        if self._new_messages_idx:
            self.new_messages_idx = max(len(self.historical_messages) - 1, 0)
            self._new_messages_idx = False
        return messages

    def parse_system_prompt(self, instruction: str) -> str:
        """解析并生成 ReAct 专用的系统提示。

        将原始指令、可用工具列表和工具调用 JSON Schema 嵌入到固定模板中，
        指导 LLM 按照 <Q><T><A><O> 格式进行推理和行动。
        设计规范：字符串格式化安全、避免注入。

        Args:
            instruction: 原始系统提示内容。

        Returns:
            格式化后的 ReAct 系统提示字符串。
        """
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

    def _parse_stream_tags(self, chunk: StreamChunk) -> List[ToolInvocation]:
        """解析流式输出中的标签，提取动作和文本内容。

        核心算法：逐个字符处理 LLM 输出，使用状态机识别标签边界和嵌套结构。
        支持标签：<Q>, <T>, <A>, <O> 以及 JSON 符号（{}, [], "", `）。
        遵循复杂算法注释规范（D-012），详细解释状态转换逻辑。
        设计模式：有限状态机（FSM）实现流式解析。

        Args:
            chunk: 输入数据块，包含字符串内容。

        Returns:
            解析出的 ToolInvocation 对象列表（通常是动作标签内的工具调用）。
        """
        if not isinstance(chunk.content, str):
            return []

        results = []
        # 逐个字符处理响应内容
        for c in chunk.content:
            self._character_buffer.append(c)  # 添加到缓存
            s = "".join(self._character_buffer)  # 构建当前字符串
            match_s = s.replace(" ", "")  # 移除空格用于匹配（忽略大小写）
            # 根据当前打开的符号决定写入哪个缓冲区
            if self.action_open in self._open_tags_stack:
                self.action_io.write(c)
            elif self.thought_open in self._open_tags_stack:
                self.thought_io.write(c)
            elif self.think_open in self._open_tags_stack:
                self._reasoning_buffer.write(c)
            elif self.question_open in self._open_tags_stack:
                self.question_io.write(c)

            action_close = self.symbol_mappings.get(self.action_open, "</A>")
            # 如果没有打开的符号，尝试匹配开始标签
            if len(self._open_tags_stack) == 0:
                # 匹配思考标签（来自父类的 think_open）
                if re.match(re.escape(match_s), self.think_open, re.I):
                    if match_s == self.think_open:
                        self._character_buffer.clear()  # 清空缓存
                        self._open_tags_stack.append(
                            self.think_open
                        )  # 标记思考标签已打开
                        self._reasoning_buffer.write(s)

                # 匹配想法标签
                elif re.match(re.escape(match_s), self.thought_open, re.I):
                    if match_s == self.thought_open:
                        self._character_buffer.clear()
                        self._open_tags_stack.append(self.thought_open)
                        self.thought_io.write(s)

                # 匹配动作标签
                elif re.match(re.escape(match_s), self.action_open, re.I):
                    if match_s == self.action_open:
                        self._character_buffer.clear()
                        self._open_tags_stack.append(self.action_open)
                        self.action_io.write(s)

                # 匹配观察标签
                elif re.match(re.escape(match_s), self.observation_open, re.I):
                    if match_s == self.observation_open:
                        self._character_buffer.clear()
                        self._open_tags_stack.append(self.observation_open)

                # 匹配问题标签
                elif re.match(re.escape(match_s), self.question_open, re.I):
                    if match_s == self.question_open:
                        self._character_buffer.clear()
                        self._open_tags_stack.append(self.question_open)
                        self.question_io.write(s)

                # 处理单独的 '<' 字符（可能是标签开始的一部分）
                elif self._character_buffer[-1] == "<":
                    self._character_buffer.clear()
                    self._character_buffer.append("<")

                # 普通字符，直接丢弃并清空缓存（不属于任何标签）
                else:
                    self._character_buffer.clear()

            # 处理思考标签内的内容
            elif self._open_tags_stack[-1] == self.think_open:
                # 匹配思考结束标签
                if re.match(
                    re.escape(match_s),
                    self.symbol_mappings.get(self.think_open, "</think>"),
                    re.I,
                ):
                    if match_s == self.symbol_mappings.get(self.think_open):
                        self._character_buffer.clear()
                        self._open_tags_stack.pop()  # 移除思考标签

                # 处理标签内的 '<' 字符
                elif self._character_buffer[-1] == "<":
                    self._character_buffer.clear()
                    self._character_buffer.append("<")

                # 思考标签内的普通内容，直接输出
                else:
                    self._character_buffer.clear()

            # 处理想法标签内的内容（与思考标签类似）
            elif self._open_tags_stack[-1] == self.thought_open:
                if re.match(
                    re.escape(match_s),
                    self.symbol_mappings.get(self.thought_open, "</T>"),
                    re.I,
                ):
                    if match_s == self.symbol_mappings.get(self.thought_open):
                        self._character_buffer.clear()
                        self._open_tags_stack.pop()
                        thought_value = self.thought_io.getvalue().strip()
                        if thought_value:
                            # 将想法内容添加到历史消息中
                            if (
                                self.historical_messages
                                and self.historical_messages[-1].get("role")
                                == "assistant"
                            ):
                                content = self.historical_messages[-1].get(
                                    "content", ""
                                )
                                if isinstance(content, str):
                                    self.historical_messages[-1]["content"] = (
                                        content + thought_value
                                    )
                            else:
                                self.historical_messages.append(
                                    ChatCompletionAssistantMessageParam(
                                        role="assistant",
                                        content=thought_value,
                                    )
                                )
                        self.thought_io.seek(0)
                        self.thought_io.truncate(0)

                elif self._character_buffer[-1] == "<":
                    self._character_buffer.clear()
                    self._character_buffer.append("<")

                else:
                    self._character_buffer.clear()

            # 处理问题标签内的内容（与思考标签类似）
            elif self._open_tags_stack[-1] == self.question_open:
                if re.match(
                    re.escape(match_s),
                    self.symbol_mappings.get(self.question_open, "</Q>"),
                    re.I,
                ):
                    if match_s == self.symbol_mappings.get(self.question_open):
                        self._character_buffer.clear()
                        self._open_tags_stack.pop()
                        if (
                            self.historical_messages
                            and self.historical_messages[-1].get("role") == "assistant"
                        ):
                            content = self.historical_messages[-1].get("content", "")
                            if isinstance(content, str):
                                self.historical_messages[-1]["content"] = (
                                    content + self.question_io.getvalue()
                                )
                        else:
                            self.historical_messages.append(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant",
                                    content=self.question_io.getvalue().strip(),
                                )
                            )
                        self.question_io.seek(0)
                        self.question_io.truncate(0)

                elif self._character_buffer[-1] == "<":
                    self._character_buffer.clear()
                    self._character_buffer.append("<")

                else:
                    self._character_buffer.clear()

            # 处理观察标签内的内容（与思考标签类似）
            elif self._open_tags_stack[-1] == self.observation_open:
                if re.match(
                    re.escape(match_s),
                    self.symbol_mappings.get(self.observation_open, "</O>"),
                    re.I,
                ):
                    if match_s == self.symbol_mappings.get(self.observation_open):
                        self._character_buffer.clear()
                        self._open_tags_stack.pop()

                elif self._character_buffer[-1] == "<":
                    self._character_buffer.clear()
                    self._character_buffer.append("<")

                else:
                    self._character_buffer.clear()

            # 处理动作标签内的内容（可能包含 JSON）
            elif self._open_tags_stack[-1] == self.action_open:
                # 匹配动作结束标签
                if re.match(
                    re.escape(match_s),
                    self.symbol_mappings.get(self.action_open, "</A>"),
                    re.I,
                ):
                    if match_s == self.symbol_mappings.get(self.action_open):
                        self._character_buffer.clear()
                        self._open_tags_stack.pop()  # 移除动作标签
                        action_value = self.action_io.getvalue().strip()
                        if action_value:
                            if (
                                self.historical_messages
                                and self.historical_messages[-1].get("role")
                                == "assistant"
                            ):
                                content = self.historical_messages[-1].get("content")
                                if isinstance(content, str):
                                    self.historical_messages[-1]["content"] = (
                                        content + action_value
                                    )
                            else:
                                self.historical_messages.append(
                                    ChatCompletionAssistantMessageParam(
                                        role="assistant",
                                        content=action_value,
                                    )
                                )
                        self.action_io.seek(0)
                        self.action_io.truncate(0)

                # 遇到 JSON 开始符号，压入栈
                elif self._character_buffer[-1] in (
                    CURLY_BRACE_OPEN,
                    BRACKET_OPEN,
                    CODE_BLOCK,
                ):
                    self._open_tags_stack.append(self._character_buffer[-1])
                    self._character_buffer.clear()
                    self._character_buffer.append(s[-1])

                # 处理动作标签内的 '<' 字符
                elif self._character_buffer[-1] == "<":
                    self._character_buffer.clear()
                    self._character_buffer.append("<")

                # 动作标签内的普通内容，直接输出
                else:
                    self._character_buffer.clear()

            # 处理代码块内的内容（在动作标签内）
            elif self._open_tags_stack[-1] == CODE_BLOCK:
                # 检查是否遇到动作结束标签（且动作标签在栈中）
                if (
                    match_s.endswith(action_close)
                    and len(self._open_tags_stack) >= 2
                    and self._open_tags_stack[-2] == self.action_open
                ):
                    self._character_buffer.clear()
                    # 解析并输出动作
                    json_str = re.sub(
                        r"^[a-zA-Z0-9\s]+\n",
                        "",
                        s[: -len(action_close)].strip("\n`\r\t "),
                        flags=re.MULTILINE,
                    )
                    # 移除代码块和动作标签从栈
                    if len(self._open_tags_stack) >= 2:
                        self._open_tags_stack.pop()  # 移除代码块
                        self._open_tags_stack.pop()  # 移除动作标签
                    results.extend(self._extract_tool_calls(json_str))

                # 遇到代码块结束符号
                elif self._character_buffer[-1] == CODE_BLOCK:
                    if (
                        len(self._open_tags_stack) >= 2
                        and self._open_tags_stack[-2] == self.action_open
                    ):
                        self._character_buffer.clear()
                        # 解析并输出动作
                        json_str = re.sub(
                            r"^[a-zA-Z0-9\s]+\n",
                            "",
                            s[: -len(action_close)].strip("\n`\r\t "),
                            flags=re.MULTILINE,
                        )
                        results.extend(self._extract_tool_calls(json_str))
                    # 从栈中移除代码块
                    if self._open_tags_stack:
                        self._open_tags_stack.pop()

            # 处理 JSON 对象内的内容（在动作标签内）
            elif self._open_tags_stack[-1] == CURLY_BRACE_OPEN:
                # 检查是否遇到动作结束标签（且动作标签在栈中）
                if (
                    match_s.endswith(action_close)
                    and len(self._open_tags_stack) >= 2
                    and self._open_tags_stack[-2] == self.action_open
                ):
                    self._character_buffer.clear()
                    # 移除 JSON 对象和动作标签从栈
                    if len(self._open_tags_stack) >= 2:
                        self._open_tags_stack.pop()  # 移除 JSON 对象
                        self._open_tags_stack.pop()  # 移除动作标签
                    # 解析并输出动作
                    results.extend(self._extract_tool_calls(s[: -len(action_close)]))

                # 遇到 JSON 对象结束符号
                elif self._character_buffer[-1] == CURLY_BRACE_CLOSE:
                    if (
                        len(self._open_tags_stack) >= 2
                        and self._open_tags_stack[-2] == self.action_open
                    ):
                        self._character_buffer.clear()
                        # 解析并输出动作
                        results.extend(self._extract_tool_calls(s))
                    # 从栈中移除 JSON 对象
                    if self._open_tags_stack:
                        self._open_tags_stack.pop()

                # 遇到嵌套的 JSON 开始符号，压入栈
                elif self._character_buffer[-1] in (
                    CURLY_BRACE_OPEN,
                    BRACKET_OPEN,
                    DOUBLE_QUOTE,
                ):
                    self._open_tags_stack.append(self._character_buffer[-1])

            # 处理 JSON 数组内的内容（在动作标签内）
            elif self._open_tags_stack[-1] == BRACKET_OPEN:
                if (
                    match_s.endswith(action_close)
                    and len(self._open_tags_stack) >= 2
                    and self._open_tags_stack[-2] == self.action_open
                ):
                    self._character_buffer.clear()
                    # 移除 JSON 数组和动作标签从栈
                    if len(self._open_tags_stack) >= 2:
                        self._open_tags_stack.pop()  # 移除 JSON 数组
                        self._open_tags_stack.pop()  # 移除动作标签
                    # 解析并输出动作
                    results.extend(self._extract_tool_calls(s[: -len(action_close)]))
                elif self._character_buffer[-1] == BRACKET_CLOSE:
                    if (
                        len(self._open_tags_stack) >= 2
                        and self._open_tags_stack[-2] == self.action_open
                    ):
                        self._character_buffer.clear()
                        # 解析并输出动作
                        results.extend(self._extract_tool_calls(s))
                    # 从栈中移除 JSON 数组
                    if self._open_tags_stack:
                        self._open_tags_stack.pop()
                elif self._character_buffer[-1] in (CURLY_BRACE_OPEN,):
                    self._open_tags_stack.append(self._character_buffer[-1])

            # 处理双引号闭合
            elif self._open_tags_stack[-1] == DOUBLE_QUOTE:
                if self._character_buffer[-1] == DOUBLE_QUOTE and s[-2] != "\\":
                    self._open_tags_stack.pop()

        # 处理结束后，缓存不清空（因为可能跨 chunk 累积）
        return results

    async def run(
        self, input_chunks: AsyncGenerator[StreamChunk, None] | Dict[str, Any]
    ) -> AsyncGenerator[StreamChunk, None]:
        """执行推理与行动循环，处理流式输出并调用工具。

        核心处理流程：初始化状态 → 多轮循环 → 解析标签 → 执行工具 → 生成观察。
        遵循责任链模式（P-006），将解析、工具调用、历史更新串联。
        支持最大循环次数限制，防止无限推理（O-013）。
        设计规范：复杂算法注释（D-012）、错误处理（C-012）、日志记录（C-016）。

        Args:
            input_chunks: 输入数据，可以是异步生成器或字典。

        Yields:
            StreamChunk: 输出数据块，状态为 "DOING" 或 "END"。
        """
        self._new_messages_idx = True
        input_dict = await self._parse_input_data(input_chunks)
        action_close = self.symbol_mappings.get(self.action_open, "</A>")
        for round_idx in range(self.max_loop):
            self.round_idx = round_idx
            actions: List[ToolInvocation] = []
            pre_chunk = StreamChunk(state="DOING", content="")
            # 调用父类（ConversationAgent）的 run 方法获取 LLM 输出流
            async for chunk in super(ConversationAgent, self).run(input_dict):
                # 解析当前 chunk 中的标签，提取可能的动作
                for action in self._parse_stream_tags(chunk):
                    # 过滤掉答案工具和用户选择工具（它们需要特殊处理）
                    if (
                        action.get("name") != "answer"
                        and action.get("name") != "request_user_choice"
                    ):
                        actions.append(action)
                        continue
                    # 如果已经有其他动作待处理，跳过本次解析
                    if len(actions) > 0:
                        if (
                            len(self._open_tags_stack) > 0
                            and self._open_tags_stack[-1] != self.action_open
                        ):
                            break
                        continue
                    # 处理答案工具或用户选择工具
                    action_value = self.action_io.getvalue().strip()
                    if self.action_open in self._open_tags_stack and action_value:
                        # 确保动作标签闭合
                        if action_value[-1] not in action_close:
                            action_value += action_close
                        elif action_value[-1] == "<":
                            action_value += action_value[1:]
                        else:
                            action_value += action_close[
                                action_close.index(action_value[-2:]) + 2 :
                            ]
                        # 将动作内容添加到历史消息
                        if (
                            self.historical_messages
                            and self.historical_messages[-1].get("role") == "assistant"
                        ):
                            content = self.historical_messages[-1].get("content")
                            if isinstance(content, str):
                                self.historical_messages[-1]["content"] = (
                                    content + action_value
                                )
                        else:
                            self.historical_messages.append(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant",
                                    content=action_value,
                                )
                            )
                        self.action_io.seek(0)
                        self.action_io.truncate(0)
                        while self._open_tags_stack.pop() != self.action_open:
                            pass
                    # 合并之前的 chunk 内容并确保动作标签闭合
                    if not isinstance(pre_chunk.content, str) or not isinstance(
                        chunk.content, str
                    ):
                        raise
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
                    logger.debug("open symbols %s", self._open_tags_stack)
                    self._open_tags_stack.clear()
                    logger.debug("cache %s", self._open_tags_stack)
                    self._character_buffer.clear()
                    # 保存消息或清空历史
                    if self.enable_chat_history:
                        await self.save_messages()
                    else:
                        self.historical_messages = []
                        self.round_idx = 0
                        self.new_messages_idx = 0
                    # 如果是答案工具，返回最终答案
                    if action.get("name") == "answer":
                        yield StreamChunk(
                            state="END", content=json.loads(action.get("arguments"))
                        )
                    self._character_buffer = []
                    self._open_tags_stack = []
                    return
                # 如果当前处于观察标签内，提前跳出循环
                if (
                    self._open_tags_stack
                    and self._open_tags_stack[-1] == self.observation_open
                ):
                    if not isinstance(pre_chunk.content, str) or not isinstance(
                        chunk.content, str
                    ):
                        raise
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
                # 传递上一个 chunk 并更新当前 chunk
                yield pre_chunk
                pre_chunk = chunk
            # 处理剩余的前一个 chunk
            if pre_chunk.content:
                if pre_chunk.state == "END":
                    pre_chunk.state = "DOING"
                yield pre_chunk
            # 清空状态
            self._open_tags_stack.clear()
            self._character_buffer.clear()
            logger.debug("actions %s", actions)
            # 如果有解析出的动作，执行工具调用
            if actions:
                # 处理想法缓冲区内容
                thought_value = self.thought_io.getvalue().strip()
                if thought_value:
                    thought_close = self.symbol_mappings.get(self.thought_open, "</T>")
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
                        content = self.historical_messages[-1].get("content")
                        if isinstance(content, str):
                            self.historical_messages[-1]["content"] = (
                                content + thought_value
                            )
                    else:
                        self.historical_messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=thought_value,
                            )
                        )
                    self.thought_io.seek(0)
                    self.thought_io.truncate(0)
                    while self.thought_open in self._open_tags_stack:
                        self._open_tags_stack.pop()
                # 处理动作缓冲区内容
                action_value = self.action_io.getvalue().strip()
                if action_value:
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
                        content = self.historical_messages[-1].get("content")
                        if isinstance(content, str):
                            self.historical_messages[-1]["content"] = (
                                content + action_value
                            )
                    else:
                        self.historical_messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=action_value,
                            )
                        )
                    self.action_io.seek(0)
                    self.action_io.truncate(0)
                    while self.action_open in self._open_tags_stack:
                        self._open_tags_stack.pop()
                # 执行工具调用并添加观察消息
                if (
                    self.historical_messages
                    and self.historical_messages[-1].get("role") == "assistant"
                ):
                    assistant_idx = len(self.historical_messages) - 1
                    self.historical_messages[assistant_idx]["tool_calls"] = []  # type: ignore
                    for action in actions:
                        tool_response = {}
                        try:
                            # 优先使用钩子，否则调用默认工具
                            if self.hook_tool_call:
                                tool_response = await self.hook_tool_call(self, action)
                            else:
                                tool_response = await self.call_tool(action)
                        except Exception as e:
                            # 错误处理规范（C-012），记录错误信息
                            tool_response["err_msg"] = str(e)
                        tool_call_id = str(uuid4())
                        tool_calls = self.historical_messages[assistant_idx].get(
                            "tool_calls", []
                        )
                        if isinstance(tool_calls, list):
                            tool_calls.append(
                                ChatCompletionMessageFunctionToolCallParam(
                                    id=tool_call_id, function=action, type="function"
                                )
                            )
                        self.historical_messages[assistant_idx]["tool_calls"] = tool_calls  # type: ignore
                        self.historical_messages.append(
                            ChatCompletionToolMessageParam(
                                role="tool",
                                content=f"{self.observation_open}{json.dumps(tool_response, ensure_ascii=False, separators=(',', ':'))}{self.symbol_mappings[self.observation_open]}",
                                tool_call_id=tool_call_id,
                            )
                        )
            # 如果最后一条消息不是用户或工具消息，添加继续提示
            if self.historical_messages and self.historical_messages[-1].get(
                "role"
            ) not in ["user", "tool"]:
                self.historical_messages.append(
                    ChatCompletionUserMessageParam(role="user", content="继续")
                )
        # 循环结束，重置状态
        self.historical_messages = []
        self.round_idx = 0
        self.new_messages_idx = 0
        self._character_buffer = []
        self._open_tags_stack = []
