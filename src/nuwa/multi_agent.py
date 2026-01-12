"""Nuwa 群组代理模块，支持多智能体场景和角色管理。

本模块提供 GroupAgent 类，继承自 ReActAgent，扩展了角色黑名单和自定义系统提示功能。
适用于需要多个智能体协作或角色隔离的场景。
设计规范：继承层次（C-007）、模块化（C-001）、接口设计（C-008）。
"""

import json
from uuid import uuid4

from .react_agent import ReasoningActingAgent, TypeAdapter, ToolEntity
from typing import List


class MultiRoleAgent(ReasoningActingAgent):
    """多角色代理，扩展 ReasoningActingAgent 以支持多角色场景。

    提供角色黑名单机制，防止角色间的冲突或重复。
    通过重写 parse_system_prompt 方法定制系统提示，注入工具和指令信息。
    遵循继承原则（C-007），扩展父类功能。

    Attributes:
        excluded_roles: 角色黑名单列表，禁止当前角色与其他黑名单角色交互。
        roles: 当前群组中所有角色的列表（保留字段，未来扩展用）。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        role_name: str,
        role_prompt: str,
        roles: List[str] = [],
        excluded_roles: List[str] = [],
        session_id=str(uuid4()),
        stream=False,
        messages_manager=None,
        tools_manager=None,
        mcp: str | None = None,
        mcp_timeout: int = 300,
        temperature=0.6,
        extra_body=None,
        max_loop=3,
        with_time: bool = False,
        stop=None,
        base_url="https://api.openai.com/v1",
        enable_chat_history=True,
    ):
        """初始化多角色代理。

        将当前角色名加入黑名单（避免自我冲突），并存储角色列表。
        其余参数传递给父类 ReasoningActingAgent 初始化。

        Args:
            model: 模型名称（如 "gpt-4"）。
            api_key: OpenAI API 密钥。
            role_name: 当前角色名称。
            role_prompt: 角色特定的系统提示。
            roles: 群组中所有角色列表，默认为空。
            excluded_roles: 角色黑名单列表，默认为空。
            session_id: 会话 ID，默认为 ...（占位符）。
            stream: 是否启用流式响应。
            messages_manager: 消息管理器实例。
            tools_manager: 工具管理器实例。
            mcp: MCP 服务器地址。
            mcp_timeout: MCP 超时时间（秒）。
            temperature: 温度参数。
            extra_body: 额外的请求体参数。
            max_loop: 最大推理循环次数。
            with_time: 是否包含时间信息。
            stop: 停止词列表。
            base_url: API 基础 URL。
            enable_chat_history: 是否启用聊天历史。

        设计规范：参数类型提示（D-014）、默认参数合理（C-011）。
        """
        self.excluded_roles = excluded_roles
        if role_name not in self.excluded_roles:
            self.excluded_roles.append(role_name)
        self.roles = roles
        super().__init__(
            model=model,
            system_prompt=role_prompt,
            api_key=api_key,
            session_id=session_id,
            stream=stream,
            messages_manager=messages_manager,
            tools_manager=tools_manager,
            mcp=mcp,
            mcp_timeout=mcp_timeout,
            temperature=temperature,
            extra_body=extra_body,
            max_loop=max_loop,
            with_time=with_time,
            stop=stop,
            base_url=base_url,
            enable_chat_history=enable_chat_history,
        )

    def parse_system_prompt(self, instruction: str):
        """解析系统提示，注入工具和指令信息。

        重写父类方法，生成包含工具列表和 JSON Schema 的详细提示。
        遵循模板方法模式（P-005），定制提示生成逻辑。

        Args:
            instruction: 用户指令字符串。

        Returns:
            格式化后的系统提示字符串。
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
