import json

from .re_act import ReActAgent, TypeAdapter, ToolEntity
from typing import List


class GroupAgent(ReActAgent):
    def __init__(
        self,
        model: str,
        api_key: str,
        role_name: str,
        role_prompt: str,
        roles: List[str] = [],
        role_blacklist: List[str] = [],
        session_id=...,
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
        self.role_blacklist = role_blacklist
        if role_name not in self.role_blacklist:
            self.role_blacklist.append(role_name)
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
