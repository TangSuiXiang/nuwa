import asyncio

from .tool import Tool, ToolEntity, ToolObjectParameter, ToolParameter
from .re_act import ReActAgent
from typing import Literal


async def get_alarm_tool(agent: ReActAgent) -> Tool:

    async def set_alarm(time: str, remindee: Literal["oneself", "user"], reminder: str):
        pass

    return Tool(
        func=set_alarm,
        entity=ToolEntity(
            name="set_alarm",
            description="设置闹钟提醒",
            parameters=ToolObjectParameter(
                type="object",
                properties={
                    "time": ToolParameter(
                        type="string",
                        description="闹钟或提醒的时间，标准ISO时间格式字符串，如：2026-01-01T20:25:56.847307",
                    ),
                    "remindee": ToolParameter(
                        type="string",
                        description="被提醒的人，可选oneself（自己）或user（用户）",
                        emum=["oneself", "user"],
                    ),
                    "reminder": ToolParameter(type="string", description="备忘信息"),
                },
            ),
        ),
    )
