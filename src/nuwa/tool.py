"""Nuwa 工具层模块，管理外部函数集成和工具注册。

本模块提供工具管理器、工具实体定义和装饰器注册机制。
遵循装饰器模式（P-003），通过 @tools.tool() 装饰器将普通函数转换为可调用工具。
支持 MCP（Model Context Protocol）工具转换和自动参数推断。
设计规范：模块化（C-001）、类型提示（D-014）、错误处理（C-012）。
"""

import json
import typing
import logging
import asyncio

from .chat import ToolInvocation
from json_repair import loads
from pydantic import BaseModel
from typing import Union, Literal, Optional, Dict, List, Any, Callable
from mcp.types import Tool as MCPTool

logger = logging.getLogger()


class ToolParameter(BaseModel):
    """工具参数定义，描述单个参数的类型、描述和可选枚举值。

    遵循 Pydantic 模型规范，确保类型安全。
    设计规范：数据类（C-006）、类型提示（D-014）。

    Attributes:
        type: 参数类型，可选 "object", "string", "number", "boolean", "array"。
        description: 参数描述，可选。
        enum: 可选值枚举列表，可选。
    """
    type: Literal["object", "string", "number", "boolean", "array"]
    description: Optional[str] = None
    enum: Optional[List[str]] = None


class ToolArrayParameter(ToolParameter):
    """数组类型工具参数，包含元素类型定义。

    继承自 ToolParameter，增加 items 字段描述数组元素类型。
    设计规范：继承层次清晰（C-005）。
    """
    items: Union["ToolObjectParameter", "ToolArrayParameter", ToolParameter]


class ToolObjectParameter(ToolParameter):
    """对象类型工具参数，包含属性字典和必需字段列表。

    用于描述复杂嵌套对象参数。
    设计规范：嵌套结构合理、类型安全。

    Attributes:
        properties: 属性字典，键为属性名，值为对应的参数定义。
        required: 必需属性名列表。
    """
    properties: Dict[
        str, Union["ToolObjectParameter", ToolArrayParameter, ToolParameter]
    ] = {}
    required: List[str] = []


class ToolEntity(BaseModel):
    """工具实体，描述工具的元数据。

    包含工具名称、参数定义和描述，用于构造 LLM 工具调用规范。
    设计规范：Pydantic 模型（C-006）、序列化友好。

    Attributes:
        name: 工具名称。
        parameters: 工具参数定义，可以是对象参数或简单参数。
        description: 工具描述，可选。
    """
    name: str
    parameters: Union[ToolObjectParameter, ToolParameter] = ToolObjectParameter(
        type="object", properties={}, required=[]
    )
    description: Optional[str] = None


class Tool:
    """工具包装类，关联函数和其元数据实体。

    设计规范：组合优于继承（C-007），将函数与元数据分离。

    Attributes:
        func: 实际的可调用函数（同步或异步）。
        entity: 工具元数据实体。
    """

    def __init__(self, func: Callable, entity: ToolEntity):
        """初始化工具包装。

        Args:
            func: 可调用函数。
            entity: 工具实体。
        """
        self.func = func
        self.entity = entity


def get_tool_entity(tool: MCPTool) -> ToolEntity:
    """将 MCP 工具转换为本地工具实体。

    从 MCP 工具的输入模式（JSON Schema）提取属性并映射为 ToolParameter。
    遵循错误处理规范（C-012），对异常类型进行降级处理（如 integer -> number）。

    Args:
        tool: MCP 工具对象。

    Returns:
        转换后的 ToolEntity。
    """
    properties = {}
    for k, v in tool.inputSchema.get("properties", {}).items():
        if not isinstance(v, dict):
            continue
        t = v.get("type")
        if not isinstance(t, str):
            # 处理 anyOf 类型（如 integer）
            for to in v.get("anyOf", []):
                if not isinstance(to, dict):
                    continue
                t = to.get("type")
                if isinstance(t, str):
                    if t == "integer":
                        t = "number"
                    break
        properties[k] = ToolParameter(type=t, description=v.get("description"))
    parameters = ToolObjectParameter(type="object", properties=properties)
    return ToolEntity(
        name=tool.name, parameters=parameters, description=tool.description
    )


class ToolRegistry:
    """工具注册表，负责工具的注册、查找和调用。

    遵循装饰器模式（P-003），提供 @tools.tool() 装饰器简化工具注册。
    支持单例模式（P-008）的全局实例，但本身不强制单例，允许多个实例。
    设计规范：类职责明确（C-006）、错误处理（C-012）、日志记录（C-016）。

    Attributes:
        _tools: 内部工具字典，键为工具名，值为 Tool 实例。
    """

    def __init__(self, init_tools: Dict[str, Tool] = {}):
        """初始化工具注册表。

        Args:
            init_tools: 初始工具字典，默认为空。
        """
        self._tools: Dict[str, Tool] = init_tools.copy()

    def tool(
        self,
        name: Optional[str] = None,
        parameters: Union[ToolObjectParameter, ToolParameter] = ToolObjectParameter(
            type="object", properties={}, required=[]
        ),
        description: Optional[str] = None,
    ):
        """装饰器工厂，用于注册函数工具。

        遵循装饰器模式（P-003），提供声明式工具注册。
        自动从函数注解推断参数类型，并合并到 parameters 中。
        设计规范：装饰器实现清晰、类型推断智能。

        Args:
            name: 工具名称，默认为函数名。
            parameters: 工具参数定义，默认为空对象。
            description: 工具描述，默认为函数文档字符串。

        Returns:
            装饰器函数。
        """

        def decorator(func: Callable):
            tool_name = name or func.__name__
            desc = description or func.__doc__

            # 处理函数注解以自动填充参数
            annotations = {}
            if isinstance(func.__annotations__, dict):
                annotations = func.__annotations__
            for k, v in annotations.items():
                # 跳过返回注解
                if k == "return":
                    continue

                # 跳过已定义的参数
                if k in parameters.properties:
                    continue

                item_type = v
                item_desc = None

                # 处理 Annotated 类型（Python 3.9+）
                if hasattr(v, "__metadata__"):
                    # 适用于新版本 Python 的 typing.Annotated
                    item_type = v.__args__[0] if hasattr(v, "__args__") else v
                    item_desc = (
                        getattr(v, "__metadata__", (None,))[0]
                        if hasattr(v, "__metadata__")
                        else None
                    )
                elif isinstance(v, typing._AnnotatedAlias):
                    # 适用于旧版本 Python
                    item_type = v.__args__[0]
                    item_desc = (v.__metadata__ or ("",))[0]

                # 将 Python 类型映射到 JSON Schema 类型
                if item_type in [int, float]:
                    parameters.properties[k] = ToolParameter(
                        type="number", description=item_desc
                    )
                elif item_type in [bool]:
                    parameters.properties[k] = ToolParameter(
                        type="boolean", description=item_desc
                    )
                elif item_type in [dict]:
                    parameters.properties[k] = ToolObjectParameter(
                        type="object", description=item_desc
                    )
                elif item_type in [list]:
                    parameters.properties[k] = ToolArrayParameter(
                        type="array",
                        items=ToolParameter(type="string", description=item_desc),
                    )
                else:
                    parameters.properties[k] = ToolParameter(
                        type="string", description=item_desc
                    )

            self.add_tool(
                tool=Tool(
                    func=func,
                    entity=ToolEntity(
                        name=tool_name, parameters=parameters, description=desc
                    ),
                ),
            )
            logger.debug("Registered tool: %s", tool_name)
            return func

        return decorator

    def add_tool(self, tool: Tool):
        """添加工具到管理器。

        如果工具已存在则跳过，避免重复注册。
        设计规范：幂等操作、日志记录（C-016）。

        Args:
            tool: 要添加的 Tool 实例。
        """
        if not self.has_tool(tool_name=tool.entity.name):
            self._tools[tool.entity.name] = tool

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """根据名称获取工具实例。

        Args:
            tool_name: 工具名称。

        Returns:
            Tool 实例，如果未找到则返回 None。
        """
        return self._tools.get(tool_name)

    async def call_tool(self, func: ToolInvocation) -> Any:
        """调用工具函数。

        根据函数规范解析参数，调用对应的工具函数，并处理同步/异步返回值。
        遵循错误处理规范（C-012），工具不存在时抛出明确异常。

        Args:
            func: 函数调用规范，包含名称和参数字符串。

        Returns:
            工具执行结果。

        Raises:
            ValueError: 当工具不存在时。
        """
        logger.debug("call tool %s", func)
        tool = self.get_tool(func.get("name"))
        if not tool:
            raise ValueError(f"Tool '{func.get('name')}' not found")

        args = loads(func.get("arguments"))
        if isinstance(args, dict):
            result = tool.func(**args)
        elif isinstance(args, list):
            result = tool.func(*args)
        else:
            result = tool.func(args)

        # 同时支持同步和异步函数
        if asyncio.iscoroutine(result):
            return await result
        else:
            return result

    def list_tools(self) -> List[str]:
        """列出所有已注册的工具名称。

        Returns:
            工具名称列表。
        """
        return list(self._tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否已注册。

        Args:
            tool_name: 工具名称。

        Returns:
            如果已注册返回 True，否则返回 False。
        """
        return tool_name in self._tools

    def clear_tools(self):
        """清空所有已注册的工具（用于测试目的）。"""
        self._tools.clear()
