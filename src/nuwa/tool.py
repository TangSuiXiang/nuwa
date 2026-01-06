import json
import typing
import logging
import asyncio

from .chat import Function
from json_repair import loads
from pydantic import BaseModel
from typing import Union, Literal, Optional, Dict, List, Any, Callable
from mcp.types import Tool as MCPTool

logger = logging.getLogger()


class ToolParameter(BaseModel):
    type: Literal["object", "string", "number", "boolean", "array"]
    description: Optional[str] = None
    enum: Optional[List[str]] = None


class ToolArrayParameter(ToolParameter):
    items: Union["ToolObjectParameter", "ToolArrayParameter", ToolParameter]


class ToolObjectParameter(ToolParameter):
    properties: Dict[
        str, Union["ToolObjectParameter", ToolArrayParameter, ToolParameter]
    ] = {}
    required: List[str] = []


class ToolEntity(BaseModel):
    name: str
    parameters: Union[ToolObjectParameter, ToolParameter] = ToolObjectParameter(
        type="object", properties={}, required=[]
    )
    description: Optional[str] = None


class Tool:
    def __init__(self, func: Callable, entity: ToolEntity):
        self.func = func
        self.entity = entity


def get_tool_entity(tool: MCPTool) -> ToolEntity:
    properties = {}
    for k, v in tool.inputSchema.get("properties", {}).items():
        if not isinstance(v, dict):
            continue
        t = v.get("type")
        if not isinstance(t, str):
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


class ToolsManager:
    """ToolManager without singleton pattern."""

    def __init__(self, init_tools: Dict[str, Tool] = {}):
        """Initialize the tool manager."""
        self._tools: Dict[str, Tool] = init_tools.copy()

    def tool(
        self,
        name: Optional[str] = None,
        parameters: Union[ToolObjectParameter, ToolParameter] = ToolObjectParameter(
            type="object", properties={}, required=[]
        ),
        description: Optional[str] = None,
    ):
        """Register a function tool with the manager."""

        def decorator(func: Callable):
            tool_name = name or func.__name__
            desc = description or func.__doc__

            # Process function annotations to automatically populate parameters
            annotations = {}
            if isinstance(func.__annotations__, dict):
                annotations = func.__annotations__
            for k, v in annotations.items():
                # Skip return annotation
                if k == "return":
                    continue

                # Skip already defined parameters
                if k in parameters.properties:
                    continue

                item_type = v
                item_desc = None

                # Handle Annotated types
                if hasattr(v, "__metadata__"):
                    # This is for newer Python versions with typing.Annotated
                    item_type = v.__args__[0] if hasattr(v, "__args__") else v
                    item_desc = (
                        getattr(v, "__metadata__", (None,))[0]
                        if hasattr(v, "__metadata__")
                        else None
                    )
                elif isinstance(v, typing._AnnotatedAlias):
                    # This is for older Python versions
                    item_type = v.__args__[0]
                    item_desc = (v.__metadata__ or ("",))[0]

                # Map Python types to JSON schema types
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
            logger.info("Registered tool: %s", tool_name)
            return func

        return decorator

    def add_tool(self, tool: Tool):
        if not self.has_tool(tool_name=tool.entity.name):
            self._tools[tool.entity.name] = tool

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool instance by name."""
        return self._tools.get(tool_name)

    async def call_tool(self, func: Function) -> Any:
        """Call a tool with the provided function specification."""
        logger.info("call tool %s", func)
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

        # Handle both sync and async functions
        if asyncio.iscoroutine(result):
            return await result
        else:
            return result

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools

    def clear_tools(self):
        """Clear all registered tools (for testing purposes)."""
        self._tools.clear()
