import pytest
import asyncio
from src.nuwa.tool import ToolRegistry, Tool, ToolEntity, ToolObjectParameter, ToolParameter


def test_tools_manager_initialization():
    """测试 ToolRegistry 初始化"""
    manager = ToolRegistry()
    assert manager.list_tools() == []
    assert manager.has_tool("nonexistent") is False


def test_add_tool():
    """测试添加工具"""
    manager = ToolRegistry()
    tool_entity = ToolEntity(
        name="test_tool",
        parameters=ToolParameter(type="string", description="测试参数"),
        description="测试工具"
    )
    def dummy_func():
        return "result"
    tool = Tool(func=dummy_func, entity=tool_entity)
    manager.add_tool(tool)
    assert manager.has_tool("test_tool") is True
    assert manager.list_tools() == ["test_tool"]
    retrieved = manager.get_tool("test_tool")
    assert retrieved is tool
    assert retrieved.func() == "result"


def test_tool_decorator():
    """测试装饰器注册工具"""
    manager = ToolRegistry()
    @manager.tool(
        name="decorated_tool",
        description="通过装饰器注册的工具",
        parameters=ToolObjectParameter(
            type="object",
            properties={
                "param": ToolParameter(type="string", description="参数")
            }
        )
    )
    async def decorated(param: str):
        return f"received {param}"
    # 装饰器已注册工具
    assert manager.has_tool("decorated_tool") is True
    tool = manager.get_tool("decorated_tool")
    assert tool is not None
    assert tool.entity.description == "通过装饰器注册的工具"


@pytest.mark.asyncio
async def test_call_tool_sync():
    """测试调用同步工具"""
    manager = ToolRegistry()
    tool_entity = ToolEntity(
        name="sync_tool",
        parameters=ToolObjectParameter(
            type="object",
            properties={
                "x": ToolParameter(type="number", description="数字")
            }
        )
    )
    def sync_func(x: int):
        return x * 2
    tool = Tool(func=sync_func, entity=tool_entity)
    manager.add_tool(tool)
    # 调用工具
    from src.nuwa.chat import ToolInvocation
    func = ToolInvocation(name="sync_tool", arguments='{"x": 5}')
    result = await manager.call_tool(func)
    assert result == 10


@pytest.mark.asyncio
async def test_call_tool_async():
    """测试调用异步工具"""
    manager = ToolRegistry()
    tool_entity = ToolEntity(
        name="async_tool",
        parameters=ToolObjectParameter(
            type="object",
            properties={
                "text": ToolParameter(type="string", description="文本")
            }
        )
    )
    async def async_func(text: str):
        await asyncio.sleep(0.001)
        return text.upper()
    tool = Tool(func=async_func, entity=tool_entity)
    manager.add_tool(tool)
    from src.nuwa.chat import ToolInvocation
    func = ToolInvocation(name="async_tool", arguments='{"text": "hello"}')
    result = await manager.call_tool(func)
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_call_tool_with_list_args():
    """测试使用列表参数调用工具"""
    manager = ToolRegistry()
    tool_entity = ToolEntity(
        name="list_tool",
        parameters=ToolParameter(type="array", description="数组")
    )
    def list_func(*args):
        return sum(args)
    tool = Tool(func=list_func, entity=tool_entity)
    manager.add_tool(tool)
    from src.nuwa.chat import ToolInvocation
    func = ToolInvocation(name="list_tool", arguments='[1, 2, 3]')
    result = await manager.call_tool(func)
    assert result == 6


@pytest.mark.asyncio
async def test_call_tool_not_found():
    """测试调用不存在的工具"""
    manager = ToolRegistry()
    from src.nuwa.chat import ToolInvocation
    func = ToolInvocation(name="missing", arguments='{}')
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        await manager.call_tool(func)


def test_clear_tools():
    """测试清除所有工具"""
    manager = ToolRegistry()
    tool_entity = ToolEntity(name="tool1", parameters=ToolParameter(type="string"))
    def dummy():
        pass
    manager.add_tool(Tool(func=dummy, entity=tool_entity))
    assert len(manager.list_tools()) == 1
    manager.clear_tools()
    assert len(manager.list_tools()) == 0
