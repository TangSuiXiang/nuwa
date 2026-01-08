import pytest
import logging

from src.nuwa.re_act import ReActAgent
from src.nuwa.tool import ToolsManager, ToolObjectParameter, ToolParameter

logger = logging.getLogger()


@pytest.mark.asyncio
async def test_re_act_agent_once_chat(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个通用助手",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "你好"}):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_multi_chat(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个通用助手",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "现在是2025年7月1号早上7点10分"}):
        logger.info("测试 %s", c)
    async for c in agent.run({"user": "你好，现在是几点"}):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_with_time(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个通用助手",
        api_key=api_key,
        with_time=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "现在是2025年7月1号早上7点10分么"}):
        logger.info("测试 %s", c)
    async for c in agent.run({"user": "你好，现在是几点"}):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_with_think(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个通用助手",
        api_key=api_key,
        extra_body={"enable_thinking": True},
        with_time=False,
        stream=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "猫有时间的概念么"}):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_with_think_and_time(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个通用助手",
        api_key=api_key,
        extra_body={"enable_thinking": True},
        with_time=True,
        stream=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "还要再过多久才天亮呢"}):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_with_tools(api_key: str):
    tools = ToolsManager()

    @tools.tool(
        name="get_weather",
        description="获取指定城市的天气。",
        parameters=ToolObjectParameter(
            type="object",
            properties={
                "city": ToolParameter(type="string", description="中国城市完整名称")
            },
        ),
    )
    async def get_weather(city: str):
        return {"desc": "部分晴朗：19摄氏度，湿度69%"}

    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个通用助手",
        api_key=api_key,
        with_time=True,
        tools_manager=tools,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "今天广州天气怎么样"}):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_with_tools_and_stream(api_key: str):
    tools = ToolsManager()

    @tools.tool(
        name="get_weather",
        description="获取指定城市的天气。",
        parameters=ToolObjectParameter(
            type="object",
            properties={
                "city": ToolParameter(type="string", description="中国城市完整名称")
            },
        ),
    )
    async def get_weather(city: str):
        return {"desc": "部分晴朗：19摄氏度，湿度69%"}

    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个通用助手",
        api_key=api_key,
        extra_body={"enable_thinking": True},
        with_time=True,
        stream=True,
        tools_manager=tools,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "今天广州天气怎么样"}):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_system_prompt_format(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        with_time=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "你好呀", "system": {"role": "猫娘女仆"}}):
        logger.info("测试 %s", c)
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}, 你的性格是{personality}",
        api_key=api_key,
        with_time=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {"user": "你好呀", "system": {"role": "猫娘女仆", "personality": "高冷"}}
    ):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_system_prompt_format_with_think(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        with_time=True,
        extra_body={"enable_thinking": True},
        stream=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run({"user": "你好呀", "system": {"role": "猫娘女仆"}}):
        logger.info("测试 %s", c)
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}, 你的性格是{personality}",
        api_key=api_key,
        extra_body={"enable_thinking": True},
        stream=True,
        with_time=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {"user": "你好呀", "system": {"role": "猫娘女仆", "personality": "高冷"}}
    ):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_mcp(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        with_time=True,
        mcp="http://192.168.110.10:12119/mcp",
        extra_body={"enable_thinking": False},
        stream=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {"user": "帮我生成一张猫娘的图片", "system": {"role": "通用助手"}}
    ):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_mcp_with_think(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        with_time=True,
        mcp="http://192.168.110.10:12119/mcp",
        extra_body={"enable_thinking": True},
        stream=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {
            "user": "帮我生成一张穿白色蕾丝丝袜的猫娘的图片",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_mcp_with_local_tool_mgr(api_key: str):
    tools = ToolsManager()

    @tools.tool(
        name="get_weather",
        description="获取指定城市的天气。",
        parameters=ToolObjectParameter(
            type="object",
            properties={
                "city": ToolParameter(type="string", description="中国城市完整名称")
            },
        ),
    )
    async def get_weather(city: str):
        return {"desc": "部分晴朗：19摄氏度，湿度69%"}

    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        with_time=True,
        tools_manager=tools,
        mcp="http://192.168.110.10:12119/mcp",
        extra_body={"enable_thinking": False},
        stream=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {
            "user": "帮我生成一张穿白色蕾丝丝袜的猫娘的图片",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)
    async for c in agent.run(
        {
            "user": "今天天气怎么样",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_mcp_with_local_tool_mgr_with_think(api_key: str):
    tools = ToolsManager()

    @tools.tool(
        name="get_weather",
        description="获取指定城市的天气。",
        parameters=ToolObjectParameter(
            type="object",
            properties={
                "city": ToolParameter(type="string", description="中国城市完整名称")
            },
        ),
    )
    async def get_weather(city: str):
        return {"desc": "部分晴朗：19摄氏度，湿度69%"}

    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        with_time=True,
        tools_manager=tools,
        mcp="http://192.168.110.10:12119/mcp",
        extra_body={"enable_thinking": True},
        stream=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {
            "user": "帮我生成一张穿白色蕾丝丝袜的猫娘的图片",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)
    async for c in agent.run(
        {
            "user": "广州今天天气怎么样",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_history(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        with_time=True,
        enable_chat_history=False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {
            "user": "请记住，我现在在广州。",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)
    async for c in agent.run(
        {
            "user": "我在哪个城市",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_history_with_think(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        extra_body={"enable_thinking": True},
        stream=True,
        with_time=True,
        enable_chat_history=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {
            "user": "请记住，我现在在广州。",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)
    async for c in agent.run(
        {
            "user": "我在哪个城市",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)


@pytest.mark.asyncio
async def test_re_act_agent_selection(api_key: str):
    agent = ReActAgent(
        model="deepseek-v3.2",
        system_prompt="你是一个{role}",
        api_key=api_key,
        extra_body={"enable_thinking": True},
        stream=True,
        with_time=True,
        enable_chat_history=True,
        enable_selection=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async for c in agent.run(
        {
            "user": "请调用工具询问用户现在是白天还是黑夜",
            "system": {"role": "通用助手"},
        }
    ):
        logger.info("测试 %s", c)
    async for c in agent.run({
            "user": "黑夜",
            "system": {"role": "通用助手"},
    }):
        logger.info("测试 %s", c)
        