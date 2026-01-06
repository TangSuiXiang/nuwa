import pytest
import logging

from src.nuwa.search import (
    get_google_search_tool,
    get_baidu_search_tool,
    get_bing_search_tool,
)

logger = logging.getLogger()


@pytest.mark.asyncio
async def test_google_search():
    tool = await get_google_search_tool(proxies=["socks5://192.168.31.45:10808"])
    logger.info("tool resp %s", await tool.func("ceshi1"))


@pytest.mark.asyncio
async def test_baidu_search():
    tool = await get_baidu_search_tool(proxies=["socks5://192.168.31.45:10808"])
    # logger.info("tool resp %s", await tool.func("ceshi1"))
    logger.info("tool resp %s", await tool.func("微信"))


@pytest.mark.asyncio
async def test_bing_search():
    tool = await get_bing_search_tool(proxies=["socks5://192.168.31.45:10808"])
    logger.info("tool resp %s", await tool.func("维基百科"))
    logger.info("tool resp %s", await tool.func("你好"))
    logger.info("tool resp %s", await tool.func("QQ"))
