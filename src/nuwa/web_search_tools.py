"""Nuwa 搜索模块，提供网页搜索工具集成。

本模块提供谷歌、百度、必应等搜索引擎的爬虫工具，使用 Playwright 和 Stealth 技术
绕过反爬虫检测，提取结构化搜索结果。遵循工具注册规范，返回 Tool 对象供 ReActAgent 使用。
设计规范：模块化（C-001）、错误处理（C-012）、外部依赖管理（M-019）。
"""

import logging
import asyncio

from lxml import etree  # type: ignore
from typing import List
from .tool import Tool, ToolEntity, ToolObjectParameter, ToolParameter
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

logger = logging.getLogger()


async def get_google_search_tool(proxies: List[str] = []) -> Tool:
    """获取谷歌搜索工具。

    使用 Playwright 控制浏览器模拟人类访问谷歌搜索，返回截图和页面内容。
    注意：此工具当前仅用于演示，未解析搜索结果，需根据实际需求扩展。
    遵循工具工厂模式，返回一个 Tool 实例。

    Args:
        proxies: 代理服务器列表，第一个元素将用作代理地址。

    Returns:
        配置好的 Tool 对象，工具名为 "google_search"。
    """

    async def google_search(query: str):
        """执行谷歌搜索。

        使用 Stealth 插件隐藏自动化特征，访问谷歌搜索页面并截图。
        当前版本仅截图，未提取结构化数据，可根据需要扩展解析逻辑。

        Args:
            query: 搜索关键词。

        Returns:
            当前返回 None，可扩展为返回搜索结果列表。
        """
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(channel="msedge", headless=False)
            # 可选择使用代理（如果提供了代理列表）
            page = await browser.new_page(
                proxy={"server": proxies[0]} if proxies else None
            )
            await page.goto(url=f"https://www.google.com/search?q={query}")
            logger.debug("resp %s", await page.content())
            await page.screenshot(path="./google_search.png", full_page=True)
            await browser.close()

    return Tool(
        func=google_search,
        entity=ToolEntity(
            name="google_search",
            description="谷歌搜索",
            parameters=ToolObjectParameter(
                type="object",
                properties={
                    "query": ToolParameter(type="string", description="搜索词")
                },
            ),
        ),
    )


async def get_baidu_search_tool(proxies: List[str] = []) -> Tool:
    """获取百度搜索工具。

    使用 Playwright 访问百度搜索，解析搜索结果页，提取标题、链接和摘要。
    遵循网页爬虫最佳实践，使用 XPath 定位元素，并等待页面加载完成。

    Args:
        proxies: 代理服务器列表。

    Returns:
        配置好的 Tool 对象，工具名为 "baidu_search"。
    """

    async def baidu_search(query: str):
        """执行百度搜索并返回结构化结果。

        访问百度搜索页面，等待结果区域加载，使用 lxml 解析 HTML 并提取信息。
        支持多种摘要选择器，适应百度搜索结果页的不同样式。

        Args:
            query: 搜索关键词。

        Returns:
            搜索结果列表，每个元素为包含 title、link、synopsis 的字典。
        """
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(channel="msedge", headless=False)
            page = await browser.new_page(
                proxy={"server": proxies[0]} if proxies else None
            )
            await page.goto(url=f"https://www.baidu.com/s?wd={query}")
            await page.wait_for_selector("#content_left")
            await asyncio.sleep(1)
            content = await page.content()
            html = etree.HTML(content)
            logger.debug("resp %s", html)
            results = []
            for content_left in html.xpath('//*[@id="content_left"]'):
                for item in content_left.xpath('div[contains(@class,"c-container")]'):
                    title = item.xpath("string(div//h3)").strip()
                    if not title:
                        continue
                    logger.debug("title %s", title)
                    link = item.xpath("string(div//h3/a/@href)").strip()
                    logger.debug("link %s", link)
                    # 尝试多种摘要选择器
                    synopsis = (
                        item.xpath('string(div//div[contains(@class,"summary-gap")])')
                        or item.xpath(
                            'string(div//div[contains(@class,"content")]//div[contains(@class,"_no-spacing")]//p)'
                        )
                        or item.xpath(
                            'string(div//div[contains(@class,"card-normal")]//div[contains(@class,"text_")])'
                        )
                        or item.xpath(
                            'string(div//div[contains(@class,"bookinfo")]//div[contains(@class,"bookinfo-intro")])'
                        )
                        or item.xpath(
                            'string(div//div[contains(@class,"pc-tabs-content")]/div)'
                        )
                    ).strip()
                    logger.debug("synopsis %s", synopsis)
                    results.append({"title": title, "link": link, "synopsis": synopsis})
            await page.screenshot(path="./baidu_search.png", full_page=True)
            await browser.close()
            return results

    return Tool(
        func=baidu_search,
        entity=ToolEntity(
            name="baidu_search",
            description="百度搜索",
            parameters=ToolObjectParameter(
                type="object",
                properties={
                    "query": ToolParameter(type="string", description="搜索词")
                },
            ),
        ),
    )


async def get_bing_search_tool(proxies: List[str] = []) -> Tool:
    """获取必应搜索工具。

    访问必应搜索（中文版），解析结果列表，提取标题、链接和摘要。
    遵循工具工厂模式，返回 Tool 实例。

    Args:
        proxies: 代理服务器列表。

    Returns:
        配置好的 Tool 对象，工具名为 "bing_search"。
    """

    async def bing_search(query: str):
        """执行必应搜索并返回结构化结果。

        使用 Playwright 访问必应搜索，等待结果区域加载，使用 XPath 提取数据。
        提取的字段包括标题、链接和摘要（caption）。

        Args:
            query: 搜索关键词。

        Returns:
            搜索结果列表，每个元素为包含 title、synopsis、link 的字典。
        """
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(channel="msedge", headless=False)
            page = await browser.new_page(
                proxy={"server": proxies[0]} if proxies else None
            )
            await page.goto(url=f"https://cn.bing.com/search?q={query}")
            await page.wait_for_selector(".b_algo")
            await asyncio.sleep(1)
            content = await page.content()
            html = etree.HTML(content)
            ret = []
            for results in html.xpath('//*[@id="b_results"]'):
                for item in results.xpath('li[@class="b_algo"]'):
                    title = item.xpath("string(h2)")
                    logger.debug("title %s", title)
                    link = item.xpath("string(h2/a/@href)")
                    logger.debug("link %s", link)
                    synopsis = item.xpath('string(div[contains(@class,"b_caption")])')
                    logger.debug("synopsis %s", synopsis)
                    ret.append({"title": title, "synopsis": synopsis, "link": link})
            logger.debug("resp %s", html)
            await page.screenshot(path="./bing_search.png", full_page=True)
            await browser.close()
            return ret

    return Tool(
        func=bing_search,
        entity=ToolEntity(
            name="bing_search",
            description="必应搜索",
            parameters=ToolObjectParameter(
                type="object",
                properties={
                    "query": ToolParameter(type="string", description="搜索词")
                },
            ),
        ),
    )
