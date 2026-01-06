import logging
import asyncio

from lxml import etree
from typing import List
from .tool import Tool, ToolEntity, ToolObjectParameter, ToolParameter
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

logger = logging.getLogger()


async def get_google_search_tool(proxies: List[str] = []) -> Tool:
    async def google_search(query: str):
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(channel="msedge", headless=False)
            # context = await browser.new_context(
            #     proxy={"server": proxies[0]}, **iphone15promax
            # )
            # page = await context.new_page()
            page = await browser.new_page(proxy={"server": proxies[0]})
            # await page.goto(url="https://arh.antoinevastel.com/bots/areyouheadless")
            await page.goto(url=f"https://www.google.com/search?q={query}")
            # await page.goto(url=f"https://www.baidu.com/s?wd={query}")
            logger.info("resp %s", await page.content())
            await page.screenshot(path="./google_search.png", full_page=True)
            # await context.close()
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
    async def baidu_search(query: str):
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(channel="msedge", headless=False)
            page = await browser.new_page(proxy={"server": proxies[0]})
            await page.goto(url=f"https://www.baidu.com/s?wd={query}")
            await page.wait_for_selector("#content_left")
            await asyncio.sleep(1)
            content = await page.content()
            html = etree.HTML(content)
            logger.info("resp %s", html)
            results = []
            for content_left in html.xpath('//*[@id="content_left"]'):
                for item in content_left.xpath('div[contains(@class,"c-container")]'):
                    title = item.xpath("string(div//h3)").strip()
                    if not title:
                        continue
                    logger.info("title %s", title)
                    link = item.xpath("string(div//h3/a/@href)").strip()
                    logger.info("link %s", link)
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
                    logger.info("synopsis %s", synopsis)
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
    async def bing_search(query: str):
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(channel="msedge", headless=False)
            page = await browser.new_page(proxy={"server": proxies[0]})
            await page.goto(url=f"https://cn.bing.com/search?q={query}")
            await page.wait_for_selector(".b_algo")
            await asyncio.sleep(1)
            content = await page.content()
            html = etree.HTML(content)
            ret = []
            for results in html.xpath('//*[@id="b_results"]'):
                for item in results.xpath('li[@class="b_algo"]'):
                    title = item.xpath("string(h2)")
                    logger.info("title %s", title)
                    link = item.xpath("string(h2/a/@href)")
                    logger.info("link %s", link)
                    synopsis = item.xpath('string(div[contains(@class,"b_caption")])')
                    logger.info("synopsis %s", synopsis)
                    ret.append({"title": title, "synopsis": synopsis, "link": link})
            logger.info("resp %s", html)
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
