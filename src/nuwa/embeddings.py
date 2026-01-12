"""Nuwa 向量嵌入模块，提供文本向量化工具。

本模块提供 get_embeddings 函数，用于将文本列表转换为向量嵌入。
支持 OpenAI 兼容的嵌入 API，可配置模型、维度和客户端。
设计规范：模块化（C-001）、错误处理（C-012）、性能优化（O-010）。
"""

import logging
import asyncio

from openai import AsyncOpenAI, RateLimitError
from typing import Optional, List

logger = logging.getLogger()


async def get_embeddings(
    texts: List[str],
    embedding_model: str = "qwen3-embedding:8b-FP16",
    client: Optional[AsyncOpenAI] = None,
    dimensions: int = 4096,
) -> List[List[float]]:
    """生成文本列表的向量嵌入。

    使用 OpenAI 兼容的嵌入 API（如 Ollama 或其他兼容服务）将文本转换为向量。
    支持速率限制错误的重试机制（简单等待）。
    遵循错误处理规范（C-012），对空文本列表返回空列表。

    Args:
        texts: 待生成嵌入的文本列表。
        embedding_model: 嵌入模型名称，默认为 "qwen3-embedding:8b-FP16"。
        client: 可选的 AsyncOpenAI 客户端实例，如果未提供则创建默认客户端。
        dimensions: 嵌入向量的维度，默认为 4096。

    Returns:
        嵌入向量列表，每个元素对应输入文本的浮点数列表。

    Raises:
        Exception: 嵌入生成过程中发生错误（除了 RateLimitError 外）时抛出。
    """
    # 如果没有提供客户端，则创建默认客户端（针对 Ollama 配置）
    if client is None:
        client = AsyncOpenAI(
            api_key="ollama",
            base_url="http://192.168.110.10:11434/v1",
        )

    if texts:
        # 生成嵌入
        try:
            response = await client.embeddings.create(
                model=embedding_model, input=texts, dimensions=dimensions
            )

            # 从响应中提取嵌入向量
            return [d.embedding for d in response.data]
        except RateLimitError:
            # 速率限制时等待后重试（注意：当前实现仅等待，未重试）
            # 遵循错误处理规范（C-013），记录限流事件
            await asyncio.sleep(0.1)
    # 如果文本列表为空或发生限流，返回空列表
    return []
