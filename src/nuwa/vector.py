"""Vector embedding utilities for text processing."""

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
    """
    Generate embeddings for a list of texts using OpenAI API.

    Args:
        texts: List of texts to generate embeddings for
        embedding_model: Name of the embedding model to use
        client: Optional AsyncOpenAI client instance
        dimensions: Dimensionality of the resulting embeddings

    Returns:
        List of embeddings, one for each input text

    Raises:
        Exception: If there's an error during embedding generation
    """
    # Create a default client if none provided
    if client is None:
        client = AsyncOpenAI(
            api_key="ollama",
            base_url="http://192.168.110.10:11434/v1",
        )

    if texts:
        # Generate embeddings
        try:
            response = await client.embeddings.create(
                model=embedding_model, input=texts, dimensions=dimensions
            )

            # Extract embeddings from response
            return [d.embedding for d in response.data]
        except RateLimitError:
            asyncio.sleep(0.1)
    return []
