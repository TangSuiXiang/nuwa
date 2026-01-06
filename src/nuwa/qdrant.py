import re
import json
import logging
import numpy as np

from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    OrderBy,
    Direction,
)
from openai.types.chat import ChatCompletionMessageParam

from .base import MessagesManager
from .vector import get_embeddings

logger = logging.getLogger()


class QdrantMessagesManager(MessagesManager):
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        embedding_client: AsyncOpenAI,
        https: bool = False,
        vector_size: int = 4096,
        embedding_model: str = "qwen3-embedding:8b-FP16",
        similarity_threshold: float = 0.7,
        min_chunk_length: int = 50,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_model = embedding_model
        self.client = AsyncQdrantClient(url=url, api_key=api_key, https=https)
        self._collection_created = False
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.min_chunk_length = min_chunk_length

    async def get_embeddings(self, inputs: List[str]) -> List[List[float]]:
        return await get_embeddings(
            inputs,
            embedding_model=self.embedding_model,
            client=self.embedding_client,
            dimensions=self.vector_size,
        )

    async def try_create_collection(self) -> bool:
        """Create collection if it doesn't exist."""
        if self._collection_created:
            return True

        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            # Create indexes for efficient querying
            await self.client.create_payload_index(
                self.collection_name, field_name="msg_id", field_schema="integer"
            )
            await self.client.create_payload_index(
                self.collection_name, field_name="create_time", field_schema="integer"
            )
            await self.client.create_payload_index(
                self.collection_name, field_name="session_id", field_schema="keyword"
            )
            await self.client.create_payload_index(
                self.collection_name,
                field_name="conversation_id",
                field_schema="keyword",
            )

        self._collection_created = True
        return True

    async def get_messages(
        self, session_id: str, user_input: str = ""
    ) -> List[ChatCompletionMessageParam]:
        """Retrieve relevant conversation history for the session."""
        await self.try_create_collection()

        messages: List[ChatCompletionMessageParam] = []

        try:
            # Get embeddings for semantic search
            query_embedding = (await self.get_embeddings([user_input]))[0]

            # Search for relevant conversation points
            search_result = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id", match=MatchValue(value=session_id)
                        )
                    ]
                ),
                with_payload=True,
                limit=2,  # Increased limit for better context
            )

            # Process search results and get full conversations
            conversation_ids = []
            logger.info("points %s", len(search_result.points))
            for point in search_result.points:
                conversation_id = point.payload.get("conversation_id", "")
                if not conversation_id or conversation_id in conversation_ids:
                    continue
                conversation_ids.append(conversation_id)

                # Get full conversation for this conversation_id
                conversation_records, _ = await self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="conversation_id",
                                match=MatchValue(value=conversation_id),
                            )
                        ]
                    ),
                    with_vectors=False,
                    with_payload=True,
                    order_by=OrderBy(key="msg_id", direction=Direction.ASC),
                    limit=100,
                )
                # Convert records to message format
                msg_ids = []
                for record in conversation_records:
                    msg_id = record.payload.get("msg_id")
                    if not msg_id or msg_id in msg_ids:
                        continue
                    logger.info(
                        "conversation_id %s, msg_id %s", conversation_id, msg_id
                    )
                    msg_ids.append(msg_id)
                    message = self._convert_payload_to_message(record.payload)
                    if message:
                        messages.append(message)

            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")

        except Exception as e:
            logger.error(f"Error retrieving messages for session {session_id}: {e}")

        return messages

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度，不依赖torch。

        Args:
            vec1: 第一个向量
            vec2: 第二个向量

        Returns:
            float: 余弦相似度
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _convert_payload_to_message(
        self, payload: dict
    ) -> Optional[ChatCompletionMessageParam]:
        """Convert Qdrant payload to OpenAI message format."""
        try:
            message = payload.copy()
            # Remove metadata fields
            for field in [
                "msg_id",
                "conversation_id",
                "session_id",
                "snippet",
                "create_time",
            ]:
                message.pop(field)
            return message
        except Exception as e:
            logger.warning(f"Failed to convert payload to message: {e}")
            return None

    async def save_messages(
        self,
        session_id: str,
        messages: List[ChatCompletionMessageParam],
    ):
        """Save conversation messages to Qdrant with semantic chunking."""
        if not messages:
            return

        await self.try_create_collection()

        conversation_id = str(uuid4())
        points: List[PointStruct] = []
        now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        if messages and messages[0].get("role") != "user":
            raise ValueError("messages the first must be user")

        for msg_id, message in enumerate(messages, 1):
            logger.info("msg_id %s, message %s", msg_id, message)
            try:
                role = message.get("role")
                if not isinstance(role, str):
                    continue

                # Handle tool calls for assistant messages
                if role == "assistant" and message.get("tool_calls"):
                    points = await self._process_tool_calls_message(
                        message, session_id, conversation_id, msg_id, now, points
                    )

                # Handle text content messages
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    points = await self._process_text_message(
                        content,
                        message,
                        session_id,
                        conversation_id,
                        msg_id,
                        now,
                        points,
                    )

            except Exception as e:
                logger.error(f"Error processing message {msg_id}: {e}")

        if points:
            try:
                await self.client.upsert(
                    collection_name=self.collection_name, points=points
                )
                logger.debug(f"Saved {len(points)} points for session {session_id}")
            except Exception as e:
                logger.error(f"Error saving points to Qdrant: {e}")

    async def _process_tool_calls_message(
        self,
        message: ChatCompletionMessageParam,
        session_id: str,
        conversation_id: str,
        msg_id: int,
        now: datetime,
        points: List[PointStruct],
    ):
        """Process assistant messages with tool calls."""
        tool_calls = message.get("tool_calls")
        if tool_calls:
            snippet = json.dumps(
                [tool_call.get("function") for tool_call in tool_calls],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            vector = (await self.get_embeddings([snippet]))[0]

            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "msg_id": msg_id,
                        "conversation_id": conversation_id,
                        **message,
                        "session_id": session_id,
                        "snippet": snippet,
                        "create_time": int(now.timestamp() * 1000),
                    },
                )
            )
        return points

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        将文本分割成段落。

        Args:
            text: 要分割的文本

        Returns:
            List[str]: 段落列表
        """
        # 按段落分割，保留分隔符
        paragraphs = re.split(r"(\n\s*\n)", text)
        # 过滤空段落并合并分隔符
        chunks = []
        current = ""

        for p in paragraphs:
            if p.strip() == "" and current:
                chunks.append(current.strip())
                current = ""
            elif p.strip():
                current += p

        if current.strip():
            chunks.append(current.strip())

        return chunks

    async def _process_text_message(
        self,
        content: str,
        message: ChatCompletionMessageParam,
        session_id: str,
        conversation_id: str,
        msg_id: int,
        now: datetime,
        points: List[PointStruct],
    ):
        """Process text content messages with semantic chunking."""
        paragraphs = [p.strip() for p in re.split(r"\n+", content) if p.strip()]
        if not paragraphs:
            return

        current_chunk = paragraphs[0]
        current_vector = (await self.get_embeddings([current_chunk]))[0]

        for paragraph in paragraphs[1:]:
            if not paragraph.strip():
                continue

            paragraph_vector = (await self.get_embeddings([paragraph]))[0]

            # Check semantic similarity
            similarity = self._cosine_similarity(
                np.array(current_vector), np.array(paragraph_vector)
            )

            if (
                similarity > self.similarity_threshold
                or len(current_chunk) < self.min_chunk_length
            ):  # Similar enough to merge
                current_chunk += "\n" + paragraph
                current_vector = (await self.get_embeddings([current_chunk]))[0]
            else:
                # Save current chunk and start new one
                points = await self._save_chunk(
                    current_chunk,
                    message,
                    session_id,
                    conversation_id,
                    msg_id,
                    now,
                    points,
                    current_vector,
                )
                current_chunk = paragraph
                current_vector = paragraph_vector

        # Save the final chunk
        return await self._save_chunk(
            current_chunk,
            message,
            session_id,
            conversation_id,
            msg_id,
            now,
            points,
            current_vector,
        )

    async def _save_chunk(
        self,
        chunk: str,
        message: ChatCompletionMessageParam,
        session_id: str,
        conversation_id: str,
        msg_id: int,
        now: datetime,
        points: List[PointStruct],
        vector: List[float],
    ):
        """Save a single chunk as a point."""
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "msg_id": msg_id,
                    "conversation_id": conversation_id,
                    **message,
                    "session_id": session_id,
                    "snippet": chunk,
                    "create_time": int(now.timestamp() * 1000),
                },
            )
        )
        return points
