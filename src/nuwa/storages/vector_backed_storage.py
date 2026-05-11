"""Nuwa 存储层模块，基于 Qdrant 向量数据库的消息管理器实现。

本模块提供 VectorBackedStorage 类，实现 ConversationStorage 抽象接口。
遵循抽象工厂模式（P-001），提供可插拔的消息存储后端。
支持语义分块、向量检索和对话历史的高效管理。
设计规范：模块化（C-001）、错误处理（C-012）、性能优化（O-010）。
"""

import re
import json
import logging
import numpy as np

from typing import List, Optional, Dict
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
    ScoredPoint,
    IntegerIndexParams,
    IntegerIndexType,
    KeywordIndexType,
    KeywordIndexParams,
)
from openai.types.chat import ChatCompletionMessageParam

from .base import ConversationStorage

logger = logging.getLogger()


class VectorBackedStorage(ConversationStorage):
    """基于向量数据库的消息存储后端。

    实现 ConversationStorage 抽象接口，提供消息的持久化、检索和清除功能。
    利用向量嵌入进行语义检索，支持对话级别的分块和相似度合并。
    遵循抽象工厂模式（P-001），可作为消息存储的具体实现。
    设计规范：类名大驼峰（D-007）、单一职责（C-006）、错误处理（C-012）。

    Attributes:
        collection_name: Qdrant 集合名称。
        vector_size: 向量维度大小。
        embedding_model: 嵌入模型名称。
        client: AsyncQdrantClient 实例。
        _collection_created: 标识集合是否已创建。
        embedding_client: 嵌入生成客户端（AsyncOpenAI）。
        similarity_threshold: 语义相似度阈值，用于分块合并。
        min_chunk_length: 最小分块长度（字符数）。
    """

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
        """初始化向量存储后端。

        Args:
            url: Qdrant 服务器 URL。
            api_key: Qdrant API 密钥。
            collection_name: 集合名称。
            embedding_client: 用于生成嵌入的 AsyncOpenAI 客户端。
            https: 是否使用 HTTPS，默认为 False。
            vector_size: 向量维度，默认为 4096。
            embedding_model: 嵌入模型名称，默认为 "qwen3-embedding:8b-FP16"。
            similarity_threshold: 语义相似度阈值，默认为 0.7。
            min_chunk_length: 最小分块长度，默认为 50。

        设计规范：参数类型提示（D-014）、默认参数合理（C-011）。
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_model = embedding_model
        self.client = AsyncQdrantClient(url=url, api_key=api_key, https=https)
        self._collection_created = False
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.min_chunk_length = min_chunk_length

    async def get_embeddings(self, inputs: List[str]) -> List[List[float]]:
        """生成文本列表的向量嵌入。

        委托给 vector 模块的 get_embeddings 函数，统一嵌入生成逻辑。
        遵循错误处理规范（C-012），异常由被调用函数处理。

        Args:
            inputs: 文本字符串列表。

        Returns:
            嵌入向量列表，每个向量为浮点数列表。
        """
        # return await get_embeddings(
        #     inputs,
        #     embedding_model=self.embedding_model,
        #     client=self.embedding_client,
        #     dimensions=self.vector_size,
        # )
        return []

    async def try_create_collection(self) -> bool:
        """确保集合存在，不存在则创建。

        幂等操作，避免重复创建集合。
        同时创建必要的索引以优化查询性能。
        遵循性能优化规范（O-010），索引加速检索。

        Returns:
            如果集合已存在或创建成功返回 True。
        """
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
            # 创建索引以优化查询
            await self.client.create_payload_index(
                self.collection_name,
                field_name="msg_id",
                field_schema=IntegerIndexParams(type=IntegerIndexType.INTEGER),
            )
            await self.client.create_payload_index(
                self.collection_name,
                field_name="create_time",
                field_schema=IntegerIndexParams(type=IntegerIndexType.INTEGER),
            )
            await self.client.create_payload_index(
                self.collection_name,
                field_name="session_id",
                field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
            )
            await self.client.create_payload_index(
                self.collection_name,
                field_name="conversation_id",
                field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
            )

        self._collection_created = True
        return True

    async def get_messages(
        self, session_id: str, user_input: str = ""
    ) -> List[ChatCompletionMessageParam]:
        """检索与会话相关的历史消息。

        结合语义检索和最近对话，返回按时间排序的消息列表。
        流程：生成查询嵌入 → 语义搜索相关点 → 获取完整对话 → 转换消息格式。
        遵循性能优化规范（O-010），限制检索数量避免超载。

        Args:
            session_id: 会话标识符。
            user_input: 用户输入（用于语义检索），默认为空。

        Returns:
            消息列表，格式符合 OpenAI ChatCompletion 接口。

        Raises:
            内部异常被捕获并记录，返回空列表。
        """
        await self.try_create_collection()

        time_messages_map: Dict[int, List[ChatCompletionMessageParam]] = {}

        try:
            # 生成查询嵌入以进行语义搜索
            query_embedding = (await self.get_embeddings([user_input]))[0]

            # 搜索相关对话点
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
                with_vectors=False,
                with_payload=True,
                limit=5,  # 增加限制以获得更好的上下文
            )
            # 获取最近记录以补充语义搜索结果
            recent_records, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id", match=MatchValue(value=session_id)
                        )
                    ]
                ),
                with_vectors=False,
                with_payload=True,
                order_by=OrderBy(key="create_time", direction=Direction.DESC),
                limit=200,
            )
            conversation_id = None
            for record in recent_records:
                if not record.payload:
                    continue
                if conversation_id != record.payload.get("conversation_id", ""):
                    search_result.points.append(
                        ScoredPoint(
                            id=record.id, version=0, score=0, payload=record.payload
                        )
                    )
                    if conversation_id is not None:
                        conversation_id = None
                        break
                    conversation_id = record.payload.get("conversation_id", "")
            # 处理搜索结果并获取完整对话
            conversation_ids = []
            logger.debug("points %s", len(search_result.points))
            for point in search_result.points:
                if not point.payload:
                    continue
                conversation_id = point.payload.get("conversation_id", "")
                if not conversation_id or conversation_id in conversation_ids:
                    continue
                conversation_ids.append(conversation_id)

                # 获取该 conversation_id 的完整对话
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
                    limit=500,
                )
                messages: List[ChatCompletionMessageParam] = []
                # 将记录转换为消息格式
                msg_ids = []
                for record in conversation_records:
                    if not record.payload:
                        continue
                    msg_id = record.payload.get("msg_id")
                    if not msg_id or msg_id in msg_ids:
                        continue
                    logger.debug(
                        "conversation_id %s, msg_id %s", conversation_id, msg_id
                    )
                    msg_ids.append(msg_id)
                    message = self._convert_payload_to_message(record.payload)
                    if message:
                        messages.append(message)
                time_messages_map[point.payload.get("create_time", 0)] = messages

        except Exception as e:
            logger.error(f"Error retrieving messages for session {session_id}: {e}")

        all_messages: List[ChatCompletionMessageParam] = []
        for k in sorted(time_messages_map.keys()):
            msgs = time_messages_map.get(k, [])
            all_messages.extend(msgs)
        return all_messages

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度。

        不依赖 torch，使用 numpy 实现，降低外部依赖。
        遵循算法注释规范（D-012），解释计算过程。

        Args:
            vec1: 第一个向量。
            vec2: 第二个向量。

        Returns:
            余弦相似度，范围 [-1, 1]，归一化到 [0, 1] 上下文。
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
        """将 Qdrant 负载转换为 OpenAI 消息格式。

        移除元数据字段（msg_id, conversation_id 等），保留原始消息内容。
        遵循错误处理规范（C-012），转换失败时返回 None 并记录警告。

        Args:
            payload: Qdrant 点负载字典。

        Returns:
            转换后的消息，如果失败则返回 None。
        """
        try:
            message = payload.copy()
            # 移除元数据字段
            for field in [
                "msg_id",
                "conversation_id",
                "session_id",
                "snippet",
                "create_time",
            ]:
                message.pop(field)
            return message  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to convert payload to message: {e}")
            return None

    async def save_messages(
        self,
        session_id: str,
        messages: List[ChatCompletionMessageParam],
    ):
        """保存对话消息到 Qdrant，支持语义分块。

        将消息列表按语义相似度分块，每个块作为一个向量点存储。
        流程：创建集合 → 遍历消息 → 处理工具调用或文本内容 → 分块保存。
        遵循性能优化规范（O-010），批量插入点以提高效率。

        Args:
            session_id: 会话标识符。
            messages: 要保存的消息列表。

        Raises:
            ValueError: 当消息列表第一条不是用户或工具消息时。
        """
        if not messages:
            return

        await self.try_create_collection()

        conversation_id = str(uuid4())
        points: List[PointStruct] = []
        now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        if messages and messages[0].get("role") not in ["user", "tool"]:
            raise ValueError("messages the first must be user or tool")

        for msg_id, message in enumerate(messages, 1):
            logger.debug("msg_id %s, message %s", msg_id, message)
            try:
                role = message.get("role")
                if not isinstance(role, str):
                    continue

                # 处理助手消息中的工具调用
                if role == "assistant" and message.get("tool_calls"):
                    points = await self._process_tool_calls_message(
                        message, session_id, conversation_id, msg_id, now, points
                    )

                # 处理文本内容消息
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    points = (
                        await self._process_text_message(
                            content,
                            message,
                            session_id,
                            conversation_id,
                            msg_id,
                            now,
                            points,
                        )
                        or []
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
        """处理包含工具调用的助手消息。

        将工具调用列表序列化为摘要片段，生成嵌入并存储为一个点。
        遵循数据完整性规范，保留原始消息和元数据。

        Args:
            message: 助手消息。
            session_id: 会话 ID。
            conversation_id: 对话 ID。
            msg_id: 消息序号。
            now: 当前时间。
            points: 点列表，将新点追加到此列表。

        Returns:
            更新后的点列表。
        """
        tool_calls = message.get("tool_calls")
        if tool_calls:
            snippet = json.dumps(
                [tool_call.get("function") for tool_call in tool_calls],  # type: ignore[attr-defined]
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
        """处理文本内容消息，进行语义分块。

        按换行符分割段落，根据相似度阈值和最小长度决定合并或分割。
        算法：遍历段落，计算当前块与下一段的相似度，决定是否合并。
        遵循复杂算法注释规范（D-012），详细解释分块逻辑。

        Args:
            content: 消息文本内容。
            message: 原始消息对象。
            session_id: 会话 ID。
            conversation_id: 对话 ID。
            msg_id: 消息序号。
            now: 当前时间。
            points: 点列表。

        Returns:
            更新后的点列表。
        """
        paragraphs = [p.strip() for p in re.split(r"\n+", content) if p.strip()]
        if not paragraphs:
            return

        current_chunk = paragraphs[0]
        current_vector = (await self.get_embeddings([current_chunk]))[0]

        for paragraph in paragraphs[1:]:
            if not paragraph.strip():
                continue

            paragraph_vector = (await self.get_embeddings([paragraph]))[0]

            # 检查语义相似度
            similarity = self._cosine_similarity(
                np.array(current_vector), np.array(paragraph_vector)
            )

            if (
                similarity > self.similarity_threshold
                or len(current_chunk) < self.min_chunk_length
            ):  # 相似度足够高或当前块太短，则合并
                current_chunk += "\n" + paragraph
                current_vector = (await self.get_embeddings([current_chunk]))[0]
            else:
                # 保存当前块并开始新块
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

        # 保存最后一块
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
        """将单个分块保存为一个点。

        构造 PointStruct 包含向量、负载（消息内容、元数据等）。
        遵循数据模型规范，确保负载结构一致。

        Args:
            chunk: 分块文本。
            message: 原始消息对象。
            session_id: 会话 ID。
            conversation_id: 对话 ID。
            msg_id: 消息序号。
            now: 当前时间。
            points: 点列表。
            vector: 分块向量。

        Returns:
            更新后的点列表。
        """
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

    async def clear_messages(self, session_id: str):
        """清除指定会话的所有历史聊天消息。

        根据会话 ID 过滤并删除 Qdrant 中的对应点。
        遵循错误处理规范（C-012），异常向上抛出。

        Args:
            session_id: 会话标识符。

        Raises:
            Exception: 当删除操作失败时。
        """
        await self.try_create_collection()
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="session_id", match=MatchValue(value=session_id)
                        )
                    ]
                ),
            )
            logger.debug(f"Cleared messages for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing messages for session {session_id}: {e}")
            raise e
