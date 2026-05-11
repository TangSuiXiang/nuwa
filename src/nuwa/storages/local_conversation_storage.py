import os
import asyncio
import msgpack

from .base import ConversationStorage
from typing import List, Optional, Union
from openai.types.chat import ChatCompletionMessageParam


class LocalFileConversationStorage(ConversationStorage):
    def __init__(self, path: Union[str, os.PathLike[str]]):
        self._path = path
        self._store: dict[str, List[ChatCompletionMessageParam]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if not os.path.exists(self._path):
            return

        loaded: dict[str, List[ChatCompletionMessageParam]] = {}
        for entry in os.scandir(self._path):
            if not entry.is_file():
                continue
            with open(entry.path, "rb") as f:
                data = msgpack.unpack(f)
            if not isinstance(data, list):
                continue
            loaded[entry.name] = data

        self._store = loaded

    async def close(self) -> None:
        async with self._lock:
            os.makedirs(self._path, exist_ok=True)
            for session_id, messages in self._store.items():
                file_path = os.path.join(self._path, session_id)
                with open(file_path, "wb") as f:
                    msgpack.pack(messages, f)
            self._store.clear()

    async def get_messages(
        self, session_id: str, user_input: str = ""
    ) -> List[ChatCompletionMessageParam]:
        async with self._lock:
            return list(self._store.get(session_id, []))

    async def save_messages(
        self, session_id: str, messages: List[ChatCompletionMessageParam]
    ):
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].extend(messages)

    async def clear_messages(self, session_id: str):
        async with self._lock:
            self._store.pop(session_id, None)
        file_path = os.path.join(self._path, session_id)
        if os.path.exists(file_path):
            os.remove(file_path)
