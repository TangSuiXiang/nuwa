from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Union, Literal, Dict


class AsyncClosableContext(ABC):
    def __init__(self) -> None:
        self._initialized = False

    async def __aenter__(self):
        await self.initialize()
        self._initialized = True
        return self

    @abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class StreamChunk(BaseModel):
    state: Literal["DOING", "DONE", "END"]
    content: Union[str, Dict]

