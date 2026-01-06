from abc import ABC, abstractmethod
from typing import List, Union, AsyncGenerator, Any, Literal, Dict
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam


class InputChunk(BaseModel):
    state: Literal["DOING", "DONE", "END"]
    content: Union[str, Dict]


class MessagesManager(ABC):
    @abstractmethod
    async def get_messages(
        self, session_id: str, user_input: str = ""
    ) -> List[ChatCompletionMessageParam]:
        raise NotImplementedError

    @abstractmethod
    async def save_messages(self, session_id: str, messages: List[ChatCompletionMessageParam]):
        raise NotImplementedError


class Node(ABC):
    def __init__(self):
        self.deps: List["Node"] = []

    def get_dep(self) -> "Node":
        if self.deps:
            return self.deps[0].get_dep()
        else:
            return self

    def __gt__(self, node: "Node") -> "Node":
        if not isinstance(node, Node):  # Fixed type check
            raise ValueError("Can only compare with Node instances")
        if self not in node.deps:
            node.deps.append(self)
        return self

    @abstractmethod
    async def run(
        self, input_chunks: Union[AsyncGenerator[InputChunk, Any], Dict]
    ) -> AsyncGenerator[InputChunk, Any]:
        raise NotImplementedError
