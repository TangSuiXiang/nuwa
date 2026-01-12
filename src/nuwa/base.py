"""Nuwa 基础层模块，定义核心抽象接口和数据结构。

本模块提供了 Nuwa 框架的基石，包括抽象基类、消息管理接口和节点抽象。
遵循设计原则：模块化（C-001）、抽象接口（C-004）、策略模式（P-002）和抽象工厂模式（P-001）。
"""

from abc import ABC, abstractmethod
from typing import List, Union, AsyncGenerator, Any, Literal, Dict
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam


class StreamChunk(BaseModel):
    """表示流式处理中的输入/输出数据块。

    遵循数据类设计规范（C-006），使用 Pydantic 模型确保类型安全。
    用于在节点之间传递带有状态标识的数据。

    Attributes:
        state: 数据块状态，取值为 "DOING"（处理中）、"DONE"（完成）或 "END"（结束）。
        content: 数据内容，可以是字符串或字典。
    """
    state: Literal["DOING", "DONE", "END"]
    content: Union[str, Dict]


class ConversationStorage(ABC):
    """对话存储抽象基类，定义消息持久化的统一接口。

    遵循抽象工厂模式（P-001），允许不同的存储后端实现（如向量存储、内存存储等）无缝替换。
    设计规范：接口设计（C-008）、类型提示（D-014）、异步接口（C-010）。

    Methods:
        get_messages: 检索会话历史消息。
        save_messages: 保存消息到存储后端。
        clear_messages: 清空指定会话的历史消息。
    """

    @abstractmethod
    async def get_messages(
        self, session_id: str, user_input: str = ""
    ) -> List[ChatCompletionMessageParam]:
        """检索与会话相关的历史消息。

        Args:
            session_id: 会话标识符，用于隔离不同用户的对话历史。
            user_input: 用户输入（可选），用于语义检索相似历史。

        Returns:
            消息列表，格式符合 OpenAI ChatCompletion 接口要求。

        Raises:
            ValueError: 当 session_id 无效时。
            ConnectionError: 当存储后端不可用时。
        """
        raise NotImplementedError

    @abstractmethod
    async def save_messages(self, session_id: str, messages: List[ChatCompletionMessageParam]):
        """保存消息到持久化存储。

        Args:
            session_id: 会话标识符。
            messages: 要保存的消息列表。

        Raises:
            ValueError: 当 messages 格式不正确时。
            ConnectionError: 当存储后端不可用时。
        """
        raise NotImplementedError

    @abstractmethod
    async def clear_messages(self, session_id: str):
        """清空指定会话的所有历史聊天消息。

        Args:
            session_id: 会话标识符。

        Raises:
            ConnectionError: 当存储后端不可用时。
        """
        raise NotImplementedError


class ProcessingNode(ABC):
    """处理节点抽象基类，所有处理组件的共同祖先。

    遵循策略模式（P-002），不同节点实现不同的处理策略。
    支持责任链模式（P-006）和 DAG 模式（P-007），通过依赖管理和链式操作形成处理流水线。
    设计规范：抽象基类（C-004）、单一职责（C-006）、组合优于继承（C-007）。

    Attributes:
        dependencies: 依赖节点列表，表示当前节点所依赖的前驱节点。

    Methods:
        get_dependency: 获取最终依赖节点（递归查找依赖链的源头）。
        __gt__: 重载大于运算符，实现节点链式连接（node1 > node2 表示 node1 的输出作为 node2 的输入）。
        run: 抽象处理方法，子类必须实现具体的处理逻辑。
    """

    def __init__(self):
        """初始化节点，依赖列表为空。"""
        self.dependencies: List["ProcessingNode"] = []

    def get_dependency(self) -> "ProcessingNode":
        """递归获取最终依赖节点。

        遵循 DAG 模式（P-007），沿依赖链向前查找，直到找到没有依赖的节点。

        Returns:
            依赖链的源头节点。
        """
        if self.dependencies:
            return self.dependencies[0].get_dependency()
        else:
            return self

    def __gt__(self, node: "ProcessingNode") -> "ProcessingNode":
        """重载大于运算符，建立节点间的依赖关系。

        实现责任链模式（P-006），允许通过 `node1 > node2` 语法将节点连接成处理链。
        设计规范：操作符重载需谨慎，确保类型安全（C-011）。

        Args:
            node: 要建立依赖的目标节点。

        Returns:
            目标节点（便于链式调用）。

        Raises:
            ValueError: 当 node 不是 ProcessingNode 实例时。
        """
        if not isinstance(node, ProcessingNode):  # Fixed type check
            raise ValueError("Can only compare with ProcessingNode instances")
        if self not in node.dependencies:
            node.dependencies.append(self)
        return self

    @abstractmethod
    async def run(
        self, input_chunks: Union[AsyncGenerator[StreamChunk, Any], Dict]
    ) -> AsyncGenerator[StreamChunk, Any]:
        """抽象处理方法，子类必须实现具体的处理逻辑。

        遵循模板方法模式（P-005），定义处理算法的骨架，子类实现具体步骤。
        支持流式处理，返回异步生成器以逐块产生输出。

        Args:
            input_chunks: 输入数据，可以是异步生成器或字典。异步生成器表示流式输入，
                          字典表示单次输入。

        Returns:
            输出数据的异步生成器，每个元素为 StreamChunk。

        Raises:
            NotImplementedError: 子类必须实现此方法。
        """
        raise NotImplementedError
