"""Nuwa 闹钟模块，提供定时提醒功能。

本模块提供闹钟管理器和相关工具，允许 ReActAgent 为自己或用户设置定时提醒。
遵循单例模式（P-008），通过全局 alarm_manager 实例提供统一管理。
设计规范：模块化（C-001）、错误处理（C-012）、异步任务管理（C-010）。
"""

import asyncio
import logging
from datetime import datetime
from typing import Literal, Callable, Optional, Dict, Any
from dataclasses import dataclass

from .tool import Tool, ToolEntity, ToolObjectParameter, ToolParameter
from .react_agent import ReasoningActingAgent

logger = logging.getLogger(__name__)


@dataclass
class AlarmTask:
    """闹钟任务数据类。

    封装单个闹钟任务的所有信息，便于管理和传递。
    遵循数据类设计规范（C-006），使用 dataclass 简化代码。

    Attributes:
        alarm_id: 闹钟唯一标识符。
        time: 提醒时间（datetime 对象）。
        remindee: 被提醒的人，可选 "oneself"（自己）或 "user"（用户）。
        reminder: 提醒信息内容。
        agent: 关联的 ReasoningActingAgent 实例（仅当 remindee 为 "oneself" 时有效）。
        callback: 触发时的回调函数，接收提醒信息作为参数。
        task: 异步任务对象，用于控制闹钟调度。
    """
    alarm_id: str
    time: datetime
    remindee: Literal["oneself", "user"]
    reminder: str
    agent: Optional[ReasoningActingAgent] = None
    callback: Optional[Callable[[str], Any]] = None
    task: Optional[asyncio.Task] = None


class AlarmManager:
    """闹钟管理器，负责闹钟的创建、调度、触发和取消。

    遵循单例模式（P-008），通过模块级全局实例提供统一访问。
    使用异步锁保证线程安全，避免竞争条件。
    设计规范：类职责明确（C-006）、错误处理（C-012）、日志记录（C-016）。

    Attributes:
        tasks: 活跃闹钟任务字典，键为 alarm_id，值为 AlarmTask。
        _next_id: 内部计数器，用于生成唯一 ID。
        _lock: 异步锁，保护共享资源。
    """

    def __init__(self):
        """初始化闹钟管理器。"""
        self.tasks: Dict[str, AlarmTask] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()

    def _generate_id(self) -> str:
        """生成唯一的闹钟 ID。

        格式为 "alarm_{数字}"，数字递增。
        遵循命名规范（D-005），ID 具有可读性。

        Returns:
            唯一闹钟 ID 字符串。
        """
        alarm_id = f"alarm_{self._next_id}"
        self._next_id += 1
        return alarm_id

    async def set_alarm_for_oneself(
        self,
        time: datetime,
        reminder: str,
        agent: ReasoningActingAgent,
        callback: Optional[Callable[[str], Any]] = None
    ) -> str:
        """为 agent 自己设置闹钟。

        闹钟触发时将唤醒指定的 agent（通过回调或日志）。
        遵循错误处理规范（C-012），对过去时间进行警告并立即触发。

        Args:
            time: 提醒时间。
            reminder: 提醒信息。
            agent: 相关的 ReasoningActingAgent 实例。
            callback: 唤醒时的回调函数，接收提醒信息作为参数，可选。

        Returns:
            闹钟 ID，可用于后续取消。
        """
        async with self._lock:
            alarm_id = self._generate_id()
            task = AlarmTask(
                alarm_id=alarm_id,
                time=time,
                remindee="oneself",
                reminder=reminder,
                agent=agent,
                callback=callback
            )

            # 计算延迟时间
            now = datetime.now()
            if time <= now:
                logger.warning(f"闹钟时间 {time} 已过，立即触发")
                delay = 0
            else:
                delay = (time - now).total_seconds()

            # 创建异步任务
            task.task = asyncio.create_task(self._schedule_alarm(task, delay))
            self.tasks[alarm_id] = task

            logger.debug(f"已设置闹钟 {alarm_id}: {time} - {reminder}")
            return alarm_id

    async def set_alarm_for_user(
        self,
        time: datetime,
        reminder: str,
        callback: Optional[Callable[[str], Any]] = None
    ) -> str:
        """为用户设置闹钟（预留接口）。

        当前仅记录日志，实际应用中应通过其他渠道（如邮件、推送）通知用户。
        遵循接口设计规范（C-008），预留扩展点。

        Args:
            time: 提醒时间。
            reminder: 提醒信息。
            callback: 唤醒时的回调函数，可选。

        Returns:
            闹钟 ID。
        """
        async with self._lock:
            alarm_id = self._generate_id()
            task = AlarmTask(
                alarm_id=alarm_id,
                time=time,
                remindee="user",
                reminder=reminder,
                agent=None,
                callback=callback
            )

            # 计算延迟时间
            now = datetime.now()
            if time <= now:
                logger.warning(f"用户闹钟时间 {time} 已过，立即触发")
                delay = 0
            else:
                delay = (time - now).total_seconds()

            # 创建异步任务
            task.task = asyncio.create_task(self._schedule_alarm(task, delay))
            self.tasks[alarm_id] = task

            logger.debug(f"已设置用户闹钟 {alarm_id}: {time} - {reminder}")
            return alarm_id

    async def _schedule_alarm(self, alarm_task: AlarmTask, delay: float):
        """调度闹钟任务。

        等待指定延迟后触发闹钟，处理取消和异常。
        遵循异步任务管理规范（C-010），妥善处理 CancelledError。

        Args:
            alarm_task: 闹钟任务对象。
            delay: 延迟秒数。
        """
        try:
            if delay > 0:
                await asyncio.sleep(delay)

            # 触发闹钟
            await self._trigger_alarm(alarm_task)

        except asyncio.CancelledError:
            logger.debug(f"闹钟 {alarm_task.alarm_id} 已被取消")
        except Exception as e:
            logger.error(f"闹钟 {alarm_task.alarm_id} 触发失败: {e}")
        finally:
            # 从任务列表中移除
            async with self._lock:
                self.tasks.pop(alarm_task.alarm_id, None)

    async def _trigger_alarm(self, alarm_task: AlarmTask):
        """触发闹钟。

        根据被提醒人类型分发到不同的处理函数。
        遵循日志记录规范（C-016），记录触发事件。

        Args:
            alarm_task: 闹钟任务对象。
        """
        logger.debug(f"闹钟触发: {alarm_task.alarm_id} - {alarm_task.reminder}")

        if alarm_task.remindee == "oneself":
            await self._wake_up_agent(alarm_task)
        else:  # "user"
            await self._notify_user(alarm_task)

    async def _wake_up_agent(self, alarm_task: AlarmTask):
        """唤醒 agent 自身。

        执行回调函数（如果提供），否则仅记录日志。
        遵循错误处理规范（C-012），回调异常被捕获并记录。

        Args:
            alarm_task: 闹钟任务对象。
        """
        reminder_msg = f"⏰ 闹钟提醒: {alarm_task.reminder}"

        if alarm_task.callback:
            try:
                await alarm_task.callback(reminder_msg)
            except Exception as e:
                logger.error(f"闹钟回调执行失败: {e}")

        # 如果没有回调，至少记录日志
        logger.debug(f"Agent闹钟唤醒: {reminder_msg}")

    async def _notify_user(self, alarm_task: AlarmTask):
        """通知用户（预留接口）。

        当前仅记录日志，实际应用中应通过其他方式通知用户。
        遵循接口设计规范（C-008），为未来扩展保留。

        Args:
            alarm_task: 闹钟任务对象。
        """
        reminder_msg = f"⏰ 用户闹钟提醒: {alarm_task.reminder}"

        if alarm_task.callback:
            try:
                await alarm_task.callback(reminder_msg)
            except Exception as e:
                logger.error(f"用户闹钟回调执行失败: {e}")

        # 记录日志，实际应用中应该通过其他方式通知用户
        logger.debug(f"用户闹钟触发（预留接口）: {reminder_msg}")

    async def cancel_alarm(self, alarm_id: str) -> bool:
        """取消闹钟。

        如果闹钟存在且未触发，则取消其异步任务并从列表中移除。
        遵循幂等操作，即使闹钟不存在也返回 False。

        Args:
            alarm_id: 闹钟 ID。

        Returns:
            如果成功取消返回 True，否则返回 False。
        """
        async with self._lock:
            task = self.tasks.get(alarm_id)
            if task and task.task:
                task.task.cancel()
                self.tasks.pop(alarm_id, None)
                logger.debug(f"已取消闹钟 {alarm_id}")
                return True
            return False

    def list_alarms(self) -> list:
        """列出所有活跃的闹钟。

        返回字典列表，包含闹钟的基本信息，便于外部查看。
        遵循数据格式规范，使用 ISO 时间字符串便于序列化。

        Returns:
            闹钟信息字典列表。
        """
        return [
            {
                "alarm_id": task.alarm_id,
                "time": task.time.isoformat(),
                "remindee": task.remindee,
                "reminder": task.reminder,
                "status": "active"
            }
            for task in self.tasks.values()
        ]


# 全局闹钟管理器实例，遵循单例模式（P-008）
alarm_manager = AlarmManager()


async def get_alarm_tool(agent: ReasoningActingAgent) -> Tool:
    """获取闹钟工具。

    返回一个 Tool 实例，该工具可用于 ReasoningActingAgent 设置闹钟。
    遵循工具注册规范，通过装饰器模式（P-003）的变体动态创建工具。

    Args:
        agent: 调用此工具的 ReasoningActingAgent 实例。

    Returns:
        配置好的 Tool 对象。
    """

    async def set_alarm(time: str, remindee: Literal["oneself", "user"], reminder: str):
        """
        设置闹钟提醒。

        Args:
            time: ISO 格式的时间字符串，如：2026-01-01T20:25:56.847307。
            remindee: 被提醒的人，oneself（自己）或 user（用户）。
            reminder: 提醒信息。

        Returns:
            设置结果信息字典，包含成功状态、消息、闹钟 ID 和计划时间。
        """
        try:
            # 解析 ISO 时间字符串
            alarm_time = datetime.fromisoformat(time)
        except ValueError:
            return {"success": False, "message": f"无效的时间格式: {time}"}

        try:
            if remindee == "oneself":
                alarm_id = await alarm_manager.set_alarm_for_oneself(
                    time=alarm_time,
                    reminder=reminder,
                    agent=agent
                )
                return {
                    "success": True,
                    "message": f"已为自己设置闹钟: {reminder}",
                    "alarm_id": alarm_id,
                    "scheduled_time": alarm_time.isoformat()
                }
            else:  # "user"
                alarm_id = await alarm_manager.set_alarm_for_user(
                    time=alarm_time,
                    reminder=reminder
                )
                return {
                    "success": True,
                    "message": f"已为用户设置闹钟: {reminder}",
                    "alarm_id": alarm_id,
                    "scheduled_time": alarm_time.isoformat()
                }
        except Exception as e:
            logger.error(f"设置闹钟失败: {e}")
            return {"success": False, "message": f"设置闹钟失败: {str(e)}"}

    return Tool(
        func=set_alarm,
        entity=ToolEntity(
            name="set_alarm",
            description="设置闹钟提醒",
            parameters=ToolObjectParameter(
                type="object",
                properties={
                    "time": ToolParameter(
                        type="string",
                        description="闹钟或提醒的时间，标准ISO时间格式字符串，如：2026-01-01T20:25:56.847307",
                    ),
                    "remindee": ToolParameter(
                        type="string",
                        description="被提醒的人，可选oneself（自己）或user（用户）",
                        enum=["oneself", "user"],
                    ),
                    "reminder": ToolParameter(type="string", description="备忘信息"),
                },
            ),
        ),
    )
