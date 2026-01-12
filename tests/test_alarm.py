import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import logging

from src.nuwa.scheduling_tools import AlarmManager, AlarmTask, alarm_manager, get_alarm_tool
from src.nuwa.react_agent import ReasoningActingAgent
from src.nuwa.tool import Tool

logger = logging.getLogger(__name__)


class TestAlarmManager:
    """测试 AlarmManager 类"""

    def test_initialization(self):
        """测试初始化"""
        manager = AlarmManager()
        assert manager.tasks == {}
        assert manager._next_id == 1
        assert manager._lock is not None

    def test_generate_id(self):
        """测试生成唯一ID"""
        manager = AlarmManager()
        id1 = manager._generate_id()
        assert id1 == "alarm_1"
        id2 = manager._generate_id()
        assert id2 == "alarm_2"
        assert manager._next_id == 3

    @pytest.mark.asyncio
    async def test_set_alarm_for_oneself(self):
        """测试为agent自己设置闹钟"""
        manager = AlarmManager()
        mock_agent = Mock(spec=ReasoningActingAgent)
        mock_callback = AsyncMock()

        # 设置未来时间
        future_time = datetime.now() + timedelta(seconds=0.1)
        alarm_id = await manager.set_alarm_for_oneself(
            time=future_time,
            reminder="测试提醒",
            agent=mock_agent,
            callback=mock_callback
        )

        assert alarm_id.startswith("alarm_")
        assert alarm_id in manager.tasks
        task = manager.tasks[alarm_id]
        assert task.remindee == "oneself"
        assert task.reminder == "测试提醒"
        assert task.agent is mock_agent
        assert task.callback is mock_callback
        assert task.task is not None

        # 等待闹钟触发
        await asyncio.sleep(0.15)
        # 回调应该被调用
        mock_callback.assert_called_once()
        # 任务应该已从列表中移除
        assert alarm_id not in manager.tasks

    @pytest.mark.asyncio
    async def test_set_alarm_for_user(self):
        """测试为用户设置闹钟"""
        manager = AlarmManager()
        mock_callback = AsyncMock()

        future_time = datetime.now() + timedelta(seconds=0.1)
        alarm_id = await manager.set_alarm_for_user(
            time=future_time,
            reminder="用户提醒",
            callback=mock_callback
        )

        assert alarm_id.startswith("alarm_")
        task = manager.tasks[alarm_id]
        assert task.remindee == "user"
        assert task.agent is None
        assert task.callback is mock_callback

        await asyncio.sleep(0.15)
        mock_callback.assert_called_once()
        assert alarm_id not in manager.tasks

    @pytest.mark.asyncio
    async def test_set_alarm_past_time(self):
        """测试设置过去时间的闹钟（立即触发）"""
        manager = AlarmManager()
        mock_callback = AsyncMock()
        past_time = datetime.now() - timedelta(seconds=10)

        with patch('src.nuwa.scheduling_tools.logger.warning') as mock_warning:
            alarm_id = await manager.set_alarm_for_oneself(
                time=past_time,
                reminder="过去时间",
                agent=Mock(spec=ReasoningActingAgent),
                callback=mock_callback
            )

        # 警告应该被记录
        mock_warning.assert_called_once()
        # 由于是过去时间，应该立即触发
        await asyncio.sleep(0.05)  # 短暂等待确保异步任务完成
        mock_callback.assert_called_once()
        assert alarm_id not in manager.tasks

    @pytest.mark.asyncio
    async def test_cancel_alarm(self):
        """测试取消闹钟"""
        manager = AlarmManager()
        future_time = datetime.now() + timedelta(seconds=1)
        alarm_id = await manager.set_alarm_for_oneself(
            time=future_time,
            reminder="可取消的闹钟",
            agent=Mock(spec=ReasoningActingAgent)
        )

        assert alarm_id in manager.tasks
        # 取消闹钟
        result = await manager.cancel_alarm(alarm_id)
        assert result is True
        assert alarm_id not in manager.tasks

        # 再次取消同一闹钟应返回 False
        result = await manager.cancel_alarm(alarm_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_alarm(self):
        """测试取消不存在的闹钟"""
        manager = AlarmManager()
        result = await manager.cancel_alarm("nonexistent_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_alarms(self):
        """测试列出所有闹钟"""
        manager = AlarmManager()
        future_time = datetime.now() + timedelta(minutes=5)
        
        # 设置两个闹钟
        alarm_id1 = await manager.set_alarm_for_oneself(
            time=future_time,
            reminder="第一个闹钟",
            agent=Mock(spec=ReasoningActingAgent)
        )
        alarm_id2 = await manager.set_alarm_for_user(
            time=future_time + timedelta(minutes=1),
            reminder="第二个闹钟"
        )

        alarms = manager.list_alarms()
        assert len(alarms) == 2
        
        # 检查格式
        for alarm in alarms:
            assert "alarm_id" in alarm
            assert "time" in alarm
            assert "remindee" in alarm
            assert "reminder" in alarm
            assert "status" in alarm
            assert alarm["status"] == "active"
        
        # 确认特定闹钟存在
        alarm_ids = [a["alarm_id"] for a in alarms]
        assert alarm_id1 in alarm_ids
        assert alarm_id2 in alarm_ids

    @pytest.mark.asyncio
    async def test_concurrent_alarm_setting(self):
        """测试并发设置闹钟（锁机制）"""
        manager = AlarmManager()
        future_time = datetime.now() + timedelta(seconds=0.5)
        
        async def set_alarm():
            return await manager.set_alarm_for_oneself(
                time=future_time,
                reminder="并发测试",
                agent=Mock(spec=ReasoningActingAgent)
            )
        
        # 同时创建多个任务
        tasks = [set_alarm() for _ in range(5)]
        alarm_ids = await asyncio.gather(*tasks)
        
        # 所有闹钟ID应该唯一
        assert len(set(alarm_ids)) == 5
        assert len(manager.tasks) == 5

    @pytest.mark.asyncio
    async def test_alarm_without_callback(self):
        """测试没有回调函数的闹钟"""
        manager = AlarmManager()
        future_time = datetime.now() + timedelta(seconds=0.1)
        
        with patch('src.nuwa.scheduling_tools.logger.debug') as mock_info:
            alarm_id = await manager.set_alarm_for_oneself(
                time=future_time,
                reminder="无回调测试",
                agent=Mock(spec=ReasoningActingAgent),
                callback=None
            )
            await asyncio.sleep(0.15)
            
            # 应该记录日志
            assert any("Agent闹钟唤醒" in str(call) for call in mock_info.call_args_list)
        
        assert alarm_id not in manager.tasks

    @pytest.mark.asyncio
    async def test_trigger_alarm_exception(self):
        """测试闹钟触发时回调抛出异常"""
        manager = AlarmManager()
        future_time = datetime.now() + timedelta(seconds=0.1)
        
        async def failing_callback(msg):
            raise ValueError("回调失败")
        
        with patch('src.nuwa.scheduling_tools.logger.error') as mock_error:
            alarm_id = await manager.set_alarm_for_oneself(
                time=future_time,
                reminder="异常测试",
                agent=Mock(spec=ReasoningActingAgent),
                callback=failing_callback
            )
            await asyncio.sleep(0.15)
            
            # 应该记录错误
            mock_error.assert_called()
            assert "回调执行失败" in str(mock_error.call_args)
        
        assert alarm_id not in manager.tasks


class TestGetAlarmTool:
    """测试 get_alarm_tool 函数"""

    @pytest.mark.asyncio
    async def test_get_alarm_tool(self):
        """测试工具创建"""
        mock_agent = Mock(spec=ReasoningActingAgent)
        tool = await get_alarm_tool(mock_agent)
        
        assert isinstance(tool, Tool)
        assert tool.entity.name == "set_alarm"
        assert tool.entity.description == "设置闹钟提醒"
        assert "time" in tool.entity.parameters.properties
        assert "remindee" in tool.entity.parameters.properties
        assert "reminder" in tool.entity.parameters.properties

    @pytest.mark.asyncio
    async def test_alarm_tool_set_alarm_oneself(self):
        """通过工具为agent自己设置闹钟"""
        mock_agent = Mock(spec=ReasoningActingAgent)
        tool = await get_alarm_tool(mock_agent)
        
        # 设置未来时间
        future_time = datetime.now() + timedelta(seconds=0.5)
        time_str = future_time.isoformat()
        
        result = await tool.func(
            time=time_str,
            remindee="oneself",
            reminder="工具测试"
        )
        
        assert result["success"] is True
        assert "已为自己设置闹钟" in result["message"]
        assert "alarm_id" in result
        assert "scheduled_time" in result
        
        # 验证闹钟确实被设置
        alarm_id = result["alarm_id"]
        assert alarm_id in alarm_manager.tasks
        
        # 清理
        await alarm_manager.cancel_alarm(alarm_id)

    @pytest.mark.asyncio
    async def test_alarm_tool_set_alarm_user(self):
        """通过工具为用户设置闹钟"""
        mock_agent = Mock(spec=ReasoningActingAgent)
        tool = await get_alarm_tool(mock_agent)
        
        future_time = datetime.now() + timedelta(seconds=0.5)
        time_str = future_time.isoformat()
        
        result = await tool.func(
            time=time_str,
            remindee="user",
            reminder="用户工具测试"
        )
        
        assert result["success"] is True
        assert "已为用户设置闹钟" in result["message"]
        alarm_id = result["alarm_id"]
        assert alarm_id in alarm_manager.tasks
        
        # 清理
        await alarm_manager.cancel_alarm(alarm_id)

    @pytest.mark.asyncio
    async def test_alarm_tool_invalid_time(self):
        """测试无效时间格式"""
        mock_agent = Mock(spec=ReasoningActingAgent)
        tool = await get_alarm_tool(mock_agent)
        
        result = await tool.func(
            time="invalid-time-format",
            remindee="oneself",
            reminder="无效时间"
        )
        
        assert result["success"] is False
        assert "无效的时间格式" in result["message"]

    @pytest.mark.asyncio
    async def test_alarm_tool_exception_handling(self):
        """测试工具异常处理"""
        mock_agent = Mock(spec=ReasoningActingAgent)
        tool = await get_alarm_tool(mock_agent)
        
        # 模拟设置闹钟时抛出异常
        with patch.object(alarm_manager, 'set_alarm_for_oneself', side_effect=Exception("模拟异常")):
            result = await tool.func(
                time="2025-01-01T12:00:00",
                remindee="oneself",
                reminder="异常测试"
            )
            
            assert result["success"] is False
            assert "设置闹钟失败" in result["message"]


class TestGlobalAlarmManager:
    """测试全局 alarm_manager 实例"""

    @pytest.mark.asyncio
    async def test_global_manager_singleton(self):
        """测试全局实例是单例"""
        from src.nuwa.scheduling_tools import alarm_manager as gm1
        from src.nuwa.scheduling_tools import alarm_manager as gm2
        assert gm1 is gm2

    @pytest.mark.asyncio
    async def test_global_manager_functionality(self):
        """测试全局实例功能"""
        future_time = datetime.now() + timedelta(seconds=0.2)
        alarm_id = await alarm_manager.set_alarm_for_oneself(
            time=future_time,
            reminder="全局实例测试",
            agent=Mock(spec=ReasoningActingAgent)
        )
        
        assert alarm_id in alarm_manager.tasks
        await asyncio.sleep(0.25)
        assert alarm_id not in alarm_manager.tasks


if __name__ == "__main__":
    pytest.main([__file__])
