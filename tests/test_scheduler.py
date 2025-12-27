"""Tests for Task Scheduler."""

import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Any

from evo2.tasks.base import Task, TaskStatus, TaskPriority
from evo2.tasks.queue import TaskQueue
from evo2.tasks.scheduler import TaskScheduler, SchedulerConfig


class TestTaskScheduler:
    """Test suite for TaskScheduler."""

    def test_scheduler_initialization(self):
        """Test TaskScheduler initialization."""
        config = SchedulerConfig(max_concurrent_tasks=2, poll_interval=0.1)
        scheduler = TaskScheduler(config)
        
        assert scheduler.max_concurrent_tasks == 2
        assert scheduler.is_running() is False
        assert scheduler.get_pending_count() == 0
        assert scheduler.get_running_count() == 0

    def test_scheduler_add_task(self):
        """Test adding tasks to scheduler."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "completed"
        
        scheduler = TaskScheduler()
        task = TestTask("test_1", TaskPriority.HIGH)
        
        scheduler.add_task(task)
        assert scheduler.get_pending_count() == 1
        assert scheduler.get_task_by_id("test_1") is task

    def test_scheduler_start_stop(self):
        """Test starting and stopping the scheduler."""
        scheduler = TaskScheduler()
        
        assert scheduler.is_running() is False
        
        scheduler.start()
        assert scheduler.is_running() is True
        
        scheduler.stop()
        assert scheduler.is_running() is False

    def test_scheduler_task_execution(self):
        """Test that scheduler executes tasks."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "test_result"
        
        scheduler = TaskScheduler(SchedulerConfig(max_concurrent_tasks=1, poll_interval=0.01))
        task = TestTask("test_1", TaskPriority.HIGH)
        
        scheduler.add_task(task)
        scheduler.start()
        
        # Wait for task to complete
        import time
        for i in range(100):  # Wait up to 1 second
            time.sleep(0.01)
            if task.is_completed():
                break
        
        scheduler.stop()
        
        print(f"Task status: {task.status}")
        print(f"Task result: {task.get_result()}")
        
        assert task.is_completed()
        assert task.get_result() == "test_result"

    def test_scheduler_concurrent_execution(self):
        """Test concurrent task execution."""
        
        import time
        
        class SlowTask(Task):
            def __init__(self, task_id: str, delay: float):
                super().__init__(task_id, TaskPriority.MEDIUM)
                self.delay = delay
            
            def execute(self) -> Any:
                time.sleep(self.delay)
                return f"completed_{self.task_id}"
        
        scheduler = TaskScheduler(SchedulerConfig(max_concurrent_tasks=2, poll_interval=0.05))
        
        # Add tasks that take different amounts of time
        task1 = SlowTask("fast", 0.1)
        task2 = SlowTask("medium", 0.2)
        task3 = SlowTask("slow", 0.3)
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        scheduler.add_task(task3)
        
        scheduler.start()
        
        # Wait for all tasks to complete
        time.sleep(0.8)
        
        scheduler.stop()
        
        # All tasks should be completed
        assert task1.is_completed()
        assert task2.is_completed()
        assert task3.is_completed()
        
        assert task1.get_result() == "completed_fast"
        assert task2.get_result() == "completed_medium"
        assert task3.get_result() == "completed_slow"

    def test_scheduler_task_failure_handling(self):
        """Test that scheduler handles task failures properly."""
        
        class FailingTask(Task):
            def execute(self) -> Any:
                raise ValueError("Task execution failed")
        
        scheduler = TaskScheduler(SchedulerConfig(max_concurrent_tasks=1, poll_interval=0.1))
        task = FailingTask("failing_task", TaskPriority.HIGH)
        
        scheduler.add_task(task)
        scheduler.start()
        
        # Wait for task to fail
        import time
        time.sleep(0.5)
        
        scheduler.stop()
        
        assert task.is_failed()
        assert isinstance(task.get_error(), ValueError)

    def test_scheduler_task_cancellation(self):
        """Test task cancellation functionality."""
        
        class LongRunningTask(Task):
            def execute(self) -> Any:
                import time
                time.sleep(10)  # Long running task
                return "should_not_complete"
        
        scheduler = TaskScheduler(SchedulerConfig(max_concurrent_tasks=1, poll_interval=0.1))
        task = LongRunningTask("long_task", TaskPriority.LOW)
        
        scheduler.add_task(task)
        
        # Cancel the task before it starts
        success = scheduler.cancel_task("long_task")
        assert success is True
        assert task.is_cancelled()

    def test_scheduler_statistics(self):
        """Test scheduler statistics reporting."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "done"
        
        scheduler = TaskScheduler()
        
        # Add tasks with different priorities
        high_task = TestTask("high", TaskPriority.HIGH)
        medium_task = TestTask("medium", TaskPriority.MEDIUM)
        low_task = TestTask("low", TaskPriority.LOW)
        
        scheduler.add_task(high_task)
        scheduler.add_task(medium_task)
        scheduler.add_task(low_task)
        
        stats = scheduler.get_statistics()
        
        assert stats["pending_tasks"] == 3
        assert stats["running_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["failed_tasks"] == 0
        assert stats["total_tasks"] == 3

    def test_scheduler_config_validation(self):
        """Test scheduler configuration validation."""
        # Valid config
        config = SchedulerConfig(max_concurrent_tasks=4, poll_interval=0.5)
        assert config.max_concurrent_tasks == 4
        assert config.poll_interval == 0.5
        
        # Invalid max_concurrent_tasks
        with pytest.raises(ValueError):
            SchedulerConfig(max_concurrent_tasks=0)
        
        with pytest.raises(ValueError):
            SchedulerConfig(max_concurrent_tasks=-1)
        
        # Invalid poll_interval
        with pytest.raises(ValueError):
            SchedulerConfig(poll_interval=0)
        
        with pytest.raises(ValueError):
            SchedulerConfig(poll_interval=-1)
