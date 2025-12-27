"""Tests for Task management components."""

import pytest
from abc import ABC
from enum import Enum
from typing import Any, Dict
from datetime import datetime

from evo2.tasks.base import Task, TaskStatus, TaskPriority
from evo2.tasks.queue import TaskQueue


class TestTask:
    """Test suite for Task base class."""

    def test_task_is_abstract(self):
        """Test that Task cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Task()

    def test_task_subclass_requires_methods(self):
        """Test that Task subclasses must implement abstract methods."""
        
        class IncompleteTask(Task):
            """Incomplete task missing required methods."""
            pass
        
        with pytest.raises(TypeError):
            IncompleteTask()

    def test_complete_task_subclass(self):
        """Test that a complete task subclass can be instantiated."""
        
        class CompleteTask(Task):
            """Complete task implementing all required methods."""
            
            def execute(self) -> Any:
                """Execute the task."""
                return "completed"
        
        task = CompleteTask(task_id="test_1", priority=TaskPriority.MEDIUM)
        assert isinstance(task, Task)
        assert task.task_id == "test_1"
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING

    def test_task_state_transitions(self):
        """Test task state transitions."""
        
        class TestTask(Task):
            """Test task implementation."""
            
            def execute(self) -> Any:
                self.set_status(TaskStatus.RUNNING)
                result = "test_result"
                self.set_status(TaskStatus.COMPLETED)
                self.set_result(result)
                return result
        
        task = TestTask(task_id="test_2", priority=TaskPriority.HIGH)
        
        # Initial state
        assert task.status == TaskStatus.PENDING
        assert task.get_created_time() is not None
        
        # Execute task
        result = task.execute()
        
        # Final state
        assert task.status == TaskStatus.COMPLETED
        assert task.get_result() == "test_result"
        assert task.get_completed_time() is not None

    def test_task_priority_ordering(self):
        """Test that task priorities are correctly ordered."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "done"
        
        low_task = TestTask("low", TaskPriority.LOW)
        medium_task = TestTask("medium", TaskPriority.MEDIUM)
        high_task = TestTask("high", TaskPriority.HIGH)
        critical_task = TestTask("critical", TaskPriority.CRITICAL)
        
        assert critical_task.priority > high_task.priority > medium_task.priority > low_task.priority


class TestTaskQueue:
    """Test suite for TaskQueue."""

    def test_task_queue_initialization(self):
        """Test TaskQueue initialization."""
        queue = TaskQueue()
        assert queue.is_empty()
        assert queue.size() == 0

    def test_task_queue_add_and_get(self):
        """Test adding and getting tasks from queue."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "done"
        
        queue = TaskQueue()
        task = TestTask("test_1", TaskPriority.MEDIUM)
        
        queue.add_task(task)
        assert queue.size() == 1
        assert not queue.is_empty()

    def test_task_queue_priority_ordering(self):
        """Test that tasks are retrieved in priority order."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "done"
        
        queue = TaskQueue()
        
        # Add tasks in random order
        low_task = TestTask("low", TaskPriority.LOW)
        high_task = TestTask("high", TaskPriority.HIGH)
        medium_task = TestTask("medium", TaskPriority.MEDIUM)
        critical_task = TestTask("critical", TaskPriority.CRITICAL)
        
        queue.add_task(low_task)
        queue.add_task(high_task)
        queue.add_task(medium_task)
        queue.add_task(critical_task)
        
        # Should retrieve in priority order
        assert queue.get_next_task().task_id == "critical"
        assert queue.get_next_task().task_id == "high"
        assert queue.get_next_task().task_id == "medium"
        assert queue.get_next_task().task_id == "low"

    def test_task_queue_get_by_id(self):
        """Test retrieving tasks by ID."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "done"
        
        queue = TaskQueue()
        task = TestTask("test_1", TaskPriority.MEDIUM)
        
        queue.add_task(task)
        retrieved_task = queue.get_task_by_id("test_1")
        
        assert retrieved_task is task
        assert queue.get_task_by_id("nonexistent") is None

    def test_task_queue_remove_task(self):
        """Test removing tasks from queue."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "done"
        
        queue = TaskQueue()
        task1 = TestTask("test_1", TaskPriority.MEDIUM)
        task2 = TestTask("test_2", TaskPriority.HIGH)
        
        queue.add_task(task1)
        queue.add_task(task2)
        
        assert queue.size() == 2
        
        removed = queue.remove_task("test_1")
        assert removed is True
        assert queue.size() == 1
        assert queue.get_task_by_id("test_1") is None
        
        removed = queue.remove_task("nonexistent")
        assert removed is False
