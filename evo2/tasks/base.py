"""Base Task classes for Evo2 task management."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import uuid
import logging


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Enumeration of task priorities with numeric values for ordering."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        """Allow priority comparison for sorting."""
        if isinstance(other, TaskPriority):
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other):
        """Allow priority comparison for sorting."""
        if isinstance(other, TaskPriority):
            return self.value > other.value
        return NotImplemented


class Task(ABC):
    """Abstract base class for all tasks in Evo2.
    
    Tasks represent units of work that can be executed by agents.
    Each task has a priority, status, and can store results and metadata.
    """
    
    def __init__(
        self,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a new task.
        
        Args:
            task_id: Unique identifier for the task. If None, generates UUID.
            priority: Priority level for task scheduling.
            metadata: Additional metadata about the task.
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.priority = priority
        self.metadata = metadata or {}
        self.status = TaskStatus.PENDING
        
        # Timestamps
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # Results and errors
        self._result: Optional[Any] = None
        self._error: Optional[Exception] = None
        
        self.logger = logging.getLogger(f"Task.{self.__class__.__name__}")
    
    @abstractmethod
    def execute(self) -> Any:
        """Execute the task and return the result.
        
        This method must be implemented by subclasses to define
        the specific work the task performs.
        
        Returns:
            The result of task execution.
            
        Raises:
            Exception: If task execution fails.
        """
        pass
    
    def set_status(self, status: TaskStatus) -> None:
        """Set the task status and update timestamps.
        
        Args:
            status: The new status to set.
        """
        old_status = self.status
        self.status = status
        
        # Update timestamps based on status transitions
        if status == TaskStatus.RUNNING and old_status != TaskStatus.RUNNING:
            self.started_at = datetime.utcnow()
            self.logger.info(f"Task {self.task_id} started")
        elif status == TaskStatus.COMPLETED:
            self.completed_at = datetime.utcnow()
            self.logger.info(f"Task {self.task_id} completed")
        elif status == TaskStatus.FAILED:
            self.completed_at = datetime.utcnow()
            self.logger.error(f"Task {self.task_id} failed")
        elif status == TaskStatus.CANCELLED:
            self.completed_at = datetime.utcnow()
            self.logger.info(f"Task {self.task_id} cancelled")
    
    def set_result(self, result: Any) -> None:
        """Set the result of task execution.
        
        Args:
            result: The result to store.
        """
        self._result = result
    
    def set_error(self, error: Exception) -> None:
        """Set the error if task execution failed.
        
        Args:
            error: The exception that occurred.
        """
        self._error = error
        self.set_status(TaskStatus.FAILED)
    
    def get_result(self) -> Optional[Any]:
        """Get the result of task execution.
        
        Returns:
            The task result, or None if not completed or failed.
        """
        return self._result
    
    def get_error(self) -> Optional[Exception]:
        """Get the error if task execution failed.
        
        Returns:
            The exception that occurred, or None if no error.
        """
        return self._error
    
    def get_created_time(self) -> datetime:
        """Get the task creation time.
        
        Returns:
            The creation timestamp.
        """
        return self.created_at
    
    def get_started_time(self) -> Optional[datetime]:
        """Get the task start time.
        
        Returns:
            The start timestamp, or None if not started.
        """
        return self.started_at
    
    def get_completed_time(self) -> Optional[datetime]:
        """Get the task completion time.
        
        Returns:
            The completion timestamp, or None if not completed.
        """
        return self.completed_at
    
    def get_duration(self) -> Optional[float]:
        """Get the task execution duration in seconds.
        
        Returns:
            Duration in seconds, or None if not completed.
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def is_pending(self) -> bool:
        """Check if task is pending.
        
        Returns:
            True if task status is PENDING.
        """
        return self.status == TaskStatus.PENDING
    
    def is_running(self) -> bool:
        """Check if task is currently running.
        
        Returns:
            True if task status is RUNNING.
        """
        return self.status == TaskStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if task is completed.
        
        Returns:
            True if task status is COMPLETED.
        """
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if task failed.
        
        Returns:
            True if task status is FAILED.
        """
        return self.status == TaskStatus.FAILED
    
    def is_cancelled(self) -> bool:
        """Check if task was cancelled.
        
        Returns:
            True if task status is CANCELLED.
        """
        return self.status == TaskStatus.CANCELLED
    
    def is_finished(self) -> bool:
        """Check if task is finished (completed, failed, or cancelled).
        
        Returns:
            True if task is in any final state.
        """
        return self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    
    def cancel(self) -> bool:
        """Cancel the task if it hasn't started.
        
        Returns:
            True if task was cancelled, False if already running or finished.
        """
        if self.status == TaskStatus.PENDING:
            self.set_status(TaskStatus.CANCELLED)
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation.
        
        Returns:
            Dictionary containing task information.
        """
        return {
            "task_id": self.task_id,
            "priority": self.priority.value,
            "status": self.status.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.get_duration(),
            "has_result": self._result is not None,
            "has_error": self._error is not None,
        }
    
    def __str__(self) -> str:
        """String representation of task."""
        return f"Task({self.task_id}, {self.priority.value}, {self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of task."""
        return (f"Task(id={self.task_id}, priority={self.priority.value}, "
                f"status={self.status.value}, created={self.created_at.isoformat()})")
