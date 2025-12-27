"""Task scheduler for managing parallel task execution."""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, Future
import logging

from .base import Task, TaskStatus
from .queue import TaskQueue


@dataclass
class SchedulerConfig:
    """Configuration for the task scheduler."""
    max_concurrent_tasks: int = 4
    poll_interval: float = 0.1  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")


class TaskScheduler:
    """Scheduler for managing parallel task execution.
    
    This scheduler manages a pool of worker threads to execute tasks
    concurrently, respecting priority and resource limits.
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """Initialize the task scheduler.
        
        Args:
            config: Optional scheduler configuration.
        """
        self.config = config or SchedulerConfig()
        self.max_concurrent_tasks = self.config.max_concurrent_tasks
        self.task_queue = TaskQueue()
        self.running_tasks: Dict[str, Future] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        self.logger = logging.getLogger("TaskScheduler")
    
    def add_task(self, task: Task) -> None:
        """Add a task to the scheduler.
        
        Args:
            task: The task to schedule.
        """
        if task.task_id in self.completed_tasks or task.task_id in self.failed_tasks:
            self.logger.warning(f"Task {task.task_id} has already been processed")
            return
        
        if task.task_id in self.running_tasks:
            self.logger.warning(f"Task {task.task_id} is already running")
            return
        
        self.task_queue.add_task(task)
        self.logger.info(f"Added task {task.task_id} to scheduler")
    
    def start(self) -> None:
        """Start the task scheduler."""
        if self._running:
            self.logger.warning("Scheduler is already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        self.logger.info("Task scheduler started")
    
    def stop(self) -> None:
        """Stop the task scheduler."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        
        # Cancel running tasks
        for task_id, future in self.running_tasks.items():
            if not future.done():
                future.cancel()
                self.logger.info(f"Cancelled task {task_id}")
        
        self._executor.shutdown(wait=True)
        self.logger.info("Task scheduler stopped")
    
    def is_running(self) -> bool:
        """Check if the scheduler is running.
        
        Returns:
            True if scheduler is running.
        """
        return self._running
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: The task ID to retrieve.
            
        Returns:
            The task if found, None otherwise.
        """
        return self.task_queue.get_task_by_id(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it hasn't started.
        
        Args:
            task_id: The ID of the task to cancel.
            
        Returns:
            True if task was cancelled, False otherwise.
        """
        # Try to cancel from queue first
        task = self.task_queue.get_task_by_id(task_id)
        if task and task.cancel():
            self.task_queue.remove_task(task_id)
            return True
        
        # Try to cancel running task
        if task_id in self.running_tasks:
            future = self.running_tasks[task_id]
            if future.cancel():
                del self.running_tasks[task_id]
                return True
        
        return False
    
    def get_pending_count(self) -> int:
        """Get the number of pending tasks.
        
        Returns:
            Number of tasks waiting to be executed.
        """
        return self.task_queue.size()
    
    def get_running_count(self) -> int:
        """Get the number of currently running tasks.
        
        Returns:
            Number of tasks currently executing.
        """
        return len(self.running_tasks)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get scheduler statistics.
        
        Returns:
            Dictionary with scheduler statistics.
        """
        queue_stats = self.task_queue.get_statistics()
        
        return {
            "pending_tasks": self.get_pending_count(),
            "running_tasks": self.get_running_count(),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tasks": queue_stats["total_tasks"] + len(self.completed_tasks) + len(self.failed_tasks),
        }
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running and not self._stop_event.is_set():
            try:
                self._process_completed_tasks()
                self._schedule_pending_tasks()
                time.sleep(self.config.poll_interval)
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(self.config.poll_interval)
    
    def _process_completed_tasks(self) -> None:
        """Process completed tasks and update their status."""
        completed_task_ids = []
        
        for task_id, future in self.running_tasks.items():
            if future.done():
                completed_task_ids.append(task_id)
                
                try:
                    result = future.result()
                    self.logger.info(f"Task {task_id} completed successfully")
                    self.completed_tasks.add(task_id)
                except Exception as e:
                    self.logger.error(f"Task {task_id} failed: {e}")
                    self.failed_tasks.add(task_id)
        
        # Remove completed tasks from running dict
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]
    
    def _schedule_pending_tasks(self) -> None:
        """Schedule pending tasks if capacity is available."""
        while (len(self.running_tasks) < self.config.max_concurrent_tasks 
               and not self.task_queue.is_empty()):
            
            task = self.task_queue.get_next_task()
            if task is None:
                break
            
            if task.is_cancelled():
                continue
            
            # Submit task for execution
            task.set_status(TaskStatus.RUNNING)
            future = self._executor.submit(self._execute_task, task)
            self.running_tasks[task.task_id] = future
            
            self.logger.info(f"Started execution of task {task.task_id}")
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a single task.
        
        Args:
            task: The task to execute.
            
        Returns:
            The task result.
            
        Raises:
            Exception: If task execution fails.
        """
        retries = 0
        last_error = None
        
        while retries <= self.config.max_retries:
            try:
                result = task.execute()
                # Set both result and status here to ensure consistency
                task.set_result(result)
                task.set_status(TaskStatus.COMPLETED)
                return result
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.config.max_retries:
                    self.logger.warning(f"Task {task.task_id} failed (attempt {retries}), retrying: {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"Task {task.task_id} failed after {self.config.max_retries} retries: {e}")
                    task.set_error(e)
                    raise e
        
        # This should never be reached, but just in case
        raise last_error if last_error else RuntimeError("Task execution failed")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
