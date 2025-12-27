"""Task queue management for Evo2."""

import heapq
from typing import Dict, List, Optional
from collections import defaultdict
import logging

from .base import Task, TaskStatus, TaskPriority


class TaskQueue:
    """Priority-based task queue for managing tasks.
    
    This class provides a priority queue implementation where tasks are
    retrieved based on their priority level. Higher priority tasks are
    returned first.
    """
    
    def __init__(self):
        """Initialize an empty task queue."""
        # Priority queue using heapq (min-heap, so we use negative priority)
        self._queue: List[tuple[int, float, Task]] = []
        self._tasks: Dict[str, Task] = {}
        self._counter = 0  # For FIFO ordering within same priority
        self.logger = logging.getLogger("TaskQueue")
    
    def add_task(self, task: Task) -> None:
        """Add a task to the queue.
        
        Args:
            task: The task to add.
        """
        if task.task_id in self._tasks:
            self.logger.warning(f"Task {task.task_id} already exists in queue")
            return
        
        # Add to priority queue (negative priority for max-heap behavior)
        priority_value = -task.priority.value
        heapq.heappush(self._queue, (priority_value, self._counter, task))
        self._tasks[task.task_id] = task
        self._counter += 1
        
        self.logger.debug(f"Added task {task.task_id} with priority {task.priority.value}")
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next highest priority task.
        
        Returns:
            The next task, or None if queue is empty.
        """
        if not self._queue:
            return None
        
        _, _, task = heapq.heappop(self._queue)
        del self._tasks[task.task_id]
        
        self.logger.debug(f"Retrieved task {task.task_id}")
        return task
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID without removing it from the queue.
        
        Args:
            task_id: The ID of the task to retrieve.
            
        Returns:
            The task if found, None otherwise.
        """
        return self._tasks.get(task_id)
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue by ID.
        
        Args:
            task_id: The ID of the task to remove.
            
        Returns:
            True if task was removed, False if not found.
        """
        if task_id not in self._tasks:
            return False
        
        # Remove from the tasks dictionary
        task = self._tasks.pop(task_id)
        
        # Mark the task as removed in the heap (lazy deletion)
        # We'll rebuild the heap if it gets too large
        self.logger.debug(f"Removed task {task_id}")
        return True
    
    def size(self) -> int:
        """Get the number of tasks in the queue.
        
        Returns:
            Number of pending tasks.
        """
        return len(self._tasks)
    
    def is_empty(self) -> bool:
        """Check if the queue is empty.
        
        Returns:
            True if no tasks are in the queue.
        """
        return len(self._tasks) == 0
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a specific status.
        
        Args:
            status: The status to filter by.
            
        Returns:
            List of tasks with the given status.
        """
        return [task for task in self._tasks.values() if task.status == status]
    
    def get_tasks_by_priority(self, priority: TaskPriority) -> List[Task]:
        """Get all tasks with a specific priority.
        
        Args:
            priority: The priority to filter by.
            
        Returns:
            List of tasks with the given priority.
        """
        return [task for task in self._tasks.values() if task.priority == priority]
    
    def peek_next_task(self) -> Optional[Task]:
        """Look at the next task without removing it.
        
        Returns:
            The next task, or None if queue is empty.
        """
        if not self._queue:
            return None
        
        _, _, task = self._queue[0]
        return task
    
    def clear(self) -> None:
        """Remove all tasks from the queue."""
        self._queue.clear()
        self._tasks.clear()
        self.logger.info("Cleared all tasks from queue")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get queue statistics.
        
        Returns:
            Dictionary with statistics about the queue.
        """
        status_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        
        for task in self._tasks.values():
            status_counts[task.status.value] += 1
            priority_counts[task.priority.value] += 1
        
        return {
            "total_tasks": self.size(),
            "status_distribution": dict(status_counts),
            "priority_distribution": dict(priority_counts),
        }
    
    def rebuild_heap(self) -> None:
        """Rebuild the heap to remove stale entries.
        
        This should be called periodically if many tasks have been removed
        to prevent the heap from growing too large with stale entries.
        """
        # Filter out tasks that are no longer in the tasks dictionary
        valid_entries = [
            (priority, count, task) 
            for priority, count, task in self._queue 
            if task.task_id in self._tasks
        ]
        
        self._queue = valid_entries
        heapq.heapify(self._queue)
        
        self.logger.debug("Rebuilt task queue heap")
    
    def __len__(self) -> int:
        """Get the number of tasks in the queue."""
        return self.size()
    
    def __contains__(self, task_id: str) -> bool:
        """Check if a task ID is in the queue.
        
        Args:
            task_id: The task ID to check.
            
        Returns:
            True if task is in the queue.
        """
        return task_id in self._tasks
