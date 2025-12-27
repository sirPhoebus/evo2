"""Agent Loop implementation for Evo2."""

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from .base import Agent
from ..tasks.scheduler import TaskScheduler, SchedulerConfig
from ..tasks.execution_engine import TaskExecutionEngine, ExecutionEngineConfig


@dataclass
class LoopConfig:
    """Configuration for the agent loop."""
    max_iterations: int = 1000
    think_interval: float = 0.1  # seconds between iterations
    enable_scheduler: bool = True
    enable_execution_engine: bool = True
    track_performance: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.think_interval <= 0:
            raise ValueError("think_interval must be positive")


class AgentLoop:
    """Main agent loop for managing agent execution cycles.
    
    This class manages the continuous execution cycle of an agent,
    including thinking, acting, and learning phases.
    """
    
    def __init__(
        self,
        agent: Agent,
        config: Optional[LoopConfig] = None,
        scheduler: Optional[TaskScheduler] = None,
        execution_engine: Optional[TaskExecutionEngine] = None
    ):
        """Initialize the agent loop.
        
        Args:
            agent: The agent to run in the loop.
            config: Optional loop configuration.
            scheduler: Optional task scheduler.
            execution_engine: Optional task execution engine.
        """
        self.agent = agent
        self.config = config or LoopConfig()
        self.scheduler = scheduler
        self.execution_engine = execution_engine
        
        # Loop state
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.iteration_count = 0
        self.history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._performance_metrics = {
            "total_iterations": 0,
            "total_think_time": 0.0,
            "total_act_time": 0.0,
            "total_learn_time": 0.0,
            "start_time": None,
            "last_iteration_time": None
        }
        
        self.logger = logging.getLogger("AgentLoop")
        
        # Initialize components if enabled
        if self.config.enable_scheduler and self.scheduler is None:
            self.scheduler = TaskScheduler()
        
        if self.config.enable_execution_engine and self.execution_engine is None:
            engine_config = ExecutionEngineConfig()
            self.execution_engine = TaskExecutionEngine(engine_config)
    
    def start(self) -> None:
        """Start the agent loop."""
        if self._running:
            self.logger.warning("Agent loop is already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._performance_metrics["start_time"] = time.time()
        
        self._loop_thread = threading.Thread(target=self._loop_main, daemon=True)
        self._loop_thread.start()
        
        self.logger.info("Agent loop started")
    
    def stop(self) -> None:
        """Stop the agent loop."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)
        
        # Stop components
        if self.scheduler:
            self.scheduler.stop()
        
        self.logger.info("Agent loop stopped")
    
    def is_running(self) -> bool:
        """Check if the loop is running.
        
        Returns:
            True if loop is running.
        """
        return self._running
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single iteration of the agent loop.
        
        Returns:
            Dictionary containing iteration results.
        """
        iteration_start = time.time()
        
        try:
            # Think phase
            think_start = time.time()
            thought = self.agent.think()
            think_time = time.time() - think_start
            
            # Act phase
            act_start = time.time()
            action_result = self.agent.act()
            act_time = time.time() - act_start
            
            # Learn phase
            learn_start = time.time()
            self.agent.learn({"thought": thought, "action": action_result})
            learn_time = time.time() - learn_start
            
            # Update iteration count
            self.iteration_count += 1
            
            # Record iteration
            iteration_record = {
                "iteration": self.iteration_count,
                "thought": thought,
                "action": action_result,
                "timestamp": time.time(),
                "durations": {
                    "think": think_time,
                    "act": act_time,
                    "learn": learn_time,
                    "total": time.time() - iteration_start
                }
            }
            self.history.append(iteration_record)
            
            # Update performance metrics
            if self.config.track_performance:
                self._update_performance_metrics(iteration_record)
            
            self.logger.debug(f"Completed iteration {self.iteration_count}")
            
            return iteration_record
            
        except Exception as e:
            self.logger.error(f"Error in iteration {self.iteration_count + 1}: {e}")
            
            # Update iteration count even on error
            self.iteration_count += 1
            
            # Record error
            error_record = {
                "iteration": self.iteration_count,
                "error": str(e),
                "timestamp": time.time(),
                "durations": {"total": time.time() - iteration_start}
            }
            self.history.append(error_record)
            
            return error_record
    
    def _loop_main(self) -> None:
        """Main loop execution method."""
        while (self._running 
               and not self._stop_event.is_set() 
               and self.iteration_count < self.config.max_iterations):
            
            # Run single iteration
            self.run_iteration()
            
            # Sleep between iterations
            if not self._stop_event.wait(self.config.think_interval):
                continue  # Continue if not stopped
            else:
                break  # Stop if stop event was set
        
        self.logger.info(f"Agent loop completed {self.iteration_count} iterations")
    
    def _update_performance_metrics(self, iteration_record: Dict[str, Any]) -> None:
        """Update performance metrics.
        
        Args:
            iteration_record: The iteration record to process.
        """
        self._performance_metrics["total_iterations"] += 1
        
        if "durations" in iteration_record:
            durations = iteration_record["durations"]
            self._performance_metrics["total_think_time"] += durations.get("think", 0)
            self._performance_metrics["total_act_time"] += durations.get("act", 0)
            self._performance_metrics["total_learn_time"] += durations.get("learn", 0)
        
        self._performance_metrics["last_iteration_time"] = iteration_record["timestamp"]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the execution history.
        
        Returns:
            List of iteration records.
        """
        return self.history.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary containing performance metrics.
        """
        metrics = self._performance_metrics.copy()
        
        if metrics["total_iterations"] > 0:
            total_time = sum(record.get("durations", {}).get("total", 0) 
                           for record in self.history)
            metrics["average_iteration_time"] = total_time / metrics["total_iterations"]
        else:
            metrics["average_iteration_time"] = 0.0
        
        if metrics["start_time"]:
            metrics["uptime"] = time.time() - metrics["start_time"]
        else:
            metrics["uptime"] = 0.0
        
        return metrics
    
    def reset(self) -> None:
        """Reset the loop state."""
        self.iteration_count = 0
        self.history.clear()
        self._performance_metrics = {
            "total_iterations": 0,
            "total_think_time": 0.0,
            "total_act_time": 0.0,
            "total_learn_time": 0.0,
            "start_time": None,
            "last_iteration_time": None
        }
        
        self.logger.info("Agent loop reset")
    
    def get_last_iteration(self) -> Optional[Dict[str, Any]]:
        """Get the last iteration record.
        
        Returns:
            The last iteration record, or None if no iterations.
        """
        return self.history[-1] if self.history else None
    
    def get_iteration_by_number(self, iteration_num: int) -> Optional[Dict[str, Any]]:
        """Get a specific iteration record.
        
        Args:
            iteration_num: The iteration number to retrieve.
            
        Returns:
            The iteration record, or None if not found.
        """
        for record in self.history:
            if record.get("iteration") == iteration_num:
                return record
        return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
