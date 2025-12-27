"""Tests for Agent Loop implementation."""

import pytest
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from evo2.agent.base import Agent
from evo2.agent.loop import AgentLoop, LoopConfig
from evo2.tasks.base import Task, TaskStatus, TaskPriority
from evo2.tasks.scheduler import TaskScheduler
from evo2.tasks.execution_engine import TaskExecutionEngine


class TestAgentLoop:
    """Test suite for AgentLoop."""

    def test_loop_initialization(self):
        """Test AgentLoop initialization."""
        config = LoopConfig(max_iterations=10, think_interval=0.1)
        
        # Mock agent
        agent = Mock(spec=Agent)
        
        loop = AgentLoop(agent, config)
        
        assert loop.agent is agent
        assert loop.max_iterations == 10
        assert loop.think_interval == 0.1
        assert loop.is_running() is False
        assert loop.iteration_count == 0

    def test_loop_configuration_validation(self):
        """Test loop configuration validation."""
        agent = Mock(spec=Agent)
        
        # Valid config
        config = LoopConfig(max_iterations=5, think_interval=0.2)
        loop = AgentLoop(agent, config)
        assert loop.max_iterations == 5
        
        # Invalid max_iterations
        with pytest.raises(ValueError):
            LoopConfig(max_iterations=0)
        
        with pytest.raises(ValueError):
            LoopConfig(max_iterations=-1)
        
        # Invalid think_interval
        with pytest.raises(ValueError):
            LoopConfig(think_interval=0)
        
        with pytest.raises(ValueError):
            LoopConfig(think_interval=-1)

    def test_loop_start_stop(self):
        """Test starting and stopping the agent loop."""
        config = LoopConfig(max_iterations=3, think_interval=0.01)
        agent = Mock(spec=Agent)
        
        # Mock agent methods
        agent.think.return_value = {"action": "wait"}
        agent.act.return_value = "acted"
        agent.learn.return_value = None
        
        loop = AgentLoop(agent, config)
        
        assert loop.is_running() is False
        
        loop.start()
        assert loop.is_running() is True
        
        # Wait for loop to complete
        import time
        time.sleep(0.5)
        
        loop.stop()
        assert loop.is_running() is False
        
        # Should have completed some iterations
        assert loop.iteration_count > 0

    def test_single_iteration(self):
        """Test single iteration of the agent loop."""
        config = LoopConfig(max_iterations=1, think_interval=0.01)
        agent = Mock(spec=Agent)
        
        # Mock agent methods
        agent.think.return_value = {"action": "execute", "param": "test"}
        agent.act.return_value = "result"
        agent.learn.return_value = None
        
        loop = AgentLoop(agent, config)
        
        # Execute single iteration
        loop.run_iteration()
        
        assert loop.iteration_count == 1
        
        # Verify agent methods were called
        agent.think.assert_called_once()
        agent.act.assert_called_once()
        agent.learn.assert_called_once()

    def test_loop_with_task_scheduler(self):
        """Test loop integration with task scheduler."""
        config = LoopConfig(max_iterations=5, think_interval=0.01)
        agent = Mock(spec=Agent)
        
        # Mock agent methods
        agent.think.return_value = {"action": "schedule_task", "task_type": "research"}
        agent.act.return_value = "task_scheduled"
        agent.learn.return_value = None
        
        # Create scheduler
        scheduler = TaskScheduler()
        loop = AgentLoop(agent, config, scheduler)
        
        loop.start()
        
        # Wait for completion
        import time
        time.sleep(0.5)
        
        loop.stop()
        
        assert loop.iteration_count > 0

    def test_loop_with_execution_engine(self):
        """Test loop integration with execution engine."""
        config = LoopConfig(max_iterations=3, think_interval=0.01)
        agent = Mock(spec=Agent)
        
        # Mock agent methods
        agent.think.return_value = {"action": "think", "data": "research_data"}
        agent.act.return_value = "thought_processed"
        agent.learn.return_value = None
        
        # Create execution engine
        from evo2.tasks.execution_engine import ExecutionEngineConfig
        engine_config = ExecutionEngineConfig(input_size=10, hidden_size=16, output_size=5)
        engine = TaskExecutionEngine(engine_config)
        
        loop = AgentLoop(agent, config, execution_engine=engine)
        
        loop.start()
        
        # Wait for completion
        import time
        time.sleep(0.5)
        
        loop.stop()
        
        assert loop.iteration_count > 0

    def test_loop_error_handling(self):
        """Test error handling in the agent loop."""
        config = LoopConfig(max_iterations=3, think_interval=0.01)
        agent = Mock(spec=Agent)
        
        # Mock agent to raise exception
        agent.think.side_effect = Exception("Thinking failed")
        
        loop = AgentLoop(agent, config)
        
        # Should handle error gracefully
        loop.run_iteration()
        
        # Iteration should still be counted
        assert loop.iteration_count == 1

    def test_loop_state_tracking(self):
        """Test loop state tracking and history."""
        config = LoopConfig(max_iterations=5, think_interval=0.01)
        agent = Mock(spec=Agent)
        
        # Mock agent methods
        agent.think.return_value = {"action": "analyze", "target": "data"}
        agent.act.return_value = "analysis_complete"
        agent.learn.return_value = None
        
        loop = AgentLoop(agent, config)
        
        # Run multiple iterations
        for _ in range(3):
            loop.run_iteration()
        
        assert loop.iteration_count == 3
        
        # Check history
        history = loop.get_history()
        assert len(history) == 3
        
        for entry in history:
            assert "iteration" in entry
            assert "thought" in entry
            assert "action" in entry
            assert "timestamp" in entry

    def test_loop_reset(self):
        """Test resetting the agent loop."""
        config = LoopConfig(max_iterations=5, think_interval=0.01)
        agent = Mock(spec=Agent)
        
        # Mock agent methods
        agent.think.return_value = {"action": "reset_test"}
        agent.act.return_value = "reset_action"
        agent.learn.return_value = None
        
        loop = AgentLoop(agent, config)
        
        # Run some iterations
        loop.run_iteration()
        loop.run_iteration()
        
        assert loop.iteration_count == 2
        
        # Reset the loop
        loop.reset()
        
        assert loop.iteration_count == 0
        assert len(loop.get_history()) == 0

    def test_loop_performance_metrics(self):
        """Test performance metrics collection."""
        config = LoopConfig(max_iterations=3, think_interval=0.01)
        agent = Mock(spec=Agent)
        
        # Mock agent methods
        agent.think.return_value = {"action": "metrics_test"}
        agent.act.return_value = "metrics_action"
        agent.learn.return_value = None
        
        loop = AgentLoop(agent, config)
        
        # Run iterations
        loop.run_iteration()
        loop.run_iteration()
        
        metrics = loop.get_metrics()
        
        assert "total_iterations" in metrics
        assert "average_iteration_time" in metrics
        assert "total_think_time" in metrics
        assert "total_act_time" in metrics
        assert "total_learn_time" in metrics
        
        assert metrics["total_iterations"] == 2
 
 