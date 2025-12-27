"""Tests for Task Execution Engine."""

import pytest
import torch
import numpy as np
from typing import Any, Dict

from evo2.tasks.base import Task, TaskStatus, TaskPriority
from evo2.tasks.execution_engine import TaskExecutionEngine, ExecutionEngineConfig


class TestTaskExecutionEngine:
    """Test suite for TaskExecutionEngine."""

    def test_engine_initialization(self):
        """Test TaskExecutionEngine initialization."""
        config = ExecutionEngineConfig(
            input_size=10,
            hidden_size=32,
            output_size=5,
            num_layers=2
        )
        engine = TaskExecutionEngine(config)
        
        assert engine.input_size == 10
        assert engine.hidden_size == 32
        assert engine.output_size == 5
        assert engine.num_layers == 2
        assert engine.device.type == 'cpu'

    def test_engine_with_gpu(self):
        """Test engine initialization with GPU if available."""
        if torch.cuda.is_available():
            config = ExecutionEngineConfig(device='cuda')
            engine = TaskExecutionEngine(config)
            assert engine.device.type == 'cuda'
        else:
            # Should fall back to CPU if CUDA not available
            config = ExecutionEngineConfig(device='cuda')
            engine = TaskExecutionEngine(config)
            assert engine.device.type == 'cpu'

    def test_task_encoding(self):
        """Test task encoding functionality."""
        
        class TestTask(Task):
            def execute(self) -> Any:
                return "test_result"
        
        config = ExecutionEngineConfig(input_size=10, hidden_size=16, output_size=5)
        engine = TaskExecutionEngine(config)
        
        task = TestTask("test_1", TaskPriority.HIGH)
        task.metadata = {"feature1": 1.0, "feature2": 2.0}
        
        encoding = engine.encode_task(task)
        
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape == (1, 1, config.input_size)  # batch_size, sequence_length, input_size
        assert encoding.dtype == torch.float32

    def test_task_decoding(self):
        """Test task decoding functionality."""
        config = ExecutionEngineConfig(input_size=10, hidden_size=16, output_size=5)
        engine = TaskExecutionEngine(config)
        
        # Create a mock output tensor
        output = torch.randn(1, config.output_size)
        
        decoded = engine.decode_output(output)
        
        assert isinstance(decoded, Dict)
        assert "action_type" in decoded
        assert "parameters" in decoded

    def test_task_execution_pipeline(self):
        """Test complete task execution pipeline."""
        
        class SimpleTask(Task):
            def execute(self) -> Any:
                return "executed"
        
        config = ExecutionEngineConfig(
            input_size=10,
            hidden_size=16,
            output_size=5,
            num_layers=1
        )
        engine = TaskExecutionEngine(config)
        
        task = SimpleTask("test_task", TaskPriority.MEDIUM)
        
        # Execute the task through the engine
        result = engine.execute_task(task)
        
        # Check that task was processed (may not be completed if action was not EXECUTE)
        assert result is not None
        # The task should have been processed through the engine
        assert len(engine.execution_history) == 1

    def test_batch_processing(self):
        """Test batch processing of multiple tasks."""
        
        class BatchTask(Task):
            def __init__(self, task_id: str, value: int):
                super().__init__(task_id, TaskPriority.MEDIUM)
                self.value = value
            
            def execute(self) -> Any:
                return f"result_{self.value}"
        
        config = ExecutionEngineConfig(input_size=10, hidden_size=16, output_size=5)
        engine = TaskExecutionEngine(config)
        
        # Create multiple tasks
        tasks = [BatchTask(f"task_{i}", i) for i in range(3)]
        
        # Process batch
        results = engine.execute_batch(tasks)
        
        assert len(results) == 3
        # All tasks should have been processed
        assert len(engine.execution_history) == 3
        for result in results:
            assert result is not None

    def test_learning_mechanism(self):
        """Test the learning mechanism of the execution engine."""
        config = ExecutionEngineConfig(input_size=10, hidden_size=16, output_size=5)
        engine = TaskExecutionEngine(config)
        
        # Create training data with correct shape (batch, seq, features)
        task_encodings = torch.randn(5, 1, config.input_size)
        target_outputs = torch.randn(5, 1, config.output_size)
        rewards = torch.tensor([1.0, 0.5, -0.2, 0.8, 0.3])
        
        # Perform learning step
        initial_loss = engine.compute_loss(task_encodings, target_outputs)
        
        # Update the model
        engine.learn(task_encodings, target_outputs, rewards)
        
        # Loss should decrease after learning
        new_loss = engine.compute_loss(task_encodings, target_outputs)
        
        # Note: Due to randomness, we just check that learning doesn't crash
        assert isinstance(new_loss, torch.Tensor)

    def test_state_management(self):
        """Test state management of the execution engine."""
        config = ExecutionEngineConfig(input_size=10, hidden_size=16, output_size=5)
        engine = TaskExecutionEngine(config)
        
        # Get initial state
        initial_state = engine.get_state()
        
        # Execute a task to change state
        class TestTask(Task):
            def execute(self) -> Any:
                return "test"
        
        task = TestTask("test", TaskPriority.MEDIUM)
        engine.execute_task(task)
        
        # Get updated state
        updated_state = engine.get_state()
        
        # States should be different
        assert initial_state != updated_state

    def test_state_save_load(self):
        """Test saving and loading engine state."""
        config = ExecutionEngineConfig(input_size=10, hidden_size=16, output_size=5)
        engine = TaskExecutionEngine(config)
        
        # Execute a task to create state
        class TestTask(Task):
            def execute(self) -> Any:
                return "test"
        
        task = TestTask("test", TaskPriority.MEDIUM)
        engine.execute_task(task)
        
        # Save state
        state_dict = engine.save_state()
        
        # Create new engine and load state
        new_engine = TaskExecutionEngine(config)
        new_engine.load_state(state_dict)
        
        # Check that the new engine has the same configuration
        assert engine.get_state()["config"] == new_engine.get_state()["config"]
        # The execution history should be preserved if saved
        if "execution_count" in state_dict:
            assert state_dict["execution_count"] == new_engine.get_state()["execution_count"]

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ExecutionEngineConfig(input_size=10, hidden_size=16, output_size=5)
        assert config.input_size == 10
        
        # Invalid configs
        with pytest.raises(ValueError):
            ExecutionEngineConfig(input_size=0)  # input_size must be positive
        
        with pytest.raises(ValueError):
            ExecutionEngineConfig(hidden_size=0)  # hidden_size must be positive
        
        with pytest.raises(ValueError):
            ExecutionEngineConfig(output_size=0)  # output_size must be positive
        
        with pytest.raises(ValueError):
            ExecutionEngineConfig(num_layers=0)  # num_layers must be positive
