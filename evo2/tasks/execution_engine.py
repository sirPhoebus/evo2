"""Task Execution Engine with RNN-based processing for Evo2."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import json

from .base import Task, TaskPriority, TaskStatus


@dataclass
class ExecutionEngineConfig:
    """Configuration for the Task Execution Engine."""
    input_size: int = 64
    hidden_size: int = 128
    output_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    device: str = 'cpu'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        # Validate device
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'


class TaskRNN(nn.Module):
    """RNN model for task processing and decision making."""
    
    def __init__(self, config: ExecutionEngineConfig):
        """Initialize the RNN model.
        
        Args:
            config: Engine configuration.
        """
        super(TaskRNN, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.input_embedding = nn.Linear(config.input_size, config.hidden_size)
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )
        
        # Action type prediction head
        self.action_head = nn.Linear(config.output_size, 4)  # 4 action types
        
        # Parameter prediction head
        self.param_head = nn.Linear(config.output_size, config.output_size)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass through the RNN.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            hidden: Optional hidden state tuple.
            
        Returns:
            Tuple of (output, hidden_state).
        """
        # Input embedding
        embedded = self.input_embedding(x)
        embedded = torch.relu(embedded)
        
        # RNN forward pass
        output, hidden = self.rnn(embedded, hidden)
        
        # Output processing
        processed = self.output_layers(output)
        
        return processed, hidden
    
    def predict_actions(self, output: torch.Tensor) -> torch.Tensor:
        """Predict action types from output.
        
        Args:
            output: RNN output tensor.
            
        Returns:
            Action type predictions.
        """
        return self.action_head(output)
    
    def predict_parameters(self, output: torch.Tensor) -> torch.Tensor:
        """Predict task parameters from output.
        
        Args:
            output: RNN output tensor.
            
        Returns:
            Parameter predictions.
        """
        return self.param_head(output)


class TaskExecutionEngine:
    """RNN-based task execution engine for Evo2.
    
    This engine uses a recurrent neural network to process tasks,
    make decisions, and learn from experience.
    """
    
    def __init__(self, config: Optional[ExecutionEngineConfig] = None):
        """Initialize the task execution engine.
        
        Args:
            config: Optional engine configuration.
        """
        self.config = config or ExecutionEngineConfig()
        self.device = torch.device(self.config.device)
        
        # Expose config properties for easier access
        self.input_size = self.config.input_size
        self.hidden_size = self.config.hidden_size
        self.output_size = self.config.output_size
        self.num_layers = self.config.num_layers
        
        # Initialize RNN model
        self.model = TaskRNN(self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # State management
        self.hidden_state: Optional[Tuple] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger("TaskExecutionEngine")
        
    def encode_task(self, task: Task) -> torch.Tensor:
        """Encode a task into a tensor representation.
        
        Args:
            task: The task to encode.
            
        Returns:
            Encoded task tensor.
        """
        # Create feature vector from task properties
        features = []
        
        # Task priority (one-hot encoded)
        priority_one_hot = [0.0] * 4
        priority_map = {
            TaskPriority.LOW: 0,
            TaskPriority.MEDIUM: 1,
            TaskPriority.HIGH: 2,
            TaskPriority.CRITICAL: 3
        }
        priority_one_hot[priority_map[task.priority]] = 1.0
        features.extend(priority_one_hot)
        
        # Task status (one-hot encoded)
        status_one_hot = [0.0] * 5
        status_map = {
            TaskStatus.PENDING: 0,
            TaskStatus.RUNNING: 1,
            TaskStatus.COMPLETED: 2,
            TaskStatus.FAILED: 3,
            TaskStatus.CANCELLED: 4
        }
        status_one_hot[status_map[task.status]] = 1.0
        features.extend(status_one_hot)
        
        # Task metadata features
        metadata_features = []
        if task.metadata:
            # Extract numeric features from metadata
            for key, value in task.metadata.items():
                if isinstance(value, (int, float)):
                    metadata_features.append(float(value))
                elif isinstance(value, str):
                    # Simple hash-based encoding for strings
                    metadata_features.append(float(hash(key + value) % 1000) / 1000.0)
        
        # Pad or truncate to fixed size
        while len(metadata_features) < 50:
            metadata_features.append(0.0)
        metadata_features = metadata_features[:50]
        features.extend(metadata_features)
        
        # Pad to input_size
        while len(features) < self.config.input_size:
            features.append(0.0)
        features = features[:self.config.input_size]
        
        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def decode_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """Decode RNN output into actionable decisions.
        
        Args:
            output: RNN output tensor.
            
        Returns:
            Decoded action dictionary.
        """
        # Get predictions
        actions = self.model.predict_actions(output)
        parameters = self.model.predict_parameters(output)
        
        # Decode action type
        action_probs = torch.softmax(actions, dim=-1)
        action_type = torch.argmax(action_probs, dim=-1).item()
        
        # Decode parameters
        params = parameters.squeeze().detach().cpu().numpy().tolist()
        
        action_names = ["EXECUTE", "WAIT", "RETRY", "ESCALATE"]
        
        return {
            "action_type": action_names[action_type],
            "confidence": action_probs.squeeze().max().item(),
            "parameters": params,
            "raw_actions": action_probs.squeeze().detach().cpu().numpy().tolist(),
            "raw_parameters": parameters.squeeze().detach().cpu().numpy().tolist()
        }
    
    def execute_task(self, task: Task) -> Any:
        """Execute a task through the RNN engine.
        
        Args:
            task: The task to execute.
            
        Returns:
            Execution result.
        """
        try:
            # Encode task
            task_tensor = self.encode_task(task)
            
            # Forward pass through RNN
            output, self.hidden_state = self.model(task_tensor, self.hidden_state)
            
            # Decode output
            decision = self.decode_output(output)
            
            # Execute based on decision
            result = self._execute_decision(task, decision)
            
            # Record execution
            self._record_execution(task, decision, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            task.set_error(e)
            return None
    
    def execute_batch(self, tasks: List[Task]) -> List[Any]:
        """Execute multiple tasks in batch.
        
        Args:
            tasks: List of tasks to execute.
            
        Returns:
            List of execution results.
        """
        results = []
        
        for task in tasks:
            result = self.execute_task(task)
            results.append(result)
        
        return results
    
    def _execute_decision(self, task: Task, decision: Dict[str, Any]) -> Any:
        """Execute the decision made by the RNN.
        
        Args:
            task: The task to execute.
            decision: The decoded decision.
            
        Returns:
            Execution result.
        """
        action_type = decision["action_type"]
        
        if action_type == "EXECUTE":
            task.set_status(TaskStatus.RUNNING)
            result = task.execute()
            task.set_status(TaskStatus.COMPLETED)
            return result
        elif action_type == "WAIT":
            # Simulate waiting
            import time
            time.sleep(0.1)
            return "waited"
        elif action_type == "RETRY":
            # Retry execution
            try:
                return task.execute()
            except Exception as e:
                task.set_error(e)
                return None
        elif action_type == "ESCALATE":
            # Mark for escalation
            task.metadata["escalated"] = True
            return "escalated"
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def _record_execution(self, task: Task, decision: Dict[str, Any], result: Any) -> None:
        """Record execution for learning.
        
        Args:
            task: The executed task.
            decision: The decision made.
            result: The execution result.
        """
        record = {
            "task_id": task.task_id,
            "task_priority": task.priority.value,
            "decision": decision,
            "result": result,
            "success": task.is_completed(),
            "timestamp": task.get_completed_time()
        }
        self.execution_history.append(record)
    
    def learn(self, task_encodings: torch.Tensor, target_outputs: torch.Tensor, rewards: torch.Tensor) -> float:
        """Learn from experience.
        
        Args:
            task_encodings: Batch of task encodings.
            target_outputs: Target outputs for supervised learning.
            rewards: Reward signals for reinforcement learning.
            
        Returns:
            Loss value.
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = self.model(task_encodings)
        
        # Compute loss (combination of supervised and reinforcement)
        supervised_loss = self.criterion(outputs, target_outputs)
        
        # Simple reward-based loss
        reward_loss = -torch.mean(rewards)
        
        # Combined loss
        total_loss = supervised_loss + 0.1 * reward_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def compute_loss(self, task_encodings: torch.Tensor, target_outputs: torch.Tensor) -> torch.Tensor:
        """Compute loss without backpropagation.
        
        Args:
            task_encodings: Batch of task encodings.
            target_outputs: Target outputs.
            
        Returns:
            Loss tensor.
        """
        with torch.no_grad():
            outputs, _ = self.model(task_encodings)
            loss = self.criterion(outputs, target_outputs)
            return loss
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the execution engine.
        
        Returns:
            State dictionary.
        """
        return {
            "execution_count": len(self.execution_history),
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "output_size": self.config.output_size,
                "num_layers": self.config.num_layers
            },
            "has_hidden_state": self.hidden_state is not None
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state for later loading.
        
        Returns:
            Serializable state dictionary.
        """
        state = self.get_state()
        
        # Add execution count to saved state
        state["execution_count"] = len(self.execution_history)
        
        # Convert tensors to CPU for serialization
        if "model_state" in state:
            model_state = {}
            for key, tensor in state["model_state"].items():
                model_state[key] = tensor.cpu().numpy()
            state["model_state"] = model_state
        
        return state
    
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load a previously saved state.
        
        Args:
            state_dict: The state dictionary to load.
        """
        # Load model state
        if "model_state" in state_dict:
            model_state = {}
            for key, array in state_dict["model_state"].items():
                model_state[key] = torch.tensor(array).to(self.device)
            self.model.load_state_dict(model_state)
        
        # Load optimizer state
        if "optimizer_state" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state"])
        
        # Load hidden state
        if "hidden_state" in state_dict and state_dict["hidden_state"] is not None:
            # Convert numpy arrays back to tensors
            hidden = state_dict["hidden_state"]
            if isinstance(hidden, tuple) and len(hidden) == 2:
                h, c = hidden
                if isinstance(h, np.ndarray):
                    h = torch.tensor(h).to(self.device)
                if isinstance(c, np.ndarray):
                    c = torch.tensor(c).to(self.device)
                self.hidden_state = (h, c)
        
        # Restore execution count if available
        if "execution_count" in state_dict:
            # Adjust execution history to match saved count
            target_count = state_dict["execution_count"]
            current_count = len(self.execution_history)
            
            if target_count > current_count:
                # Add dummy entries to match the saved count
                for _ in range(target_count - current_count):
                    self.execution_history.append({"restored": True})
            elif target_count < current_count:
                # Remove excess entries
                self.execution_history = self.execution_history[:target_count]
        
        self.logger.info("State loaded successfully")
    
    def reset(self) -> None:
        """Reset the execution engine to initial state."""
        self.hidden_state = None
        self.execution_history.clear()
        self.logger.info("Execution engine reset")
