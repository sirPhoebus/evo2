"""Base Agent class for Evo2 Meta-RL Scientist."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging


class Agent(ABC):
    """Abstract base class for all agents in Evo2.
    
    This class defines the core interface that all agents must implement.
    Agents are autonomous entities that can think, act, and learn from
    experience in their environment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent with optional configuration.
        
        Args:
            config: Optional configuration dictionary for the agent.
        """
        self.config = config or {}
        self.state = 'uninitialized'
        self.logger = logging.getLogger(self.__class__.__name__)
        self._experience_history = []
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the agent's internal state and components.
        
        This method should set up all necessary components for the agent
        to function properly, including neural networks, memory systems,
        and any other required infrastructure.
        """
        pass
    
    @abstractmethod
    def think(self) -> Any:
        """Execute the agent's thinking process.
        
        This method implements the core cognitive processes of the agent,
        including literature review, causal reasoning, hypothesis generation,
        and experiment planning.
        
        Returns:
            The result of the thinking process, which could be a hypothesis,
            experiment plan, or other cognitive output.
        """
        pass
    
    @abstractmethod
    def act(self) -> Any:
        """Select and execute an action based on current state.
        
        This method translates the agent's cognitive processes into
        concrete actions in the environment, such as running experiments,
        updating models, or communicating results.
        
        Returns:
            The result of the executed action.
        """
        pass
    
    @abstractmethod
    def learn(self, experience: Any) -> None:
        """Learn from experience and update internal models.
        
        This method implements the learning mechanisms of the agent,
        updating causal models, neural networks, and other knowledge
        structures based on new experience.
        
        Args:
            experience: The experience data to learn from.
        """
        pass
    
    def get_state(self) -> str:
        """Get the current state of the agent.
        
        Returns:
            The current state string.
        """
        return self.state
    
    def set_state(self, state: str) -> None:
        """Set the agent's state.
        
        Args:
            state: The new state to set.
        """
        self.state = state
        self.logger.info(f"Agent state changed to: {state}")
    
    def get_experience_history(self) -> list:
        """Get the agent's experience history.
        
        Returns:
            List of past experiences.
        """
        return self._experience_history.copy()
    
    def add_experience(self, experience: Any) -> None:
        """Add an experience to the agent's history.
        
        Args:
            experience: The experience to add.
        """
        self._experience_history.append(experience)
    
    def reset(self) -> None:
        """Reset the agent to its initial state.
        
        This method clears the agent's experience history and resets
        its state to 'uninitialized'. Subclasses should override this
        method to perform additional cleanup as needed.
        """
        self._experience_history.clear()
        self.state = 'uninitialized'
        self.logger.info("Agent reset to initial state")
