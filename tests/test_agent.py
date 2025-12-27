"""Tests for the Agent base class."""

import pytest
from abc import ABC

from evo2.agent.base import Agent


class TestAgent:
    """Test suite for Agent base class."""

    def test_agent_is_abstract(self):
        """Test that Agent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Agent()

    def test_agent_subclass_requires_methods(self):
        """Test that Agent subclasses must implement abstract methods."""
        
        class IncompleteAgent(Agent):
            """Incomplete agent missing required methods."""
            pass
        
        with pytest.raises(TypeError):
            IncompleteAgent()

    def test_complete_agent_subclass(self):
        """Test that a complete agent subclass can be instantiated."""
        
        class CompleteAgent(Agent):
            """Complete agent implementing all required methods."""
            
            def initialize(self):
                """Initialize the agent."""
                pass
            
            def think(self):
                """Agent thinking process."""
                pass
            
            def act(self):
                """Agent action selection."""
                pass
            
            def learn(self, experience):
                """Agent learning from experience."""
                pass
        
        agent = CompleteAgent()
        assert isinstance(agent, Agent)
        assert hasattr(agent, 'initialize')
        assert hasattr(agent, 'think')
        assert hasattr(agent, 'act')
        assert hasattr(agent, 'learn')

    def test_agent_state_management(self):
        """Test agent state management functionality."""
        
        class TestAgent(Agent):
            """Test agent implementation."""
            
            def initialize(self):
                """Initialize the agent."""
                self.state = 'initialized'
            
            def think(self):
                """Agent thinking process."""
                return 'thinking'
            
            def act(self):
                """Agent action selection."""
                return 'action'
            
            def learn(self, experience):
                """Agent learning from experience."""
                self.experience = experience
        
        agent = TestAgent()
        agent.initialize()
        assert agent.state == 'initialized'
        
        result = agent.think()
        assert result == 'thinking'
        
        action = agent.act()
        assert action == 'action'
        
        agent.learn('test_experience')
        assert agent.experience == 'test_experience'
