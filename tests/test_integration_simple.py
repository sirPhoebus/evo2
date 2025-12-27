"""Simple integration tests for the complete Evo2 system."""

import pytest
import numpy as np
from typing import Any, Dict, List
import time

from evo2.agent.integrated import IntegratedAgent, IntegratedAgentConfig
from evo2.agent.loop import LoopConfig
from evo2.causal.model import CausalModel, CausalModelConfig
from evo2.experiments.framework import Experiment, ExperimentConfig, SimpleExperiment
from evo2.experiments.executor import ExperimentExecutor, ExecutorConfig
from evo2.experiments.analyzer import ExperimentAnalyzer, AnalysisConfig


class TestSimpleIntegration:
    """Test suite for basic system integration."""
    
    def test_integrated_agent_initialization(self):
        """Test integrated agent initialization."""
        config = IntegratedAgentConfig(
            agent_name="TestAgent",
            loop_config=LoopConfig(max_iterations=5),
            causal_config=CausalModelConfig(num_variables=5)
        )
        
        agent = IntegratedAgent(config)
        
        assert agent.config.agent_name == "TestAgent"
        assert agent.config.loop_config.max_iterations == 5
        assert len(agent.causal_model.variables) == 0  # Start empty
        assert agent.current_iteration == 0
        assert agent.total_experiments_run == 0
    
    def test_integrated_agent_think_phase(self):
        """Test think phase of integrated agent."""
        config = IntegratedAgentConfig(
            agent_name="ThinkTest",
            loop_config=LoopConfig(max_iterations=3)
        )
        
        agent = IntegratedAgent(config)
        agent.current_iteration = 1
        
        # Run think phase
        thought = agent.think()
        
        assert "iteration" in thought
        assert "model_summary" in thought
        assert "hypotheses" in thought
        assert "experiment_plans" in thought
        assert "confidence" in thought
        assert thought["iteration"] == 1
        assert isinstance(thought["hypotheses"], list)
        assert isinstance(thought["experiment_plans"], list)
    
    def test_integrated_agent_act_phase(self):
        """Test act phase of integrated agent."""
        config = IntegratedAgentConfig(
            agent_name="ActTest",
            executor_config=ExecutorConfig(max_concurrent_experiments=2)
        )
        
        agent = IntegratedAgent(config)
        agent.current_iteration = 1
        
        # Create a mock thought
        thought = {
            "iteration": 1,
            "hypotheses": [{"type": "exploratory", "description": "Test hypothesis"}],
            "experiment_plans": [
                {
                    "type": "exploratory",
                    "description": "Test experiment",
                    "variables": ["test_var"],
                    "sample_size": 20
                }
            ]
        }
        
        # Run act phase
        action_result = agent.act(thought)
        
        assert "iteration" in action_result
        assert "experiment_results" in action_result
        assert "experiments_run" in action_result
        assert action_result["iteration"] == 1
        assert len(action_result["experiment_results"]) == 1
        assert action_result["experiments_run"] == 1
    
    def test_integrated_agent_learn_phase(self):
        """Test learn phase of integrated agent."""
        config = IntegratedAgentConfig(
            agent_name="LearnTest",
            causal_config=CausalModelConfig(num_variables=3)
        )
        
        agent = IntegratedAgent(config)
        agent.current_iteration = 1
        
        # Create mock action result with successful experiment
        action_result = {
            "iteration": 1,
            "experiment_results": [
                {
                    "plan": {"type": "exploratory", "variables": ["A", "B"]},
                    "experiment": {
                        "experiment_id": "test_exp",
                        "config": {"name": "test"}
                    },
                    "result": {
                        "A": np.random.normal(0, 1, 15),
                        "B": np.random.normal(0.5, 1, 15)
                    },
                    "success": True
                }
            ]
        }
        
        # Run learn phase
        learning_result = agent.learn(action_result)
        
        assert "iteration" in learning_result
        assert "learning_events" in learning_result
        assert "model_updates" in learning_result
        assert learning_result["iteration"] == 1
        assert len(learning_result["learning_events"]) == 1
    
    def test_integrated_agent_single_iteration(self):
        """Test complete single iteration of integrated agent."""
        config = IntegratedAgentConfig(
            agent_name="IterationTest",
            loop_config=LoopConfig(max_iterations=1),
            executor_config=ExecutorConfig(max_concurrent_experiments=1),
            causal_config=CausalModelConfig(num_variables=3)
        )
        
        agent = IntegratedAgent(config)
        
        # Run single iteration
        result = agent.run(max_iterations=1)
        
        assert "agent_info" in result
        assert "run_summary" in result
        assert "causal_model_summary" in result
        assert "performance_metrics" in result
        assert result["run_summary"]["total_iterations"] == 1
        assert result["run_summary"]["successful_iterations"] == 1
        assert agent.current_iteration == 1
        assert agent.total_experiments_run > 0
    
    def test_causal_model_update_integration(self):
        """Test causal model updates from experiments."""
        config = IntegratedAgentConfig(
            agent_name="CausalTest",
            causal_config=CausalModelConfig(num_variables=3)
        )
        
        agent = IntegratedAgent(config)
        
        # Add some variables to model
        var1_id = agent.causal_model.add_variable("X", "continuous")
        var2_id = agent.causal_model.add_variable("Y", "continuous")
        
        # Create experiment data
        data = {
            var1_id: np.random.normal(1, 0.1, 20),
            var2_id: np.random.normal(0.5, 0.1, 20)
        }
        
        # Update model
        agent.causal_model.update(data)
        
        # Verify model state
        assert len(agent.causal_model.variables) == 2
        assert agent.causal_model.get_variable_statistics(var1_id) is not None
        assert agent.causal_model.get_variable_statistics(var2_id) is not None
    
    def test_experiment_executor_integration(self):
        """Test experiment executor integration."""
        config = IntegratedAgentConfig(
            agent_name="ExecutorTest",
            executor_config=ExecutorConfig(max_concurrent_experiments=2)
        )
        
        agent = IntegratedAgent(config)
        
        # Create simple experiment
        exp_config = ExperimentConfig(name="integration_test")
        
        def exp_function(exp):
            return {
                "treatment": np.random.normal(1, 0.1, 15),
                "control": np.random.normal(0, 0.1, 15)
            }
        
        experiment = SimpleExperiment(exp_config, exp_function)
        
        # Execute experiment
        result = agent.experiment_executor.execute_experiment(experiment)
        
        assert "treatment" in result
        assert "control" in result
        assert len(result["treatment"]) == 15
        assert len(result["control"]) == 15
    
    def test_experiment_analyzer_integration(self):
        """Test experiment analyzer integration."""
        config = IntegratedAgentConfig(
            agent_name="AnalyzerTest",
            analysis_config=AnalysisConfig(significance_level=0.1)
        )
        
        agent = IntegratedAgent(config)
        
        # Create experiment with results
        exp_config = ExperimentConfig(name="analysis_test")
        experiment = Experiment(exp_config)
        experiment.start()
        experiment.complete({
            "treatment": np.random.normal(0.8, 0.1, 20),
            "control": np.random.normal(0, 0.1, 20)
        })
        
        # Analyze experiment
        analysis = agent.experiment_analyzer.analyze_experiment(experiment)
        
        assert "statistical_tests" in analysis
        assert "effect_size" in analysis
        assert "significance" in analysis
        assert isinstance(analysis["statistical_tests"], dict)
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking."""
        config = IntegratedAgentConfig(
            agent_name="MetricsTest",
            loop_config=LoopConfig(max_iterations=2)
        )
        
        agent = IntegratedAgent(config)
        
        # Run a few iterations
        agent.run(max_iterations=2)
        
        # Check metrics
        metrics = agent.performance_metrics
        
        assert "total_think_time" in metrics
        assert "total_act_time" in metrics
        assert "total_learn_time" in metrics
        assert "successful_experiments" in metrics
        assert "hypotheses_generated" in metrics
        
        assert metrics["successful_experiments"] > 0
        assert metrics["hypotheses_generated"] > 0
        assert metrics["total_think_time"] >= 0
        assert metrics["total_act_time"] >= 0
        assert metrics["total_learn_time"] >= 0
    
    def test_agent_state_management(self):
        """Test agent state management."""
        config = IntegratedAgentConfig(
            agent_name="StateTest",
            loop_config=LoopConfig(max_iterations=1)
        )
        
        agent = IntegratedAgent(config)
        
        # Get initial state
        initial_state = agent.get_state()
        assert initial_state["current_iteration"] == 0
        assert initial_state["total_experiments_run"] == 0
        
        # Run one iteration
        agent.run(max_iterations=1)
        
        # Get updated state
        updated_state = agent.get_state()
        assert updated_state["current_iteration"] == 1
        assert updated_state["total_experiments_run"] > 0
        assert updated_state["total_learning_events"] > 0
    
    def test_knowledge_base_integration(self):
        """Test knowledge base integration."""
        config = IntegratedAgentConfig(
            agent_name="KnowledgeTest",
            loop_config=LoopConfig(max_iterations=1)
        )
        
        agent = IntegratedAgent(config)
        
        # Run iteration to populate knowledge base
        agent.run(max_iterations=1)
        
        # Check knowledge base
        assert "last_thought" in agent.knowledge_base
        assert "last_learning" in agent.knowledge_base
        assert len(agent.hypothesis_history) > 0
        assert len(agent.experiment_history) > 0
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = IntegratedAgentConfig(
            agent_name="ValidTest",
            loop_config=LoopConfig(max_iterations=5),
            causal_config=CausalModelConfig(num_variables=3)
        )
        
        agent = IntegratedAgent(config)
        assert agent.config.agent_name == "ValidTest"
        assert agent.config.loop_config.max_iterations == 5
        
        # Test default configuration
        default_agent = IntegratedAgent()
        assert default_agent.config.agent_name == "Evo2_Scientist"
        assert default_agent.config.loop_config.max_iterations == 100
