"""Core integration tests for the complete Evo2 system."""

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


class TestCoreIntegration:
    """Test suite for core system integration."""
    
    def test_integrated_agent_basic_workflow(self):
        """Test basic integrated agent workflow without experiment execution."""
        config = IntegratedAgentConfig(
            agent_name="BasicTest",
            loop_config=LoopConfig(max_iterations=2),
            causal_config=CausalModelConfig(num_variables=3),
            executor_config=None,  # Disable executor
            enable_experiment_analysis=False
        )
        
        agent = IntegratedAgent(config)
        
        # Test think phase
        thought = agent.think()
        assert "iteration" in thought
        assert "model_summary" in thought
        assert "hypotheses" in thought
        assert "experiment_plans" in thought
        
        # Test act phase (without execution)
        action_result = agent.act()
        assert "iteration" in action_result
        assert "thought" in action_result
        assert "experiment_results" in action_result
        
        # Test learn phase (without data)
        learning_result = agent.learn(action_result)
        assert "iteration" in learning_result
        assert "learning_events" in learning_result
        
        # Check agent state
        state = agent.get_state()
        assert state["current_iteration"] == 0
        assert state["total_experiments_run"] == 0
        assert state["hypothesis_history_size"] > 0
    
    def test_causal_model_integration_basic(self):
        """Test basic causal model integration."""
        config = IntegratedAgentConfig(
            agent_name="CausalBasicTest",
            causal_config=CausalModelConfig(num_variables=3)
        )
        
        agent = IntegratedAgent(config)
        
        # Add variables manually
        var1_id = agent.causal_model.add_variable("X", "continuous")
        var2_id = agent.causal_model.add_variable("Y", "continuous")
        
        # Add edge
        edge_id = agent.causal_model.add_edge(var1_id, var2_id, strength=0.5)
        
        # Test model summary
        summary = agent.causal_model.get_summary()
        assert summary["num_variables"] == 2
        assert summary["num_edges"] == 1
        
        # Test variable statistics
        stats = agent.causal_model.get_variable_statistics(var1_id)
        assert stats is not None
        assert "mean" in stats
        assert "std" in stats
        
        # Test model update
        data = {
            var1_id: np.random.normal(1, 0.1, 20),
            var2_id: np.random.normal(0.5, 0.1, 20)
        }
        agent.causal_model.update(data)
        
        # Check updated statistics
        updated_stats = agent.causal_model.get_variable_statistics(var1_id)
        assert updated_stats["count"] == 20
    
    def test_experiment_framework_integration(self):
        """Test experiment framework integration."""
        config = IntegratedAgentConfig(
            agent_name="ExpFrameworkTest",
            executor_config=ExecutorConfig(max_concurrent_experiments=1)
        )
        
        agent = IntegratedAgent(config)
        
        # Create simple experiment
        exp_config = ExperimentConfig(name="framework_test")
        
        def exp_function(exp):
            return {
                "treatment": np.random.normal(1, 0.1, 15),
                "control": np.random.normal(0, 0.1, 15)
            }
        
        experiment = SimpleExperiment(exp_config, exp_function)
        
        # Test experiment lifecycle
        assert experiment.status.name == "PENDING"
        
        experiment.start()
        assert experiment.status.name == "RUNNING"
        
        result = exp_function(experiment)
        experiment.complete(result)
        assert experiment.status.name == "COMPLETED"
        assert experiment.results == result
    
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
    
    def test_agent_configuration_validation(self):
        """Test agent configuration validation."""
        # Test default configuration
        default_agent = IntegratedAgent()
        assert default_agent.config.agent_name == "Evo2_Scientist"
        assert default_agent.config.loop_config.max_iterations == 100
        assert default_agent.config.causal_config.num_variables == 20
        
        # Test custom configuration
        custom_config = IntegratedAgentConfig(
            agent_name="CustomTest",
            loop_config=LoopConfig(max_iterations=5),
            causal_config=CausalModelConfig(num_variables=3)
        )
        
        custom_agent = IntegratedAgent(custom_config)
        assert custom_agent.config.agent_name == "CustomTest"
        assert custom_agent.config.loop_config.max_iterations == 5
        assert custom_agent.config.causal_config.num_variables == 3
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking."""
        config = IntegratedAgentConfig(
            agent_name="MetricsTest",
            loop_config=LoopConfig(max_iterations=2),
            executor_config=None,
            enable_experiment_analysis=False
        )
        
        agent = IntegratedAgent(config)
        
        # Run think phase
        thought = agent.think()
        
        # Check metrics
        metrics = agent.performance_metrics
        assert "total_think_time" in metrics
        assert "total_act_time" in metrics
        assert "total_learn_time" in metrics
        assert "successful_experiments" in metrics
        assert "hypotheses_generated" in metrics
        
        assert metrics["hypotheses_generated"] > 0
        assert metrics["total_think_time"] >= 0
    
    def test_knowledge_base_integration(self):
        """Test knowledge base integration."""
        config = IntegratedAgentConfig(
            agent_name="KnowledgeTest",
            loop_config=LoopConfig(max_iterations=1),
            executor_config=None,
            enable_experiment_analysis=False
        )
        
        agent = IntegratedAgent(config)
        
        # Run think phase
        thought = agent.think()
        
        # Check knowledge base
        assert "last_thought" in agent.knowledge_base
        assert len(agent.hypothesis_history) > 0
        assert agent.knowledge_base["last_thought"] == thought
        
        # Check hypothesis history
        assert len(agent.hypothesis_history) == len(thought["hypotheses"])
        for i, hypothesis in enumerate(thought["hypotheses"]):
            assert agent.hypothesis_history[i] == hypothesis
    
    def test_agent_state_management(self):
        """Test agent state management."""
        config = IntegratedAgentConfig(
            agent_name="StateTest",
            loop_config=LoopConfig(max_iterations=1),
            executor_config=None,
            enable_experiment_analysis=False
        )
        
        agent = IntegratedAgent(config)
        
        # Get initial state
        initial_state = agent.get_state()
        assert initial_state["current_iteration"] == 0
        assert initial_state["total_experiments_run"] == 0
        assert initial_state["total_learning_events"] == 0
        
        # Run think phase
        agent.think()
        
        # Get updated state
        updated_state = agent.get_state()
        assert updated_state["current_iteration"] == 0
        assert updated_state["total_experiments_run"] == 0
        assert updated_state["total_learning_events"] == 0
        assert updated_state["knowledge_base_size"] > 0
        assert updated_state["hypothesis_history_size"] > 0
    
    def test_error_handling_integration(self):
        """Test error handling in integrated system."""
        config = IntegratedAgentConfig(
            agent_name="ErrorTest",
            loop_config=LoopConfig(max_iterations=1),
            executor_config=None,
            enable_experiment_analysis=False
        )
        
        agent = IntegratedAgent(config)
        
        # Test think phase with error simulation
        # (This would normally be tested with mocking, but we'll check basic error handling)
        try:
            thought = agent.think()
            assert "error" not in thought  # Should not error under normal conditions
        except Exception as e:
            # If an error occurs, it should be caught and returned in the result
            assert False, f"Unexpected error in think phase: {e}"
    
    def test_component_interaction(self):
        """Test interaction between components."""
        config = IntegratedAgentConfig(
            agent_name="InteractionTest",
            loop_config=LoopConfig(max_iterations=1),
            causal_config=CausalModelConfig(num_variables=3),
            executor_config=None,
            enable_experiment_analysis=False
        )
        
        agent = IntegratedAgent(config)
        
        # Test causal model and think phase interaction
        thought = agent.think()
        model_summary = thought["model_summary"]
        
        assert "summary" in model_summary
        assert "gaps" in model_summary
        assert "complexity" in model_summary
        
        # The think phase should identify gaps in the model
        gaps = model_summary["gaps"]
        assert isinstance(gaps, list)
        
        # Should generate hypotheses based on gaps
        hypotheses = thought["hypotheses"]
        assert isinstance(hypotheses, list)
        
        # Should generate experiment plans based on hypotheses
        plans = thought["experiment_plans"]
        assert isinstance(plans, list)
        
        # Each plan should be linked to a hypothesis
        for plan in plans:
            assert "hypothesis" in plan
            assert "type" in plan
            assert "description" in plan
