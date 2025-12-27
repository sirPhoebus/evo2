"""System validation tests for the complete Evo2 system."""

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


class TestSystemValidation:
    """Test suite for complete system validation."""
    
    def test_complete_system_workflow(self):
        """Test complete system workflow with all components."""
        config = IntegratedAgentConfig(
            agent_name="SystemValidation",
            loop_config=LoopConfig(max_iterations=3),
            causal_config=CausalModelConfig(num_variables=5),
            executor_config=ExecutorConfig(max_concurrent_experiments=2),
            analysis_config=AnalysisConfig(significance_level=0.1),
            enable_causal_learning=True,
            enable_experiment_analysis=True,
            enable_task_scheduling=False
        )
        
        agent = IntegratedAgent(config)
        
        # Run complete workflow
        result = agent.run(max_iterations=3)
        
        # Validate system results
        assert "agent_info" in result
        assert "run_summary" in result
        assert "causal_model_summary" in result
        assert "performance_metrics" in result
        assert "final_state" in result
        
        # Check run summary
        run_summary = result["run_summary"]
        assert run_summary["total_iterations"] == 3
        assert run_summary["successful_iterations"] == 3
        assert run_summary["total_experiments"] > 0
        
        # Check performance metrics
        metrics = result["performance_metrics"]
        assert metrics["successful_experiments"] > 0
        assert metrics["hypotheses_generated"] > 0
        assert metrics["total_think_time"] > 0
        assert metrics["total_act_time"] > 0
        assert metrics["total_learn_time"] > 0
        
        # Check final state
        final_state = result["final_state"]
        assert final_state["current_iteration"] == 3
        assert final_state["total_experiments_run"] > 0
        assert final_state["total_learning_events"] > 0
        assert final_state["knowledge_base_size"] > 0
        assert final_state["hypothesis_history_size"] > 0
    
    def test_causal_learning_integration(self):
        """Test causal learning integration across iterations."""
        config = IntegratedAgentConfig(
            agent_name="CausalLearning",
            loop_config=LoopConfig(max_iterations=5),
            causal_config=CausalModelConfig(num_variables=3),
            executor_config=ExecutorConfig(max_concurrent_experiments=1),
            enable_causal_learning=True
        )
        
        agent = IntegratedAgent(config)
        
        # Add initial variables
        var1_id = agent.causal_model.add_variable("X", "continuous")
        var2_id = agent.causal_model.add_variable("Y", "continuous")
        
        # Run iterations to enable learning
        agent.run(max_iterations=5)
        
        # Check that model has been updated
        summary = agent.causal_model.get_summary()
        assert summary["num_variables"] >= 2
        assert summary["update_count"] > 0
        
        # Check that variables have statistics
        stats1 = agent.causal_model.get_variable_statistics(var1_id)
        stats2 = agent.causal_model.get_variable_statistics(var2_id)
        assert stats1 is not None
        assert stats2 is not None
    
    def test_experiment_analysis_integration(self):
        """Test experiment analysis integration."""
        config = IntegratedAgentConfig(
            agent_name="ExperimentAnalysis",
            loop_config=LoopConfig(max_iterations=2),
            executor_config=ExecutorConfig(max_concurrent_experiments=1),
            analysis_config=AnalysisConfig(significance_level=0.05),
            enable_experiment_analysis=True
        )
        
        agent = IntegratedAgent(config)
        
        # Run workflow with analysis enabled
        result = agent.run(max_iterations=2)
        
        # Check that experiments were analyzed
        metrics = result["performance_metrics"]
        assert metrics["successful_experiments"] > 0
        
        # Check experiment history for analysis results
        assert len(agent.experiment_history) > 0
        for exp_result in agent.experiment_history:
            if exp_result.get("success", False):
                # Should have analysis if analysis is enabled
                if "result" in exp_result and "analysis" in exp_result["result"]:
                    analysis = exp_result["result"]["analysis"]
                    assert "statistical_tests" in analysis
                    assert "effect_size" in analysis
    
    def test_performance_monitoring_validation(self):
        """Test performance monitoring across the system."""
        config = IntegratedAgentConfig(
            agent_name="PerformanceMonitoring",
            loop_config=LoopConfig(max_iterations=5, track_performance=True),
            executor_config=ExecutorConfig(max_concurrent_experiments=2, enable_monitoring=True),
            enable_experiment_analysis=True
        )
        
        agent = IntegratedAgent(config)
        
        # Run with performance monitoring
        start_time = time.time()
        result = agent.run(max_iterations=5)
        end_time = time.time()
        
        # Validate performance metrics
        metrics = result["performance_metrics"]
        
        # Check timing metrics
        assert metrics["total_think_time"] > 0
        assert metrics["total_act_time"] > 0
        assert metrics["total_learn_time"] > 0
        
        # Check that total time makes sense
        total_phase_time = (metrics["total_think_time"] + 
                           metrics["total_act_time"] + 
                           metrics["total_learn_time"])
        assert total_phase_time < (end_time - start_time) + 1.0  # Allow some overhead
        
        # Check experiment metrics
        assert metrics["successful_experiments"] >= 0
        assert metrics["failed_experiments"] >= 0
        assert metrics["hypotheses_generated"] > 0
        
        # Check executor monitoring
        if agent.experiment_executor:
            monitoring_data = agent.experiment_executor.get_monitoring_data()
            assert "statistics" in monitoring_data
            assert "resource_usage" in monitoring_data
    
    def test_knowledge_accumulation_validation(self):
        """Test knowledge accumulation across iterations."""
        config = IntegratedAgentConfig(
            agent_name="KnowledgeAccumulation",
            loop_config=LoopConfig(max_iterations=4),
            causal_config=CausalModelConfig(num_variables=3),
            executor_config=ExecutorConfig(max_concurrent_experiments=1),
            enable_causal_learning=True,
            enable_experiment_analysis=True
        )
        
        agent = IntegratedAgent(config)
        
        # Track knowledge growth
        initial_state = agent.get_state()
        
        # Run iterations
        agent.run(max_iterations=4)
        
        final_state = agent.get_state()
        
        # Validate knowledge accumulation
        assert final_state["current_iteration"] == 4
        assert final_state["total_experiments_run"] > initial_state["total_experiments_run"]
        assert final_state["total_learning_events"] > initial_state["total_learning_events"]
        assert final_state["knowledge_base_size"] > initial_state["knowledge_base_size"]
        assert final_state["hypothesis_history_size"] > initial_state["hypothesis_history_size"]
        
        # Check knowledge base contents
        assert "last_thought" in agent.knowledge_base
        assert "last_learning" in agent.knowledge_base
        
        # Check hypothesis history growth
        assert len(agent.hypothesis_history) > 0
        assert len(agent.hypothesis_history) == final_state["hypothesis_history_size"]
        
        # Check experiment history growth
        assert len(agent.experiment_history) > 0
        assert len(agent.experiment_history) == final_state["experiment_history_size"]
    
    def test_error_resilience_validation(self):
        """Test system resilience to errors."""
        config = IntegratedAgentConfig(
            agent_name="ErrorResilience",
            loop_config=LoopConfig(max_iterations=3),
            executor_config=ExecutorConfig(max_concurrent_experiments=1),
            enable_experiment_analysis=True
        )
        
        agent = IntegratedAgent(config)
        
        # Simulate error conditions by running with potentially problematic configurations
        try:
            result = agent.run(max_iterations=3)
            
            # Should complete despite potential issues
            assert result["run_summary"]["total_iterations"] == 3
            
            # Should have some successful iterations
            assert result["run_summary"]["successful_iterations"] >= 0
            
            # Should track failures appropriately
            metrics = result["performance_metrics"]
            assert metrics["failed_experiments"] >= 0
            
        except Exception as e:
            # If a critical error occurs, it should be informative
            assert False, f"System should be resilient to errors: {e}"
    
    def test_configuration_flexibility_validation(self):
        """Test system flexibility with different configurations."""
        # Test minimal configuration
        minimal_config = IntegratedAgentConfig(
            agent_name="MinimalConfig",
            loop_config=LoopConfig(max_iterations=1),
            executor_config=None,
            enable_causal_learning=False,
            enable_experiment_analysis=False,
            enable_task_scheduling=False
        )
        
        minimal_agent = IntegratedAgent(minimal_config)
        minimal_result = minimal_agent.run(max_iterations=1)
        
        assert minimal_result["run_summary"]["total_iterations"] == 1
        assert minimal_result["run_summary"]["successful_iterations"] == 1
        
        # Test maximal configuration
        maximal_config = IntegratedAgentConfig(
            agent_name="MaximalConfig",
            loop_config=LoopConfig(max_iterations=2),
            causal_config=CausalModelConfig(num_variables=5),
            executor_config=ExecutorConfig(max_concurrent_experiments=2, enable_monitoring=True),
            analysis_config=AnalysisConfig(significance_level=0.01),
            enable_causal_learning=True,
            enable_experiment_analysis=True,
            enable_task_scheduling=False
        )
        
        maximal_agent = IntegratedAgent(maximal_config)
        maximal_result = maximal_agent.run(max_iterations=2)
        
        assert maximal_result["run_summary"]["total_iterations"] == 2
        assert maximal_result["run_summary"]["successful_iterations"] == 2
        assert maximal_result["performance_metrics"]["successful_experiments"] > 0
    
    def test_system_scalability_validation(self):
        """Test system scalability with different workloads."""
        # Test small workload
        small_config = IntegratedAgentConfig(
            agent_name="SmallWorkload",
            loop_config=LoopConfig(max_iterations=2),
            causal_config=CausalModelConfig(num_variables=2),
            executor_config=ExecutorConfig(max_concurrent_experiments=1)
        )
        
        small_agent = IntegratedAgent(small_config)
        start_time = time.time()
        small_result = small_agent.run(max_iterations=2)
        small_time = time.time() - start_time
        
        # Test medium workload
        medium_config = IntegratedAgentConfig(
            agent_name="MediumWorkload",
            loop_config=LoopConfig(max_iterations=3),
            causal_config=CausalModelConfig(num_variables=4),
            executor_config=ExecutorConfig(max_concurrent_experiments=2)
        )
        
        medium_agent = IntegratedAgent(medium_config)
        start_time = time.time()
        medium_result = medium_agent.run(max_iterations=3)
        medium_time = time.time() - start_time
        
        # Validate that both complete successfully
        assert small_result["run_summary"]["successful_iterations"] == 2
        assert medium_result["run_summary"]["successful_iterations"] == 3
        
        # Medium workload should take more time but still be reasonable
        assert medium_time > small_time
        assert medium_time < 30.0  # Should complete within reasonable time
    
    def test_data_consistency_validation(self):
        """Test data consistency across system components."""
        config = IntegratedAgentConfig(
            agent_name="DataConsistency",
            loop_config=LoopConfig(max_iterations=3),
            causal_config=CausalModelConfig(num_variables=3),
            executor_config=ExecutorConfig(max_concurrent_experiments=1),
            enable_causal_learning=True,
            enable_experiment_analysis=True
        )
        
        agent = IntegratedAgent(config)
        
        # Run system
        result = agent.run(max_iterations=3)
        
        # Validate data consistency
        final_state = result["final_state"]
        causal_summary = result["causal_model_summary"]
        
        # Check that state counts match
        assert final_state["causal_model"]["variables"] == causal_summary["num_variables"]
        assert final_state["causal_model"]["edges"] == causal_summary["num_edges"]
        
        # Check that knowledge base is consistent
        assert final_state["knowledge_base_size"] == len(agent.knowledge_base)
        assert final_state["hypothesis_history_size"] == len(agent.hypothesis_history)
        assert final_state["experiment_history_size"] == len(agent.experiment_history)
        
        # Check that performance metrics are consistent
        metrics = result["performance_metrics"]
        assert metrics["successful_experiments"] + metrics["failed_experiments"] == final_state["total_experiments_run"]
    
    def test_system_reproducibility_validation(self):
        """Test system reproducibility with same configuration."""
        config = IntegratedAgentConfig(
            agent_name="ReproducibilityTest",
            loop_config=LoopConfig(max_iterations=2),
            causal_config=CausalModelConfig(num_variables=3),
            executor_config=ExecutorConfig(max_concurrent_experiments=1),
            enable_experiment_analysis=True
        )
        
        # Run system twice with same configuration
        agent1 = IntegratedAgent(config)
        result1 = agent1.run(max_iterations=2)
        
        agent2 = IntegratedAgent(config)
        result2 = agent2.run(max_iterations=2)
        
        # Both should complete successfully
        assert result1["run_summary"]["successful_iterations"] == 2
        assert result2["run_summary"]["successful_iterations"] == 2
        
        # Should have same structure (though random data may differ)
        assert "agent_info" in result1 and "agent_info" in result2
        assert "run_summary" in result1 and "run_summary" in result2
        assert "performance_metrics" in result1 and "performance_metrics" in result2
        
        # Performance metrics should be in similar ranges
        metrics1 = result1["performance_metrics"]
        metrics2 = result2["performance_metrics"]
        
        # Both should have generated hypotheses
        assert metrics1["hypotheses_generated"] > 0
        assert metrics2["hypotheses_generated"] > 0
        
        # Both should have run experiments
        assert metrics1["successful_experiments"] > 0
        assert metrics2["successful_experiments"] > 0
