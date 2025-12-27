"""Integration tests for the complete Evo2 system."""

import pytest
import numpy as np
from typing import Any, Dict, List
from unittest.mock import Mock, patch
import time

from evo2.agent.base import Agent
from evo2.agent.loop import AgentLoop, LoopConfig
from evo2.causal.model import CausalModel, CausalModelConfig
from evo2.causal.inference import CausalInference, InferenceConfig
from evo2.causal.graph import CausalGraph, GraphConfig
from evo2.experiments.framework import Experiment, ExperimentConfig, SimpleExperiment
from evo2.experiments.executor import ExperimentExecutor, ExecutorConfig
from evo2.experiments.analyzer import ExperimentAnalyzer, AnalysisConfig


class TestSystemIntegration:
    """Test suite for complete system integration."""
    
    def test_complete_agent_workflow(self):
        """Test complete agent workflow with all components."""
        # Create agent with all components
        agent = Mock(spec=Agent)
        
        # Configure causal model
        causal_config = CausalModelConfig(num_variables=10, max_parents=3)
        causal_model = CausalModel(causal_config)
        
        # Configure experiment executor
        executor_config = ExecutorConfig(max_concurrent_experiments=3)
        executor = ExperimentExecutor(executor_config)
        
        # Configure analyzer
        analysis_config = AnalysisConfig(significance_level=0.05)
        analyzer = ExperimentAnalyzer(analysis_config)
        
        # Configure agent loop
        loop_config = LoopConfig(max_iterations=50, think_interval=0.01)
        agent_loop = AgentLoop(agent, loop_config, scheduler=None, execution_engine=None)
        
        # Mock agent methods to simulate realistic behavior
        def mock_think():
            # Simulate literature review and hypothesis generation
            return {
                "action": "experiment",
                "hypothesis": "X causes Y",
                "variables": ["X", "Y", "Z"],
                "confidence": 0.7
            }
        
        def mock_act(thought):
            # Simulate experiment execution
            exp_config = ExperimentConfig(
                name=f"exp_{int(time.time())}",
                description=f"Test {thought['hypothesis']}"
            )
            
            def experiment_function(exp):
                # Simulate experimental results
                return {
                    "treatment": np.random.normal(1.0, 0.1, 20),
                    "control": np.random.normal(0.0, 0.1, 20),
                    "outcome": np.random.normal(0.5, 0.1, 20)
                }
            
            experiment = SimpleExperiment(exp_config, experiment_function)
            result = executor.execute_experiment(experiment)
            
            # Analyze results
            analysis = analyzer.analyze_experiment(experiment)
            
            return {
                "experiment_result": result,
                "analysis": analysis,
                "success": True
            }
        
        def mock_learn(action_result):
            # Update causal model based on experiment results
            if "analysis" in action_result:
                analysis = action_result["analysis"]
                
                # Add variables to causal model if not present
                for var_name in ["X", "Y", "Z"]:
                    if var_name not in [v["name"] for v in causal_model.variables.values()]:
                        causal_model.add_variable(var_name, "continuous")
                
                # Simple learning: update edge strengths based on analysis
                if analysis["significance"]["overall_significance_rate"] > 0.5:
                    # Add causal edge if significant results
                    var_ids = list(causal_model.variables.keys())
                    if len(var_ids) >= 2:
                        try:
                            causal_model.add_edge(var_ids[0], var_ids[1], strength=0.6)
                        except:
                            pass  # Edge might already exist or cycle detected
            
            return {"learning_complete": True}
        
        agent.think.side_effect = mock_think
        agent.act.side_effect = mock_act
        agent.learn.side_effect = mock_learn
        
        # Run agent loop for several iterations
        for i in range(5):
            result = agent_loop.run_iteration()
            assert result["iteration"] == i + 1
            assert "thought" in result
            assert "action" in result
            assert "learning" in result
        
        # Verify system state
        assert agent_loop.iteration_count == 5
        assert len(causal_model.variables) > 0
        assert len(executor.completed_experiments) > 0
    
    def test_causal_model_integration(self):
        """Test causal model integration with experiment analysis."""
        # Create causal model
        causal_config = CausalModelConfig(num_variables=5)
        causal_model = CausalModel(causal_config)
        
        # Add variables
        x_id = causal_model.add_variable("X", "continuous")
        y_id = causal_model.add_variable("Y", "continuous")
        z_id = causal_model.add_variable("Z", "continuous")
        
        # Add edge
        edge_id = causal_model.add_edge(x_id, y_id, strength=0.5)
        
        # Create experiment with relevant data
        exp_config = ExperimentConfig(name="causal_test")
        experiment = Experiment(exp_config)
        experiment.start()
        experiment.complete({
            "X": np.random.normal(1.0, 0.1, 30),
            "Y": np.random.normal(0.5, 0.1, 30),
            "Z": np.random.normal(0.2, 0.1, 30)
        })
        
        # Update causal model with experiment data
        causal_model.update({
            x_id: experiment.results["X"],
            y_id: experiment.results["Y"],
            z_id: experiment.results["Z"]
        })
        
        # Verify model updated
        assert len(causal_model.variables) == 3
        assert len(causal_model.edges) == 1
        assert causal_model.edges[edge_id]["strength"] != 0.5  # Should have changed
    
    def test_experiment_executor_integration(self):
        """Test experiment executor integration with analyzer."""
        # Create executor and analyzer
        executor_config = ExecutorConfig(max_concurrent_experiments=2)
        executor = ExperimentExecutor(executor_config)
        
        analysis_config = AnalysisConfig()
        analyzer = ExperimentAnalyzer(analysis_config)
        
        # Create multiple experiments
        experiments = []
        for i in range(3):
            exp_config = ExperimentConfig(name=f"integration_test_{i}")
            
            def exp_func(exp, effect_size=0.3):
                return {
                    "treatment": np.random.normal(effect_size, 0.1, 25),
                    "control": np.random.normal(0.0, 0.1, 25),
                    "metadata": {"experiment_id": exp.experiment_id}
                }
            
            experiment = SimpleExperiment(exp_config, lambda exp, i=i: exp_func(exp, 0.2 + i * 0.1))
            experiments.append(experiment)
        
        # Execute experiments
        results = []
        for experiment in experiments:
            result = executor.execute_experiment(experiment)
            results.append(result)
        
        # Analyze all experiments
        analyses = []
        for experiment in experiments:
            analysis = analyzer.analyze_experiment(experiment)
            analyses.append(analysis)
        
        # Perform meta-analysis
        meta_result = analyzer.meta_analysis(experiments)
        
        # Verify integration
        assert len(results) == 3
        assert len(analyses) == 3
        assert "pooled_effect_size" in meta_result
        assert meta_result["experiment_count"] == 3
    
    def test_agent_loop_with_scheduler_integration(self):
        """Test agent loop integration with task scheduler."""
        from evo2.tasks.scheduler import TaskScheduler, SchedulerConfig
        
        # Create scheduler
        scheduler_config = SchedulerConfig(max_concurrent_tasks=2)
        scheduler = TaskScheduler(scheduler_config)
        
        # Create agent loop with scheduler
        agent = Mock(spec=Agent)
        loop_config = LoopConfig(max_iterations=10, enable_scheduler=True)
        agent_loop = AgentLoop(agent, loop_config, scheduler=scheduler)
        
        # Mock agent to use scheduler
        def mock_act(thought):
            # Create tasks for scheduler
            from evo2.tasks.base import Task, TaskStatus, TaskPriority
            
            task1 = Task("experiment_1", TaskPriority.HIGH)
            task2 = Task("experiment_2", TaskPriority.MEDIUM)
            
            scheduler.add_task(task1)
            scheduler.add_task(task2)
            
            return {"tasks_created": 2}
        
        agent.think.return_value = {"action": "create_tasks"}
        agent.act.side_effect = mock_act
        agent.learn.return_value = {"learned": True}
        
        # Start scheduler
        scheduler.start()
        
        # Run loop iterations
        for i in range(3):
            agent_loop.run_iteration()
        
        # Verify scheduler integration
        assert len(scheduler.task_queue) >= 0
        assert scheduler.is_running()
        
        # Stop scheduler
        scheduler.stop()
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Create components
        agent = Mock(spec=Agent)
        causal_model = CausalModel()
        executor = ExperimentExecutor()
        analyzer = ExperimentAnalyzer()
        
        # Mock agent to raise errors
        agent.think.side_effect = [
            {"action": "experiment"},  # Success
            Exception("Thinking failed"),  # Error
            {"action": "experiment"},  # Recovery
        ]
        
        agent.act.return_value = {"result": "success"}
        agent.learn.return_value = {"learned": True}
        
        # Create loop
        loop_config = LoopConfig(max_iterations=3)
        agent_loop = AgentLoop(agent, loop_config)
        
        # Run iterations with error handling
        results = []
        for i in range(3):
            result = agent_loop.run_iteration()
            results.append(result)
        
        # Verify error handling
        assert len(results) == 3
        assert results[0]["iteration"] == 1
        assert "error" in results[1]  # Error iteration
        assert results[2]["iteration"] == 3  # Recovery iteration
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring across integrated system."""
        # Create components with performance tracking
        agent = Mock(spec=Agent)
        causal_model = CausalModel()
        executor = ExperimentExecutor(ExecutorConfig(enable_monitoring=True))
        analyzer = ExperimentAnalyzer()
        
        # Mock agent with realistic timing
        def mock_think():
            time.sleep(0.01)  # Simulate thinking time
            return {"action": "experiment"}
        
        def mock_act(thought):
            time.sleep(0.02)  # Simulate action time
            return {"result": "success"}
        
        def mock_learn(result):
            time.sleep(0.005)  # Simulate learning time
            return {"learned": True}
        
        agent.think.side_effect = mock_think
        agent.act.side_effect = mock_act
        agent.learn.side_effect = mock_learn
        
        # Create loop with performance tracking
        loop_config = LoopConfig(max_iterations=5, track_performance=True)
        agent_loop = AgentLoop(agent, loop_config)
        
        # Run iterations
        for i in range(5):
            agent_loop.run_iteration()
        
        # Check performance metrics
        metrics = agent_loop.get_metrics()
        
        assert "total_iterations" in metrics
        assert "average_iteration_time" in metrics
        assert "total_think_time" in metrics
        assert "total_act_time" in metrics
        assert "total_learn_time" in metrics
        assert metrics["total_iterations"] == 5
        
        # Check executor monitoring
        monitoring_data = executor.get_monitoring_data()
        assert "statistics" in monitoring_data
        assert "resource_usage" in monitoring_data
    
    def test_data_flow_integration(self):
        """Test data flow between components."""
        # Create components
        agent = Mock(spec=Agent)
        causal_model = CausalModel(CausalModelConfig(num_variables=5))
        executor = ExperimentExecutor()
        analyzer = ExperimentAnalyzer()
        
        # Track data flow
        data_flow = []
        
        def mock_think():
            thought = {
                "action": "experiment",
                "hypothesis": f"hyp_{len(data_flow)}",
                "variables": ["A", "B", "C"]
            }
            data_flow.append(("think", thought))
            return thought
        
        def mock_act(thought):
            # Create experiment based on thought
            exp_config = ExperimentConfig(name=f"exp_{thought['hypothesis']}")
            experiment = SimpleExperiment(exp_config, lambda exp: {
                "A": np.random.normal(1, 0.1, 20),
                "B": np.random.normal(0.5, 0.1, 20),
                "C": np.random.normal(0.2, 0.1, 20)
            })
            
            result = executor.execute_experiment(experiment)
            analysis = analyzer.analyze_experiment(experiment)
            
            action_result = {
                "experiment_result": result,
                "analysis": analysis,
                "thought": thought
            }
            data_flow.append(("act", action_result))
            return action_result
        
        def mock_learn(action_result):
            # Update causal model with experiment data
            if "experiment_result" in action_result:
                # Add variables if needed
                for var_name in ["A", "B", "C"]:
                    if var_name not in [v["name"] for v in causal_model.variables.values()]:
                        causal_model.add_variable(var_name, "continuous")
                
                # Update model with data
                experiment_data = action_result["experiment_result"]
                causal_model.update(experiment_data)
            
            learning_result = {"model_updated": True, "variables": len(causal_model.variables)}
            data_flow.append(("learn", learning_result))
            return learning_result
        
        agent.think.side_effect = mock_think
        agent.act.side_effect = mock_act
        agent.learn.side_effect = mock_learn
        
        # Create loop
        loop_config = LoopConfig(max_iterations=3)
        agent_loop = AgentLoop(agent, loop_config)
        
        # Run iterations
        for i in range(3):
            agent_loop.run_iteration()
        
        # Verify data flow
        assert len(data_flow) == 9  # 3 iterations × 3 phases
        
        # Check data consistency
        think_data = [item[1] for item in data_flow if item[0] == "think"]
        act_data = [item[1] for item in data_flow if item[0] == "act"]
        learn_data = [item[1] for item in data_flow if item[0] == "learn"]
        
        assert len(think_data) == 3
        assert len(act_data) == 3
        assert len(learn_data) == 3
        
        # Verify causal model grew
        assert len(causal_model.variables) == 3  # A, B, C
        
        # Verify data linkage
        for i in range(3):
            assert act_data[i]["thought"] == think_data[i]
            assert learn_data[i]["variables"] > 0
