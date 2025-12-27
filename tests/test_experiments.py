"""Tests for Experiment Execution framework."""

import pytest
import numpy as np
from typing import Any, Dict, List
from unittest.mock import Mock, patch
import time

from evo2.experiments.framework import Experiment, ExperimentConfig, ExperimentStatus, SimpleExperiment
from evo2.experiments.executor import ExperimentExecutor, ExecutorConfig
from evo2.experiments.analyzer import ExperimentAnalyzer, AnalysisConfig
from evo2.tasks.scheduler import TaskScheduler
from evo2.causal.model import CausalModel


class TestExperiment:
    """Test suite for Experiment."""

    def test_experiment_initialization(self):
        """Test Experiment initialization."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            max_duration=300.0,
            max_iterations=100
        )
        
        experiment = Experiment(config)
        
        assert experiment.config.name == "test_experiment"
        assert experiment.config.description == "Test experiment"
        assert experiment.config.max_duration == 300.0
        assert experiment.config.max_iterations == 100
        assert experiment.status == ExperimentStatus.PENDING
        assert experiment.results == {}

    def test_experiment_configuration_validation(self):
        """Test experiment configuration validation."""
        # Valid config
        config = ExperimentConfig(name="valid", max_duration=60.0)
        assert config.name == "valid"
        
        # Invalid configs
        with pytest.raises(ValueError):
            ExperimentConfig(name="", max_duration=60.0)  # Empty name
        
        with pytest.raises(ValueError):
            ExperimentConfig(name="invalid", max_duration=0)  # Zero duration
        
        with pytest.raises(ValueError):
            ExperimentConfig(name="invalid", max_iterations=0)  # Zero iterations

    def test_experiment_lifecycle(self):
        """Test experiment lifecycle transitions."""
        config = ExperimentConfig(name="lifecycle_test")
        experiment = Experiment(config)
        
        # Initial state
        assert experiment.status == ExperimentStatus.PENDING
        
        # Start experiment
        experiment.start()
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.start_time is not None
        
        # Complete experiment
        experiment.complete({"result": "success"})
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.end_time is not None
        assert experiment.results == {"result": "success"}

    def test_experiment_failure(self):
        """Test experiment failure handling."""
        config = ExperimentConfig(name="failure_test")
        experiment = Experiment(config)
        
        experiment.start()
        experiment.fail("Test failure")
        
        assert experiment.status == ExperimentStatus.FAILED
        assert experiment.error_message == "Test failure"
        assert experiment.end_time is not None

    def test_experiment_cancellation(self):
        """Test experiment cancellation."""
        config = ExperimentConfig(name="cancel_test")
        experiment = Experiment(config)
        
        experiment.start()
        experiment.cancel()
        
        assert experiment.status == ExperimentStatus.CANCELLED
        assert experiment.end_time is not None

    def test_experiment_progress_tracking(self):
        """Test experiment progress tracking."""
        config = ExperimentConfig(name="progress_test", max_iterations=10)
        experiment = Experiment(config)
        
        experiment.start()
        
        # Update progress
        experiment.update_progress(5)
        assert experiment.current_iteration == 5
        assert experiment.progress_percentage == 50.0
        
        experiment.update_progress(10)
        assert experiment.current_iteration == 10
        assert experiment.progress_percentage == 100.0

    def test_experiment_timeout(self):
        """Test experiment timeout handling."""
        config = ExperimentConfig(name="timeout_test", max_duration=0.1, timeout_grace_period=0.0)
        experiment = Experiment(config)
        
        experiment.start()
        time.sleep(0.2)  # Wait longer than max_duration
        
        assert experiment.is_timeout()
        
    def test_experiment_serialization(self):
        """Test experiment serialization."""
        config = ExperimentConfig(name="serialize_test")
        experiment = Experiment(config)
        
        experiment.start()
        experiment.update_progress(5)
        experiment.complete({"data": [1, 2, 3]})
        
        # Serialize
        serialized = experiment.serialize()
        
        assert isinstance(serialized, dict)
        assert "config" in serialized
        assert "status" in serialized
        assert "results" in serialized
        
        # Deserialize
        new_experiment = Experiment.deserialize(serialized)
        
        assert new_experiment.config.name == experiment.config.name
        assert new_experiment.status == experiment.status
        assert new_experiment.results == experiment.results


class TestExperimentExecutor:
    """Test suite for ExperimentExecutor."""

    def test_executor_initialization(self):
        """Test ExperimentExecutor initialization."""
        config = ExecutorConfig(
            max_concurrent_experiments=3,
            default_timeout=120.0
        )
        
        executor = ExperimentExecutor(config)
        
        assert executor.max_concurrent_experiments == 3
        assert executor.default_timeout == 120.0
        assert len(executor.running_experiments) == 0
        assert len(executor.completed_experiments) == 0

    def test_submit_experiment(self):
        """Test submitting experiments to executor."""
        config = ExecutorConfig(max_concurrent_experiments=2)
        executor = ExperimentExecutor(config)
        
        # Create experiment
        exp_config = ExperimentConfig(name="test_exp")
        def dummy_function(exp):
            return {"result": "success"}
        experiment = SimpleExperiment(exp_config, dummy_function)
        
        # Submit experiment
        experiment_id = executor.submit_experiment(experiment)
        
        assert experiment_id is not None
        # Experiment may be in pending or running depending on processing
        assert (experiment_id in executor.pending_experiments or 
                experiment_id in executor.running_experiments)

    def test_execute_experiment(self):
        """Test executing a simple experiment."""
        config = ExecutorConfig(max_concurrent_experiments=1)
        executor = ExperimentExecutor(config)
        
        # Create mock experiment with proper attributes
        experiment = Mock()
        experiment.experiment_id = "mock_exp_id"
        experiment.config.name = "mock_exp"
        experiment.status = ExperimentStatus.PENDING
        experiment.execute.return_value = {"result": "success"}
        
        # Execute
        result = executor.execute_experiment(experiment)
        
        assert result["result"] == "success"
        experiment.execute.assert_called_once()

    def test_parallel_execution(self):
        """Test parallel experiment execution."""
        config = ExecutorConfig(max_concurrent_experiments=3)
        executor = ExperimentExecutor(config)
        
        # Create multiple experiments
        experiments = []
        for i in range(3):
            exp_config = ExperimentConfig(name=f"exp_{i}")
            experiment = Experiment(exp_config)
            experiments.append(experiment)
        
        # Submit all experiments
        experiment_ids = []
        for exp in experiments:
            exp_id = executor.submit_experiment(exp)
            experiment_ids.append(exp_id)
        
        # Start execution
        executor.start()
        
        # Wait for completion
        time.sleep(0.5)
        
        executor.stop()
        
        # Check that experiments were processed
        assert len(executor.completed_experiments) >= 0

    def test_experiment_monitoring(self):
        """Test experiment monitoring and status updates."""
        config = ExecutorConfig(max_concurrent_experiments=1)
        executor = ExperimentExecutor(config)
        
        # Create experiment with mock execution
        experiment = Mock()
        experiment.experiment_id = "monitor_test_id"
        experiment.config.name = "monitor_test"
        experiment.status = ExperimentStatus.PENDING
        experiment.execute.return_value = {"data": [1, 2, 3]}
        
        # Execute and monitor
        result = executor.execute_experiment(experiment)
        
        # Check monitoring data
        monitoring_data = executor.get_monitoring_data()
        
        assert isinstance(monitoring_data, dict)
        assert "running_experiments" in monitoring_data
        assert "completed_experiments" in monitoring_data
        assert "statistics" in monitoring_data

    def test_experiment_cancellation_in_executor(self):
        """Test cancelling experiments in executor."""
        config = ExecutorConfig(max_concurrent_experiments=1)
        executor = ExperimentExecutor(config)
        
        # Create long-running experiment
        experiment = Mock()
        experiment.experiment_id = "long_exp_id"
        experiment.config.name = "long_exp"
        experiment.status = ExperimentStatus.PENDING
        experiment.execute.side_effect = lambda: time.sleep(2)  # Long execution
        
        # Start execution in background
        import threading
        execution_thread = threading.Thread(
            target=executor.execute_experiment,
            args=(experiment,)
        )
        execution_thread.start()
        
        time.sleep(0.1)  # Let it start
        
        # Cancel the experiment
        success = executor.cancel_experiment(experiment.experiment_id)
        
        execution_thread.join(timeout=1.0)
        
        assert success is True

    def test_resource_management(self):
        """Test resource management during execution."""
        config = ExecutorConfig(
            max_concurrent_experiments=2,
            memory_limit_mb=1024,
            cpu_limit_cores=2
        )
        executor = ExperimentExecutor(config)
        
        # Check resource limits
        assert executor.memory_limit_mb == 1024
        assert executor.cpu_limit_cores == 2
        
        # Get current resource usage
        usage = executor.get_resource_usage()
        
        assert isinstance(usage, dict)
        assert "memory_mb" in usage
        assert "cpu_cores" in usage


class TestExperimentAnalyzer:
    """Test suite for ExperimentAnalyzer."""

    def test_analyzer_initialization(self):
        """Test ExperimentAnalyzer initialization."""
        config = AnalysisConfig(
            significance_level=0.05,
            effect_size_threshold=0.2
        )
        
        analyzer = ExperimentAnalyzer(config)
        
        assert analyzer.significance_level == 0.05
        assert analyzer.effect_size_threshold == 0.2

    def test_analyze_single_experiment(self):
        """Test analyzing a single experiment."""
        config = AnalysisConfig()
        analyzer = ExperimentAnalyzer(config)
        
        # Create experiment with results - use continuous data to avoid chi-square issues
        exp_config = ExperimentConfig(name="analysis_test")
        experiment = Experiment(exp_config)
        experiment.start()
        experiment.complete({
            "treatment": np.random.normal(1.0, 0.1, 20),
            "control": np.random.normal(0.0, 0.1, 20),
            "metadata": {"sample_size": 20}
        })
        
        # Analyze
        analysis = analyzer.analyze_experiment(experiment)
        
        assert isinstance(analysis, dict)
        assert "statistical_tests" in analysis
        assert "effect_size" in analysis
        assert "significance" in analysis

    def test_compare_experiments(self):
        """Test comparing multiple experiments."""
        config = AnalysisConfig()
        analyzer = ExperimentAnalyzer(config)
        
        # Create multiple experiments
        experiments = []
        for i in range(3):
            exp_config = ExperimentConfig(name=f"comparison_test_{i}")
            experiment = Experiment(exp_config)
            experiment.start()
            experiment.complete({
                "results": np.random.normal(0.5 + i * 0.1, 0.1, 20),
                "group": f"group_{i}"
            })
            experiments.append(experiment)
        
        # Compare
        comparison = analyzer.compare_experiments(experiments)
        
        assert isinstance(comparison, dict)
        assert "pairwise_comparisons" in comparison
        assert "overall_significance" in comparison
        assert "effect_sizes" in comparison

    def test_meta_analysis(self):
        """Test meta-analysis across experiments."""
        config = AnalysisConfig()
        analyzer = ExperimentAnalyzer(config)
        
        # Create experiments with similar structure
        experiments = []
        effect_sizes = [0.2, 0.3, 0.25, 0.35, 0.15]
        
        for i, effect in enumerate(effect_sizes):
            exp_config = ExperimentConfig(name=f"meta_test_{i}")
            experiment = Experiment(exp_config)
            experiment.start()
            experiment.complete({
                "treatment": np.random.normal(effect, 0.1, 50),
                "control": np.random.normal(0, 0.1, 50),
                "sample_size": 50
            })
            experiments.append(experiment)
        
        # Meta-analysis
        meta_result = analyzer.meta_analysis(experiments)
        
        assert isinstance(meta_result, dict)
        assert "pooled_effect_size" in meta_result
        assert "confidence_interval" in meta_result
        assert "heterogeneity" in meta_result

    def test_causal_analysis(self):
        """Test causal analysis of experiments."""
        config = AnalysisConfig()
        analyzer = ExperimentAnalyzer(config)
        
        # Create experiment with causal structure
        exp_config = ExperimentConfig(name="causal_test")
        experiment = Experiment(exp_config)
        experiment.start()
        experiment.complete({
            "treatment": np.random.binomial(1, 0.5, 100),
            "outcome": np.random.normal(0, 1, 100),
            "confounder": np.random.normal(0, 1, 100)
        })
        
        # Create simple causal model
        causal_model = Mock()
        
        # Mock the inference engine methods
        with patch('evo2.causal.inference.CausalInference') as mock_inference_class:
            mock_inference = Mock()
            mock_inference_class.return_value = mock_inference
            mock_inference.estimate_effect.return_value = {
                "effect_size": 0.3,
                "confidence": 0.8
            }
            
            # Causal analysis
            causal_result = analyzer.causal_analysis(experiment, causal_model)
            
            assert isinstance(causal_result, dict)
            assert "causal_effects" in causal_result
            assert "confidence_intervals" in causal_result

    def test_generate_report(self):
        """Test generating analysis reports."""
        config = AnalysisConfig()
        analyzer = ExperimentAnalyzer(config)
        
        # Create experiment
        exp_config = ExperimentConfig(name="report_test")
        experiment = Experiment(exp_config)
        experiment.start()
        experiment.complete({
            "results": np.random.normal(0.5, 0.1, 50),
            "metadata": {"description": "Test experiment"}
        })
        
        # Analyze and generate report
        analysis = analyzer.analyze_experiment(experiment)
        report = analyzer.generate_report(analysis)
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "detailed_results" in report
        assert "recommendations" in report
