"""Integrated Agent system combining all Phase 3 components."""

import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

from .base import Agent
from .loop import AgentLoop, LoopConfig
from ..causal.model import CausalModel, CausalModelConfig
from ..causal.inference import CausalInference, InferenceConfig
from ..causal.graph import CausalGraph, GraphConfig
from ..experiments.framework import Experiment, ExperimentConfig, SimpleExperiment
from ..experiments.executor import ExperimentExecutor, ExecutorConfig
from ..experiments.analyzer import ExperimentAnalyzer, AnalysisConfig
from ..tasks.scheduler import TaskScheduler, SchedulerConfig


@dataclass
class IntegratedAgentConfig:
    """Configuration for the integrated agent system."""
    
    # Agent loop configuration
    loop_config: LoopConfig = field(default_factory=lambda: LoopConfig(
        max_iterations=100,
        think_interval=0.1,
        track_performance=True
    ))
    
    # Causal model configuration
    causal_config: CausalModelConfig = field(default_factory=lambda: CausalModelConfig(
        num_variables=20,
        max_parents=5,
        learning_rate=0.01
    ))
    
    # Experiment executor configuration
    executor_config: ExecutorConfig = field(default_factory=lambda: ExecutorConfig(
        max_concurrent_experiments=3,
        enable_monitoring=True
    ))
    
    # Analysis configuration
    analysis_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        significance_level=0.05,
        effect_size_threshold=0.2
    ))
    
    # Scheduler configuration
    scheduler_config: Optional[SchedulerConfig] = field(default_factory=lambda: SchedulerConfig(
        max_concurrent_tasks=5
    ))
    
    # Agent-specific configuration
    agent_name: str = "Evo2_Scientist"
    research_domain: str = "General"
    experiment_budget: int = 1000
    learning_rate: float = 0.01
    
    # Integration settings
    enable_causal_learning: bool = True
    enable_experiment_analysis: bool = True
    enable_task_scheduling: bool = True
    save_state_interval: int = 10


class IntegratedAgent:
    """Integrated agent combining causal modeling, experiment execution, and learning.
    
    This agent implements the complete scientific research workflow:
    1. Literature review and hypothesis generation (think)
    2. Experiment design and execution (act)
    3. Model updating and learning (learn)
    """
    
    def __init__(self, config: Optional[IntegratedAgentConfig] = None):
        """Initialize the integrated agent.
        
        Args:
            config: Optional configuration for the agent.
        """
        self.config = config or IntegratedAgentConfig()
        self.agent_id = str(uuid.uuid4())
        
        # Initialize core components
        self.causal_model = CausalModel(self.config.causal_config)
        self.causal_inference = CausalInference(InferenceConfig())
        self.causal_graph = CausalGraph(GraphConfig())
        
        self.experiment_executor = ExperimentExecutor(self.config.executor_config)
        self.experiment_analyzer = ExperimentAnalyzer(self.config.analysis_config)
        
        # Initialize scheduler if enabled
        self.scheduler = None
        if self.config.enable_task_scheduling and self.config.scheduler_config:
            self.scheduler = TaskScheduler(self.config.scheduler_config)
        
        # Agent state
        self.current_iteration = 0
        self.total_experiments_run = 0
        self.total_learning_events = 0
        self.knowledge_base: Dict[str, Any] = {}
        self.hypothesis_history: List[Dict[str, Any]] = []
        self.experiment_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.performance_metrics = {
            "total_think_time": 0.0,
            "total_act_time": 0.0,
            "total_learn_time": 0.0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "hypotheses_generated": 0,
            "causal_discoveries": 0
        }
        
        self.logger = logging.getLogger(f"IntegratedAgent.{self.config.agent_name}")
        
        # Create agent loop
        self.agent_loop = AgentLoop(
            agent=self,
            config=self.config.loop_config,
            scheduler=self.scheduler
        )
    
    def think(self) -> Dict[str, Any]:
        """Generate hypotheses and research directions.
        
        Returns:
            Thought dictionary containing hypotheses and research plans.
        """
        start_time = time.time()
        
        try:
            # Analyze current causal model state
            model_summary = self._analyze_causal_model()
            
            # Generate hypotheses based on model gaps
            hypotheses = self._generate_hypotheses(model_summary)
            
            # Plan experiments
            experiment_plans = self._plan_experiments(hypotheses)
            
            thought = {
                "iteration": self.current_iteration,
                "model_summary": model_summary,
                "hypotheses": hypotheses,
                "experiment_plans": experiment_plans,
                "confidence": self._calculate_confidence(),
                "timestamp": time.time()
            }
            
            # Update knowledge base
            self.knowledge_base["last_thought"] = thought
            self.hypothesis_history.extend(hypotheses)
            self.performance_metrics["hypotheses_generated"] += len(hypotheses)
            
            think_time = time.time() - start_time
            self.performance_metrics["total_think_time"] += think_time
            
            self.logger.info(f"Generated {len(hypotheses)} hypotheses")
            return thought
            
        except Exception as e:
            self.logger.error(f"Error in think phase: {e}")
            return {"error": str(e), "iteration": self.current_iteration}
    
    def act(self) -> Dict[str, Any]:
        """Execute experiments based on current state.
        
        Returns:
            Action results including experiment outcomes.
        """
        # Get the last thought from knowledge base
        thought = self.knowledge_base.get("last_thought", {})
        
        return self._act_with_thought(thought)
    
    def _act_with_thought(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Execute experiments based on thought.
        
        Args:
            thought: Thought dictionary from think phase.
            
        Returns:
            Action results including experiment outcomes.
        """
        start_time = time.time()
        
        try:
            if "error" in thought:
                return {"error": "Cannot act on error thought", "thought": thought}
            
            experiment_plans = thought.get("experiment_plans", [])
            results = []
            
            for plan in experiment_plans[:self.config.executor_config.max_concurrent_experiments if self.config.executor_config else 1]:
                try:
                    # Create experiment
                    experiment = self._create_experiment(plan)
                    
                    # Execute experiment
                    if self.config.executor_config is None:
                        # Skip execution if no executor
                        results.append({
                            "plan": plan,
                            "skipped": True,
                            "reason": "No executor configured"
                        })
                        continue
                    
                    if self.scheduler:
                        # Use scheduler for execution
                        experiment_result = self._execute_with_scheduler(experiment)
                    else:
                        # Direct execution
                        experiment_result = self.experiment_executor.execute_experiment(experiment)
                    
                    # Analyze results
                    if self.config.enable_experiment_analysis:
                        analysis = self.experiment_analyzer.analyze_experiment(experiment)
                        experiment_result["analysis"] = analysis
                    
                    results.append({
                        "plan": plan,
                        "experiment": experiment.serialize(),
                        "result": experiment_result,
                        "success": True
                    })
                    
                    self.total_experiments_run += 1
                    self.performance_metrics["successful_experiments"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Experiment failed: {e}")
                    results.append({
                        "plan": plan,
                        "error": str(e),
                        "success": False
                    })
                    self.performance_metrics["failed_experiments"] += 1
            
            action_result = {
                "iteration": self.current_iteration,
                "thought": thought,
                "experiment_results": results,
                "experiments_run": len(results),
                "timestamp": time.time()
            }
            
            # Update experiment history
            self.experiment_history.extend(results)
            
            act_time = time.time() - start_time
            self.performance_metrics["total_act_time"] += act_time
            
            self.logger.info(f"Executed {len(results)} experiments")
            return action_result
            
        except Exception as e:
            self.logger.error(f"Error in act phase: {e}")
            return {"error": str(e), "thought": thought}
    
    def learn(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update causal model and learn from experiment results.
        
        Args:
            action_result: Results from act phase.
            
        Returns:
            Learning results and model updates.
        """
        start_time = time.time()
        
        try:
            if "error" in action_result:
                return {"error": "Cannot learn from error action", "action_result": action_result}
            
            learning_events = []
            model_updates = []
            
            for experiment_result in action_result.get("experiment_results", []):
                if not experiment_result.get("success", False):
                    continue
                
                try:
                    # Extract experiment data
                    experiment_data = self._extract_experiment_data(experiment_result)
                    
                    # Update causal model
                    if self.config.enable_causal_learning and experiment_data:
                        update_result = self._update_causal_model(experiment_data, experiment_result)
                        model_updates.append(update_result)
                        
                        # Check for new causal discoveries
                        if update_result.get("new_discoveries", 0) > 0:
                            self.performance_metrics["causal_discoveries"] += update_result["new_discoveries"]
                    
                    learning_events.append({
                        "experiment_id": experiment_result["experiment"]["experiment_id"],
                        "data_points": len(experiment_data) if experiment_data else 0,
                        "model_updated": len(model_updates) > 0
                    })
                    
                except Exception as e:
                    self.logger.error(f"Learning from experiment failed: {e}")
                    learning_events.append({"error": str(e)})
            
            # Update knowledge base
            self.knowledge_base["last_learning"] = {
                "events": learning_events,
                "model_updates": model_updates,
                "timestamp": time.time()
            }
            
            self.total_learning_events += len(learning_events)
            
            learn_time = time.time() - start_time
            self.performance_metrics["total_learn_time"] += learn_time
            
            learning_result = {
                "iteration": self.current_iteration,
                "learning_events": learning_events,
                "model_updates": model_updates,
                "total_events": len(learning_events),
                "timestamp": time.time()
            }
            
            self.logger.info(f"Processed {len(learning_events)} learning events")
            return learning_result
            
        except Exception as e:
            self.logger.error(f"Error in learn phase: {e}")
            return {"error": str(e), "action_result": action_result}
    
    def run(self, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Run the integrated agent for specified iterations.
        
        Args:
            max_iterations: Maximum number of iterations to run.
            
        Returns:
            Final results and performance summary.
        """
        if max_iterations:
            original_max = self.config.loop_config.max_iterations
            self.config.loop_config.max_iterations = max_iterations
        
        try:
            # Start components
            if self.scheduler:
                self.scheduler.start()
            
            self.experiment_executor.start()
            
            # Run agent loop
            self.logger.info(f"Starting {self.config.agent_name} for {self.config.loop_config.max_iterations} iterations")
            
            results = []
            for i in range(self.config.loop_config.max_iterations):
                self.current_iteration = i + 1
                result = self.agent_loop.run_iteration()
                results.append(result)
                
                # Save state periodically
                if i % self.config.save_state_interval == 0:
                    self._save_state()
            
            # Generate final summary
            summary = self._generate_summary(results)
            
            self.logger.info(f"Completed {len(results)} iterations")
            return summary
            
        finally:
            # Cleanup
            if self.scheduler:
                self.scheduler.stop()
            
            self.experiment_executor.stop()
            
            if max_iterations:
                self.config.loop_config.max_iterations = original_max
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state.
        
        Returns:
            Current state dictionary.
        """
        return {
            "agent_id": self.agent_id,
            "agent_name": self.config.agent_name,
            "current_iteration": self.current_iteration,
            "total_experiments_run": self.total_experiments_run,
            "total_learning_events": self.total_learning_events,
            "causal_model": {
                "variables": len(self.causal_model.variables),
                "edges": len(self.causal_model.edges),
                "summary": self.causal_model.get_summary()
            },
            "performance_metrics": self.performance_metrics,
            "knowledge_base_size": len(self.knowledge_base),
            "hypothesis_history_size": len(self.hypothesis_history),
            "experiment_history_size": len(self.experiment_history)
        }
    
    def _analyze_causal_model(self) -> Dict[str, Any]:
        """Analyze current causal model state.
        
        Returns:
            Model analysis summary.
        """
        summary = self.causal_model.get_summary()
        
        # Identify gaps and opportunities
        gaps = []
        if len(self.causal_model.variables) < self.config.causal_config.num_variables:
            gaps.append("Need more variables")
        
        if len(self.causal_model.edges) < len(self.causal_model.variables) - 1:
            gaps.append("Need more causal connections")
        
        return {
            "summary": summary,
            "gaps": gaps,
            "complexity": len(self.causal_model.edges) / max(len(self.causal_model.variables), 1),
            "learning_progress": self.total_learning_events / max(self.config.loop_config.max_iterations, 1)
        }
    
    def _generate_hypotheses(self, model_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypotheses based on model analysis.
        
        Args:
            model_analysis: Analysis of current causal model.
            
        Returns:
            List of hypothesis dictionaries.
        """
        hypotheses = []
        
        # Generate hypotheses for model gaps
        gaps = model_analysis.get("gaps", [])
        
        if "Need more variables" in gaps:
            # Hypothesis: adding new variables will improve model
            hypotheses.append({
                "type": "variable_addition",
                "description": "Adding new variables will improve causal understanding",
                "confidence": 0.7,
                "variables_needed": min(3, self.config.causal_config.num_variables - len(self.causal_model.variables))
            })
        
        if "Need more causal connections" in gaps:
            # Hypothesis: testing causal relationships between existing variables
            existing_vars = list(self.causal_model.variables.keys())
            if len(existing_vars) >= 2:
                hypotheses.append({
                    "type": "causal_testing",
                    "description": "Testing causal relationships between existing variables",
                    "confidence": 0.6,
                    "variable_pairs": [(existing_vars[i], existing_vars[i+1]) for i in range(len(existing_vars)-1)]
                })
        
        # Generate hypotheses based on current model structure
        if len(self.causal_model.edges) > 0:
            hypotheses.append({
                "type": "model_validation",
                "description": "Validating existing causal relationships",
                "confidence": 0.8,
                "edges_to_test": list(self.causal_model.edges.keys())[:3]
            })
        
        return hypotheses
    
    def _plan_experiments(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan experiments based on hypotheses.
        
        Args:
            hypotheses: List of hypothesis dictionaries.
            
        Returns:
            List of experiment plans.
        """
        plans = []
        
        for hypothesis in hypotheses:
            if hypothesis["type"] == "variable_addition":
                # Plan experiments to discover new variables
                for i in range(min(2, hypothesis.get("variables_needed", 1))):
                    plans.append({
                        "type": "exploratory",
                        "hypothesis": hypothesis,
                        "description": f"Exploratory experiment {i+1}",
                        "variables": ["exploration_var"],
                        "sample_size": 30
                    })
            
            elif hypothesis["type"] == "causal_testing":
                # Plan experiments to test causal relationships
                for var1, var2 in hypothesis.get("variable_pairs", []):
                    plans.append({
                        "type": "causal_test",
                        "hypothesis": hypothesis,
                        "description": f"Test causal relationship {var1} -> {var2}",
                        "variables": [var1, var2],
                        "sample_size": 25
                    })
            
            elif hypothesis["type"] == "model_validation":
                # Plan experiments to validate existing relationships
                for edge_id in hypothesis.get("edges_to_test", []):
                    edge = self.causal_model.edges[edge_id]
                    plans.append({
                        "type": "validation",
                        "hypothesis": hypothesis,
                        "description": f"Validate edge {edge_id}",
                        "variables": [edge["source"], edge["target"]],
                        "sample_size": 40
                    })
        
        return plans
    
    def _create_experiment(self, plan: Dict[str, Any]) -> SimpleExperiment:
        """Create experiment based on plan.
        
        Args:
            plan: Experiment plan dictionary.
            
        Returns:
            Configured SimpleExperiment.
        """
        # Create a fresh experiment config each time
        exp_config = ExperimentConfig(
            name=f"exp_{self.current_iteration}_{len(self.experiment_history)}_{int(time.time() * 1000)}",
            description=plan["description"],
            max_duration=60.0,
            parameters=plan
        )
        
        def experiment_function(experiment):
            # Generate synthetic data based on plan type
            variables = plan.get("variables", [])
            sample_size = plan.get("sample_size", 20)
            
            data = {}
            
            if plan["type"] == "exploratory":
                # Generate random exploratory data
                for var in variables:
                    data[var] = np.random.normal(0, 1, sample_size)
            
            elif plan["type"] == "causal_test":
                # Generate data with potential causal relationship
                if len(variables) >= 2:
                    cause = np.random.normal(0, 1, sample_size)
                    effect = 0.3 * cause + np.random.normal(0, 0.5, sample_size)
                    data[variables[0]] = cause
                    data[variables[1]] = effect
            
            elif plan["type"] == "validation":
                # Generate data to validate existing relationship
                if len(variables) >= 2:
                    # Look up existing edge strength
                    edge_id = None
                    for eid, edge in self.causal_model.edges.items():
                        if edge["source"] in variables and edge["target"] in variables:
                            edge_id = eid
                            break
                    
                    strength = self.causal_model.edges[edge_id]["strength"] if edge_id else 0.3
                    cause = np.random.normal(0, 1, sample_size)
                    effect = strength * cause + np.random.normal(0, 0.5, sample_size)
                    data[variables[0]] = cause
                    data[variables[1]] = effect
            
            return data
        
        # Create a new SimpleExperiment instance each time
        return SimpleExperiment(exp_config, experiment_function)
    
    def _execute_with_scheduler(self, experiment: Experiment) -> Dict[str, Any]:
        """Execute experiment using task scheduler.
        
        Args:
            experiment: Experiment to execute.
            
        Returns:
            Experiment results.
        """
        # Direct execution for now (scheduler integration can be added later)
        return self.experiment_executor.execute_experiment(experiment)
    
    def _extract_experiment_data(self, experiment_result: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract data from experiment result.
        
        Args:
            experiment_result: Result from experiment execution.
            
        Returns:
            Dictionary of variable data arrays.
        """
        if "result" not in experiment_result:
            return {}
        
        result = experiment_result["result"]
        if isinstance(result, dict):
            # Extract numerical arrays
            data = {}
            for key, value in result.items():
                if isinstance(value, (list, np.ndarray)):
                    try:
                        data[key] = np.array(value)
                    except:
                        pass
            return data
        
        return {}
    
    def _update_causal_model(self, data: Dict[str, np.ndarray], experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update causal model with new data.
        
        Args:
            data: Experiment data.
            experiment_result: Full experiment result.
            
        Returns:
            Update result summary.
        """
        # Add new variables if needed
        new_variables = 0
        for var_name in data.keys():
            if var_name not in [v["name"] for v in self.causal_model.variables.values()]:
                self.causal_model.add_variable(var_name, "continuous")
                new_variables += 1
        
        # Update model with data
        old_edge_count = len(self.causal_model.edges)
        self.causal_model.update(data)
        new_edges = len(self.causal_model.edges) - old_edge_count
        
        # Analyze results for causal insights
        new_discoveries = 0
        if "analysis" in experiment_result.get("result", {}):
            analysis = experiment_result["result"]["analysis"]
            significance_rate = analysis.get("significance", {}).get("overall_significance_rate", 0)
            
            if significance_rate > 0.7:
                new_discoveries = 1
        
        return {
            "new_variables": new_variables,
            "new_edges": new_edges,
            "new_discoveries": new_discoveries,
            "total_variables": len(self.causal_model.variables),
            "total_edges": len(self.causal_model.edges)
        }
    
    def _calculate_confidence(self) -> float:
        """Calculate overall confidence in current model.
        
        Returns:
            Confidence score (0-1).
        """
        # Base confidence on model completeness and learning progress
        var_completeness = len(self.causal_model.variables) / max(self.config.causal_config.num_variables, 1)
        learning_progress = self.total_learning_events / max(self.config.loop_config.max_iterations, 1)
        
        return min(1.0, (var_completeness + learning_progress) / 2)
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary of agent run.
        
        Args:
            results: List of iteration results.
            
        Returns:
            Summary dictionary.
        """
        return {
            "agent_info": {
                "agent_id": self.agent_id,
                "agent_name": self.config.agent_name,
                "config": self.config.__dict__
            },
            "run_summary": {
                "total_iterations": len(results),
                "successful_iterations": len([r for r in results if "error" not in r]),
                "total_experiments": self.total_experiments_run,
                "total_learning_events": self.total_learning_events
            },
            "causal_model_summary": self.causal_model.get_summary(),
            "performance_metrics": self.performance_metrics,
            "knowledge_summary": {
                "hypotheses_generated": len(self.hypothesis_history),
                "experiments_completed": len(self.experiment_history),
                "knowledge_base_size": len(self.knowledge_base)
            },
            "final_state": self.get_state(),
            "timestamp": time.time()
        }
    
    def _save_state(self) -> None:
        """Save current agent state."""
        state = self.get_state()
        # In a real implementation, this would save to disk
        self.knowledge_base["saved_state"] = state
        self.logger.debug(f"Saved state at iteration {self.current_iteration}")
    
    def __str__(self) -> str:
        """String representation of integrated agent."""
        return (f"IntegratedAgent({self.config.agent_name}, "
                f"iterations={self.current_iteration}, "
                f"experiments={self.total_experiments_run}, "
                f"variables={len(self.causal_model.variables)})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"IntegratedAgent(id={self.agent_id}, "
                f"name={self.config.agent_name}, "
                f"causal_model_vars={len(self.causal_model.variables)}, "
                f"causal_model_edges={len(self.causal_model.edges)})")
