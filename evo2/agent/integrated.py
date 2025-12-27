"""Integrated Agent system combining all Phase 3 components."""

import time
import uuid
import logging
import torch
from pathlib import Path
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
from ..utils.literature import LiteratureStore


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
    seed: int = 42
    
    # Integration settings
    enable_causal_learning: bool = True
    enable_experiment_analysis: bool = True
    enable_task_scheduling: bool = True
    enable_literature_review: bool = True
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
        self.logger = logging.getLogger(f"IntegratedAgent.{self.config.agent_name}")
        
        # Setup deterministic seeding
        self._setup_seeding()
        
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
        
        # Initialize literature store
        self.literature_store = LiteratureStore() if self.config.enable_literature_review else None
        
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
        
        self.agent_loop = AgentLoop(
            agent=self,
            config=self.config.loop_config,
            scheduler=self.scheduler
        )

        # Scientific variable pool (empty for purely organic discovery from literature)
        self.potential_variables = []
    
    def _setup_seeding(self) -> None:
        """Initialize deterministic seeding for reproducibility."""
        seed = self.config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import random
        random.seed(seed)
        # Ensure deterministic algorithms in torch if needed
        # torch.use_deterministic_algorithms(True) 
        self.logger.info(f"Initialized deterministic seeding with seed: {seed}")
    
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
            
            # Review literature if enabled
            literature_insights = []
            if self.literature_store:
                literature_insights = self._review_literature(hypotheses)
                # Refine hypotheses based on literature
                hypotheses = self._refine_hypotheses_with_literature(hypotheses, literature_insights)
            
            # Plan experiments
            experiment_plans = self._plan_experiments(hypotheses)
            
            thought = {
                "iteration": self.current_iteration,
                "model_summary": model_summary,
                "hypotheses": hypotheses,
                "literature_insights": literature_insights,
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
                    self.logger.info(f"Dispatching experiment: {experiment.config.name} - {experiment.config.description}")
                    self.logger.debug(f"Parameters: {experiment.config.parameters}")
                    if self.config.executor_config is None:
                        # Skip execution if no executor
                        results.append({
                            "plan": plan,
                            "skipped": True,
                            "reason": "No executor configured"
                        })
                        continue
                    
                    if self.agent_loop.execution_engine:
                        # Use RNN-based execution engine
                        # We need to wrap the experiment in a Task
                        from ..tasks.base import Task
                        
                        class ExperimentTask(Task):
                            def __init__(self, exp, executor):
                                # Task ID can be the experiment name
                                super().__init__(task_id=f"task_{exp.config.name}")
                                self.exp = exp
                                self.executor = executor
                            def execute(self):
                                return self.executor.execute_experiment(self.exp)
                        
                        task = ExperimentTask(experiment, self.experiment_executor)
                        experiment_result = self.agent_loop.execution_engine.execute_task(task)
                    elif self.scheduler:
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
            
            # Extract action results (handle both direct and AgentLoop formats)
            results_to_process = []
            if "action" in action_result:
                results_to_process = action_result["action"].get("experiment_results", [])
            else:
                results_to_process = action_result.get("experiment_results", [])
                
            for experiment_result in results_to_process:
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
                    
                    # Log experiment outcome
                    exp_name = experiment_result["experiment"]["config"]["name"]
                    self.logger.info(f"Experiment {exp_name} completed. Data points: {len(experiment_data) if experiment_data else 0}")
                    if experiment_data:
                        for var_name, values in experiment_data.items():
                            self.logger.debug(f"  Variable '{var_name}': mean={np.mean(values):.4f}, std={np.std(values):.4f}")
                    
                    if self.config.enable_causal_learning and experiment_data:
                        res = model_updates[-1]
                        self.logger.info(f"Causal Update: New Vars: {res['new_variables']}, New Edges: {res['new_edges']}, Discoveries: {res['new_discoveries']}")
                    
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
        
        # Organic growth: if we have literature, see if there are new concepts to explore
        if self.literature_store:
            existing_names = [v["name"] for v in self.causal_model.variables.values()]
            new_concepts = self.literature_store.discover_variables(existing_names, limit=1)
            if new_concepts:
                gaps.append("Need more variables")
        
        # Fallback to config limit if no literature or specifically requested
        if "Need more variables" not in gaps and len(self.causal_model.variables) < self.config.causal_config.num_variables:
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
            # Organic Discovery: Pick potential variables from literature store if available
            existing_names = [v["name"] for v in self.causal_model.variables.values()]
            
            available = []
            if self.literature_store:
                available = self.literature_store.discover_variables(existing_names, limit=3)
                
            # Fallback to predefined pool if literature discovery yields nothing
            if not available:
                available = [v for v in self.potential_variables if v not in existing_names]
            
            if available:
                target_vars = available[:2]
                hypotheses.append({
                    "type": "variable_addition",
                    "description": f"Identify impact of {' and '.join(target_vars)} on agent performance",
                    "confidence": 0.75,
                    "target_variables": target_vars,
                    "variables_needed": len(target_vars)
                })
        
        if "Need more causal connections" in gaps or len(self.causal_model.variables) >= 2:
            # Hypothesis: testing causal relationships between existing variables
            existing_vars = list(self.causal_model.variables.keys())
            if len(existing_vars) >= 2:
                # Find pairs without edges
                pairs = []
                import random
                for _ in range(2):
                    v1 = random.choice(existing_vars)
                    v2 = random.choice([v for v in existing_vars if v != v1])
                    pairs.append((v1, v2))
                
                hypotheses.append({
                    "type": "causal_testing",
                    "description": "Testing causal relationships between existing variables",
                    "confidence": 0.6,
                    "variable_pairs": pairs
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
                vars_to_add = hypothesis.get("target_variables", ["exploration_var"])
                for i, var_name in enumerate(vars_to_add):
                    plans.append({
                        "type": "exploratory",
                        "hypothesis": hypothesis,
                        "description": f"Exploratory experiment for {var_name}",
                        "variables": [var_name],
                        "sample_size": 30
                    })
            
            elif hypothesis["type"] == "causal_testing":
                # Plan experiments to test causal relationships
                for var_id1, var_id2 in hypothesis.get("variable_pairs", []):
                    # Get variable names
                    var1 = self.causal_model.variables[var_id1]["name"]
                    var2 = self.causal_model.variables[var_id2]["name"]
                    plans.append({
                        "type": "causal_test",
                        "hypothesis": hypothesis,
                        "description": f"Test causal relationship {var1} -> {var2}",
                        "variables": [var1, var2],
                        "sample_size": 30
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
            
            self.logger.debug(f"Generating synthetic data for {plan['type']} experiment: {variables}")
            data = {}
            
            if plan["type"] == "exploratory":
                # Generate random exploratory data
                for var in variables:
                    data[var] = np.random.normal(0, 1, sample_size)
            
            elif plan["type"] == "causal_test" or plan["type"] == "validation":
                # Generate data with potential causal relationship based on GROUND TRUTH
                if len(variables) >= 2:
                    cause_var = variables[0]
                    effect_var = variables[1]
                    
                    # GROUND TRUTH ORACLE (Scientific Rigor)
                    # Defining the actual laws of the simulation universe
                    truth_rules = [
                        ("Exploration", "discovery of better policies"),
                        ("High discount factor", "preference for long-term rewards"),
                        ("Large positive rewards", "faster learning of actions"),
                        ("Negative rewards", "avoidance of actions"),
                        # Variations for robustness
                        ("Exploration", "better policies"),
                        ("High discount", "long-term rewards"),
                        ("Positive rewards", "faster learning"),
                        ("Negative rewards", "avoidance")
                    ]
                    
                    true_strength = 0.05 # Default: No relationship (Noise)
                    
                    # Check if this pair matches a known truth
                    for t_cause, t_effect in truth_rules:
                        # Fuzzy match: check if key phrases appear in the variables
                        # We normalize by lowercasing for checking
                        c_lower = cause_var.lower()
                        e_lower = effect_var.lower()
                        tc_lower = t_cause.lower()
                        te_lower = t_effect.lower()
                        
                        if (tc_lower in c_lower and te_lower in e_lower):
                            true_strength = 0.95 # Strong relationship
                            break
                    
                    # Generate data based on True Strength
                    sample_size = plan.get("sample_size", 30)
                    cause = np.random.normal(0, 1, sample_size)
                    effect = true_strength * cause + np.random.normal(0, 0.2, sample_size)
                    
                    data[cause_var] = cause
                    data[effect_var] = effect
            
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
        # print(f"DEBUG: Result keys: {experiment_result.keys()}")
        self.logger.warning(f"DEBUG: Extracting data from keys: {list(experiment_result.keys())}")
        if "data" in experiment_result:
            result = experiment_result["data"]
            self.logger.warning(f"DEBUG: Found 'data' with type {type(result)} and length {len(result) if hasattr(result, '__len__') else 'N/A'}")
        elif "result" in experiment_result:
             result = experiment_result["result"]
             self.logger.warning(f"DEBUG: Found 'result' with type {type(result)}")
        else:
            self.logger.warning("DEBUG: No 'data' or 'result' found!")
            return {}
        
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
        
        # Map variable names to IDs for causal model update
        id_data = {}
        name_to_id = {v["name"]: k for k, v in self.causal_model.variables.items()}
        for var_name, values in data.items():
            if var_name in name_to_id:
                id_data[name_to_id[var_name]] = values
        
        # Update model with data using IDs
        old_edge_count = len(self.causal_model.edges)
        if id_data:
            self.causal_model.update(id_data)
            
            # Simple Structure Learning: Discover new edges from strong correlations
            # This is crucial for Game Mode where we don't start with a graph
            if self.config.enable_causal_learning:
                import itertools
                var_ids = list(id_data.keys())
                print(f"DEBUG: Checking correlations for {len(var_ids)} variables: {[self.causal_model.variables[v]['name'] for v in var_ids]}")
                
                # Check pairs
                for i in range(len(var_ids)):
                    for j in range(len(var_ids)):
                        if i == j: continue
                        
                        src_id = var_ids[i]
                        tgt_id = var_ids[j]
                        
                        # Apply heuristics for directionality if available
                        src_name = self.causal_model.variables[src_id]["name"]
                        tgt_name = self.causal_model.variables[tgt_id]["name"]
                        
                        # Game Mode Heuristic: actions cause state changes
                        if "post_" in src_name and "did_" in tgt_name:
                            continue # Ignore effect->cause
                        if "did_" in src_name and "did_" in tgt_name:
                            continue # Ignore action->action
                            
                        # Check if edge already exists
                        exists = False
                        edge_id = f"{src_id}->{tgt_id}"
                        current_edge = None
                        
                        for eid, edge in self.causal_model.edges.items():
                            if edge["source"] == src_id and edge["target"] == tgt_id:
                                exists = True
                                current_edge = edge
                                edge_id = eid
                                break
                        
                        # Calculate correlation
                        src_vals = id_data[src_id]
                        tgt_vals = id_data[tgt_id]
                        
                        if len(src_vals) > 10: # Increased from 5 to 10
                            corr = np.corrcoef(src_vals, tgt_vals)[0, 1]
                            
                            if np.isnan(corr):
                                continue
                                
                            abs_corr = abs(corr)
                            
                            # 1. Negative Evidence (Pruning)
                            if exists:
                                # If correlation is weak, weaken the edge
                                if abs_corr < 0.2:
                                    # Decay strength
                                    new_strength = current_edge.get("strength", 0.5) * 0.8
                                    self.causal_model.edges[edge_id]["strength"] = new_strength
                                    
                                    rationale = f"Weakened due to low correlation {abs_corr:.2f}"
                                    print(f"DEBUG: {rationale} for {src_name}->{tgt_name}")
                                    
                                    # Prune if too weak
                                    if new_strength < 0.3:
                                        del self.causal_model.edges[edge_id]
                                        rationale = f"PRUNED due to strength {new_strength:.2f} < 0.3"
                                        print(f"DEBUG: {rationale} for {src_name}->{tgt_name}")
                                        # Use a special key for negative learning
                                        update_result.setdefault("pruned_edges", []).append(f"{src_name}->{tgt_name}")
                                
                                # Reinforce if strong validation
                                elif abs_corr > 0.8:
                                    new_strength = min(1.0, current_edge.get("strength", 0.5) * 1.1)
                                    self.causal_model.edges[edge_id]["strength"] = new_strength
                                     
                            # 2. Discovery (New Edges)
                            elif abs_corr > 0.85: # Increased from 0.4 to 0.85
                                try:
                                    self.causal_model.add_edge(src_id, tgt_id, strength=abs_corr)
                                    new_discoveries += 1
                                    rationale = f"Discovered (Corr: {abs_corr:.2f} > 0.85)"
                                    print(f"DEBUG: {rationale} for {src_name} -> {tgt_name}")
                                    update_result.setdefault("discoveries", []).append({
                                        "edge": f"{src_name}->{tgt_name}",
                                        "rationale": rationale
                                    })
                                except ValueError:
                                    pass
                                    pass # Cycle or max parents reached
        
        # Analyze results for causal insights
        new_discoveries = 0
        plan = experiment_result.get("plan", {})
        
        # Explicit edge discovery for causal tests
        if plan.get("type") == "causal_test" and len(plan.get("variables", [])) >= 2:
            v1_name, v2_name = plan["variables"][0], plan["variables"][1]
            if v1_name in data and v2_name in data:
                corr = np.corrcoef(data[v1_name], data[v2_name])[0, 1]
                if abs(corr) > 0.4:  # Threshold for discovery
                    v1_id = name_to_id.get(v1_name)
                    v2_id = name_to_id.get(v2_name)
                    if v1_id and v2_id:
                        try:
                            self.causal_model.add_edge(v1_id, v2_id, strength=abs(corr))
                            new_discoveries += 1
                        except Exception as e:
                            self.logger.debug(f"Could not add edge: {e}")

        new_edges = len(self.causal_model.edges) - old_edge_count
        
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
        """Save current agent state to in-memory knowledge base."""
        state = self.get_state()
        self.knowledge_base["saved_state"] = state
        self.logger.debug(f"Saved in-memory state at iteration {self.current_iteration}")
        
    def save_checkpoint(self, path: str) -> None:
        """Save a persistent checkpoint to disk.
        
        Args:
            path: Path to save the checkpoint file.
        """
        import pickle
        checkpoint_data = {
            "config": self.config,
            "agent_id": self.agent_id,
            "current_iteration": self.current_iteration,
            "total_experiments_run": self.total_experiments_run,
            "total_learning_events": self.total_learning_events,
            "causal_model": self.causal_model,
            "knowledge_base": self.knowledge_base,
            "hypothesis_history": self.hypothesis_history,
            "experiment_history": self.experiment_history,
            "performance_metrics": self.performance_metrics,
            "model_state_dict": self.agent_loop.execution_engine.model.state_dict() if self.agent_loop.execution_engine else None
        }
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        self.logger.info(f"Saved persistent checkpoint to: {save_path}")

    @classmethod
    def load_checkpoint(cls, path: str) -> 'IntegratedAgent':
        """Load an agent from a persistent checkpoint.
        
        Args:
            path: Path to the checkpoint file.
            
        Returns:
            The loaded IntegratedAgent instance.
        """
        import pickle
        
        class NumpyCompatibilityUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'numpy._core.multiarray':
                    module = 'numpy.core.multiarray'
                elif module == 'numpy._core':
                    module = 'numpy.core'
                return super().find_class(module, name)
            
        with open(path, 'rb') as f:
            data = NumpyCompatibilityUnpickler(f).load()
            
        agent = cls(data["config"])
        agent.agent_id = data["agent_id"]
        agent.current_iteration = data["current_iteration"]
        agent.total_experiments_run = data["total_experiments_run"]
        agent.total_learning_events = data["total_learning_events"]
        agent.causal_model = data["causal_model"]
        agent.knowledge_base = data["knowledge_base"]
        agent.hypothesis_history = data["hypothesis_history"]
        agent.experiment_history = data["experiment_history"]
        agent.performance_metrics = data["performance_metrics"]
        
        if data["model_state_dict"] and agent.agent_loop.execution_engine:
            agent.agent_loop.execution_engine.model.load_state_dict(data["model_state_dict"])
            
        return agent

    def _review_literature(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Query literature for insights related to current hypotheses.
        
        Args:
            hypotheses: Current list of hypotheses.
            
        Returns:
            List of literature insights.
        """
        if not self.literature_store:
            return []
            
        all_topics = []
        for h in hypotheses:
            if h["type"] == "causal_testing":
                all_topics.extend(["Causality", "Statistics"])
            elif h["type"] == "variable_addition":
                all_topics.append("RL")
            
        if not all_topics:
            all_topics = ["RL"]
            
        papers = self.literature_store.query(all_topics)
        insights = []
        for paper in papers:
            insights.append({
                "title": paper["title"],
                "summary": paper["summary"],
                "relevant_topics": paper["topics"]
            })
            
        return insights

    def _refine_hypotheses_with_literature(
        self, 
        hypotheses: List[Dict[str, Any]], 
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Refine hypotheses based on literature insights.
        
        Args:
            hypotheses: Current hypotheses.
            insights: Insights from literature review.
            
        Returns:
            Refined hypotheses.
        """
        if not insights:
            return hypotheses
            
        refined = []
        for h in hypotheses:
            new_h = h.copy()
            # Conceptual refinement: boost confidence if literature supports the topic
            for insight in insights:
                relevant = False
                if h["type"] == "causal_testing" and "Causality" in insight["relevant_topics"]:
                    relevant = True
                elif h["type"] == "variable_addition" and "RL" in insight["relevant_topics"]:
                    relevant = True
                
                if relevant:
                    new_h["confidence"] = min(1.0, new_h.get("confidence", 0.5) + 0.1)
                    new_h["description"] += f" (Literature support: {insight['title']})"
            
            refined.append(new_h)
            
        return refined
    
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
