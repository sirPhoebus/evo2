import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the numpy fix globally for this script
# import sys
# import numpy
# if not hasattr(numpy, '_core') and hasattr(numpy, 'core'):
#    sys.modules['numpy._core'] = numpy.core

from ..agent.integrated import IntegratedAgent, IntegratedAgentConfig
from ..tasks.base import Task, TaskPriority

class AgentEvaluator:
    """Evaluates an Evo2 Meta-RL Scientist agent's performance and knowledge."""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.logger = logging.getLogger("AgentEvaluator")
        self.agent = None
        if checkpoint_path:
            self.load_agent(checkpoint_path)
            
    def load_agent(self, checkpoint_path: str):
        """Load an agent from a checkpoint."""
        self.logger.info(f"Loading agent from {checkpoint_path}")
        self.agent = IntegratedAgent.load_checkpoint(checkpoint_path)
        return self.agent

    def evaluate_causal_model(self) -> Dict[str, Any]:
        """Evaluate the quality and complexity of the discovered causal model."""
        if not self.agent:
            return {"error": "No agent loaded"}
            
        cm = self.agent.causal_model
        n_vars = len(cm.variables)
        n_edges = len(cm.edges)
        
        # Calculate Density
        max_edges = n_vars * (n_vars - 1) if n_vars > 1 else 1
        density = n_edges / max_edges
        
        # Calculate Connectivity
        connectivity = 0.0
        if n_vars > 0:
            reachable_counts = []
            for start_node in cm.variables.keys():
                visited = {start_node}
                stack = [start_node]
                while stack:
                    curr = stack.pop()
                    for edge in cm.edges.values():
                        if edge['source'] == curr and edge['target'] not in visited:
                            visited.add(edge['target'])
                            stack.append(edge['target'])
                reachable_counts.append(len(visited))
            connectivity = np.mean(reachable_counts) / n_vars if n_vars > 0 else 0
            
        strengths = [e.get('strength', 0.5) for e in cm.edges.values()]
        avg_strength = np.mean(strengths) if strengths else 0.0
        
        return {
            "num_variables": n_vars,
            "num_edges": n_edges,
            "density": float(density),
            "connectivity_index": float(connectivity),
            "average_edge_strength": float(avg_strength),
            "discovery_efficiency": self.agent.total_learning_events / max(self.agent.total_experiments_run, 1)
        }

    def evaluate_decision_engine(self, num_test_tasks: int = 20) -> Dict[str, Any]:
        """Evaluate the RNN decision engine performance on novel tasks."""
        if not self.agent or not self.agent.agent_loop.execution_engine:
            return {"error": "No execution engine found"}
            
        engine = self.agent.agent_loop.execution_engine
        
        from ..tasks.base import Task, TaskPriority
        
        class MockExperimentTask(Task):
            def __init__(self, name, priority=TaskPriority.MEDIUM):
                super().__init__(task_id=f"task_eval_{name}")
                self.priority = priority
            def execute(self):
                return {"status": "success"}

        test_tasks = []
        for i in range(num_test_tasks):
            p = np.random.choice([TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.CRITICAL])
            task = MockExperimentTask(f"{i}", priority=p)
            test_tasks.append(task)
            
        history_start = len(engine.execution_history)
        engine.execute_batch(test_tasks)
        recent_history = engine.execution_history[history_start:]
        
        confidences = [h['decision']['confidence'] for h in recent_history if 'decision' in h]
        action_types = [h['decision']['action_type'] for h in recent_history if 'decision' in h]
        
        from collections import Counter
        action_dist = Counter(action_types)
        
        return {
            "average_decision_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "action_distribution": dict(action_dist)
        }

    def evaluate_literature_alignment(self) -> Dict[str, Any]:
        """Check alignment with literature."""
        if not self.agent or not self.agent.literature_store:
            return {"error": "No literature store found", "overlap_count": 0, "literature_coverage": 0.0, "agent_grounding": 0.0}
            
        discovered_vars = {v['name'].lower() for v in self.agent.causal_model.variables.values()}
        try:
            lit_topics = {t.lower() for t in self.agent.literature_store.get_all_topics()}
        except:
            lit_topics = set()
        
        overlap = discovered_vars.intersection(lit_topics)
        coverage = len(overlap) / len(lit_topics) if lit_topics else 0.0
        grounding = len(overlap) / len(discovered_vars) if discovered_vars else 0.0
        
        return {
            "overlap_count": len(overlap),
            "literature_coverage": float(coverage),
            "agent_grounding": float(grounding)
        }

    def evaluate_scientific_thinking(self) -> Dict[str, Any]:
        """Qualitative analysis of the agent's scientific process."""
        if not self.agent:
            return {}
            
        history = self.agent.hypothesis_history
        if not history:
            return {"diversity": 0.0, "lit_integration": 0.0}
            
        # 1. Hypothesis Diversity
        types = [h.get("type", "unknown") for h in history]
        unique_types = set(types)
        diversity = len(unique_types) / len(history) if history else 0.0
        
        # 2. Literature Usage (based on description tag)
        lit_supported = sum(1 for h in history if "Literature support" in h.get("description", ""))
        lit_integration = lit_supported / len(history) if history else 0.0
        
        return {
            "hypothesis_diversity": float(diversity),
            "literature_integration_rate": float(lit_integration),
            "total_hypotheses": len(history),
            "unique_hypothesis_types": list(unique_types)
        }

    def evaluate_experiment_evolution(self) -> Dict[str, Any]:
        """Analyze if the agent gets better at designing experiments."""
        if not self.agent:
            return {}
            
        history = self.agent.experiment_history
        if not history:
            return {"success_trend": "insufficient_data"}
            
        # Split history into chunks (e.g., first half vs second half)
        mid = len(history) // 2
        first_half = history[:mid]
        second_half = history[mid:]
        
        def success_rate(experiments):
            if not experiments: return 0.0
            return sum(1 for e in experiments if e.get("success", False)) / len(experiments)
            
        rate_early = success_rate(first_half)
        rate_late = success_rate(second_half)
        
        return {
            "early_success_rate": float(rate_early),
            "late_success_rate": float(rate_late),
            "improvement": float(rate_late - rate_early),
            "total_experiments": len(history)
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        results = {
            "causal_model": self.evaluate_causal_model(),
            "decision_engine": self.evaluate_decision_engine(),
            "literature_alignment": self.evaluate_literature_alignment(),
            "scientific_thinking": self.evaluate_scientific_thinking(),
            "experiment_evolution": self.evaluate_experiment_evolution(),
            "overall_score": 0.0
        }
        
        cm = results["causal_model"]
        de = results["decision_engine"]
        la = results["literature_alignment"]
        st = results["scientific_thinking"]
        ee = results["experiment_evolution"]
        
        if "error" not in cm:
            # Enhanced Score Formula
            score = (
                cm["density"] * 15 + 
                cm["connectivity_index"] * 15 + 
                de["average_decision_confidence"] * 20 + 
                la["agent_grounding"] * 15 +
                st.get("diversity", 0.0) * 15 +  # Reward diverse thinking
                max(0, ee.get("improvement", 0.0)) * 20 # Reward improvement
            ) * 100
            
            # Cap at 100 or normalize roughly
            results["overall_score"] = float(min(100.0, score))
            
        return results

    def generate_report_markdown(self, results: Dict[str, Any]) -> str:
        cm = results["causal_model"]
        de = results["decision_engine"]
        la = results["literature_alignment"]
        st = results["scientific_thinking"]
        ee = results["experiment_evolution"]
        
        report = [
            f"# Evo2 Scientific Competence Report",
            f"**Overall Scientific Score: {results['overall_score']:.2f}/100**",
            f"\n## 1. Does the agent learn to think like a scientist?",
            f"**Analysis of Hypothesis Generation:**",
            f"- **Hypothesis Diversity**: {st.get('hypothesis_diversity', 0):.4f} (Unique Types: {st.get('unique_hypothesis_types')})",
            f"- **Decision Confidence**: {de.get('average_decision_confidence', 0):.4f}",
            f"- **Confidence Profile**: {de.get('action_distribution')}",
            
            f"\n## 2. Does it use literature effectively?",
            f"**Literature Usage Analysis:**",
            f"- **Concepts Grounded in Literature**: {la.get('agent_grounding', 0):.2%}",
            f"- **Literature-Informed Hypotheses**: {st.get('literature_integration_rate', 0):.2%} of hypotheses explicitly cite literature support.",
            f"- **Coverage of Available Knowledge**: {la.get('literature_coverage', 0):.2%}",

            f"\n## 3. Does it design better experiments over time?",
            f"**Experiment Evolution Track:**",
            f"- **Early Success Rate**: {ee.get('early_success_rate', 0):.2%}",
            f"- **Late Success Rate**: {ee.get('late_success_rate', 0):.2%}",
            f"- **Improvement Trend**: {'POSITIVE (+{:.2%})'.format(ee.get('improvement')) if ee.get('improvement') > 0 else 'NEUTRAL/NEGATIVE ({:.2%})'.format(ee.get('improvement'))}",
            f"- **Total Experiments Run**: {ee.get('total_experiments')}",

            f"\n## 4. Does its causal model become more accurate/complex?",
            f"**Knowledge Graph Quality:**",
            f"- **Complexity**: {cm.get('num_variables')} variables, {cm.get('num_edges')} edges",
            f"- **Graph Density**: {cm.get('density', 0):.4f}",
            f"- **Connectivity Index**: {cm.get('connectivity_index', 0):.4f}",
            f"- **Discovery Efficiency**: {cm.get('discovery_efficiency', 0):.2f} discoveries per event"
        ]
        
        return "\n".join(report)

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/eval_copy.pth")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    evaluator = AgentEvaluator(args.checkpoint)
    print(json.dumps(evaluator.run_full_evaluation(), indent=2))
