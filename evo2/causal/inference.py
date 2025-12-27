"""Causal Inference implementation for Evo2."""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
from .model import CausalModel


@dataclass
class InferenceConfig:
    """Configuration for causal inference."""
    method: str = "do_calculus"  # do_calculus, backdoor, front_door
    confidence_threshold: float = 0.8
    num_samples: int = 1000
    bootstrap_samples: int = 100
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_methods = ["do_calculus", "backdoor", "front_door"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        if not 0 <= self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")


class CausalInference:
    """Causal inference engine for estimating causal effects.
    
    This class implements various methods for causal inference including
    do-calculus, backdoor criterion, and front-door criterion.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the causal inference engine.
        
        Args:
            config: Optional inference configuration.
        """
        self.config = config or InferenceConfig()
        self.logger = logging.getLogger("CausalInference")
    
    def estimate_effect(
        self,
        model: CausalModel,
        treatment_id: str,
        outcome_id: str,
        adjustment_set: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Estimate the causal effect of treatment on outcome.
        
        Args:
            model: Causal model.
            treatment_id: Treatment variable ID.
            outcome_id: Outcome variable ID.
            adjustment_set: Optional set of variables to adjust for.
            
        Returns:
            Dictionary containing effect estimate and metadata.
        """
        if treatment_id not in model.variables:
            raise ValueError(f"Treatment variable {treatment_id} not found")
        if outcome_id not in model.variables:
            raise ValueError(f"Outcome variable {outcome_id} not found")
        
        # Determine adjustment set if not provided
        if adjustment_set is None:
            adjustment_set = self._find_adjustment_set(model, treatment_id, outcome_id)
        
        # Estimate effect based on method
        if self.config.method == "do_calculus":
            effect = self._estimate_effect_do_calculus(model, treatment_id, outcome_id, adjustment_set)
        elif self.config.method == "backdoor":
            effect = self._estimate_effect_backdoor(model, treatment_id, outcome_id, adjustment_set)
        elif self.config.method == "front_door":
            effect = self._estimate_effect_front_door(model, treatment_id, outcome_id, adjustment_set)
        else:
            raise ValueError(f"Unknown inference method: {self.config.method}")
        
        return {
            "effect_size": effect["size"],
            "confidence": effect["confidence"],
            "method": self.config.method,
            "adjustment_set": adjustment_set,
            "treatment": treatment_id,
            "outcome": outcome_id
        }
    
    def do_calculation(
        self,
        model: CausalModel,
        intervention_id: str,
        intervention_value: float
    ) -> Dict[str, Any]:
        """Perform do-calculus intervention.
        
        Args:
            model: Causal model.
            intervention_id: Variable to intervene on.
            intervention_value: Value to set the variable to.
            
        Returns:
            Dictionary containing intervention results.
        """
        if intervention_id not in model.variables:
            raise ValueError(f"Intervention variable {intervention_id} not found")
        
        # Get children of intervention variable
        children = model.get_children(intervention_id)
        
        # Calculate expected outcomes for each child
        outcomes = {}
        for child_id in children:
            # Simple linear model for expected outcome
            edge_strength = 0.0
            for edge in model.edges.values():
                if edge["source"] == intervention_id and edge["target"] == child_id:
                    edge_strength = edge["strength"]
                    break
            
            # Expected outcome = baseline + edge_strength * intervention
            baseline = model._variable_stats.get(child_id, {}).get("mean", 0.0)
            expected_outcome = baseline + edge_strength * intervention_value
            
            outcomes[child_id] = {
                "expected_value": expected_outcome,
                "edge_strength": edge_strength,
                "baseline": baseline
            }
        
        return {
            "intervention": {intervention_id: intervention_value},
            "expected_outcomes": outcomes,
            "affected_variables": children,
            "method": "do_calculus"
        }
    
    def counterfactual(
        self,
        model: CausalModel,
        treatment_id: str,
        counterfactual_value: float,
        factual_state: Dict[str, float]
    ) -> Dict[str, Any]:
        """Perform counterfactual analysis.
        
        Args:
            model: Causal model.
            treatment_id: Treatment variable.
            counterfactual_value: Counterfactual treatment value.
            factual_state: Factual state of all variables.
            
        Returns:
            Dictionary containing counterfactual predictions.
        """
        if treatment_id not in model.variables:
            raise ValueError(f"Treatment variable {treatment_id} not found")
        
        # Get outcome variables (children of treatment)
        outcomes = model.get_children(treatment_id)
        
        counterfactual_predictions = {}
        
        for outcome_id in outcomes:
            # Find the causal path from treatment to outcome
            path_strength = 0.0
            for edge in model.edges.values():
                if edge["source"] == treatment_id and edge["target"] == outcome_id:
                    path_strength = edge["strength"]
                    break
            
            # Calculate counterfactual outcome
            factual_treatment = factual_state.get(treatment_id, 0.0)
            factual_outcome = factual_state.get(outcome_id, 0.0)
            
            # Simple counterfactual calculation
            treatment_effect = path_strength * (counterfactual_value - factual_treatment)
            counterfactual_outcome = factual_outcome + treatment_effect
            
            counterfactual_predictions[outcome_id] = {
                "predicted_outcome": counterfactual_outcome,
                "factual_outcome": factual_outcome,
                "treatment_effect": treatment_effect,
                "confidence": min(0.9, path_strength * 1.2)  # Simple confidence estimate
            }
        
        return {
            "treatment": treatment_id,
            "counterfactual_value": counterfactual_value,
            "factual_state": factual_state,
            "predictions": counterfactual_predictions,
            "method": "counterfactual"
        }
    
    def _find_adjustment_set(
        self,
        model: CausalModel,
        treatment_id: str,
        outcome_id: str
    ) -> List[str]:
        """Find an appropriate adjustment set using backdoor criterion.
        
        Args:
            model: Causal model.
            treatment_id: Treatment variable ID.
            outcome_id: Outcome variable ID.
            
        Returns:
            List of variable IDs to adjust for.
        """
        # Get all variables except treatment and outcome
        all_vars = set(model.variables.keys())
        candidates = all_vars - {treatment_id, outcome_id}
        
        # Simple heuristic: adjust for common causes
        adjustment_set = []
        
        for var_id in candidates:
            treatment_parents = set(model.get_parents(treatment_id))
            outcome_parents = set(model.get_parents(outcome_id))
            
            # Adjust for common parents
            if var_id in treatment_parents and var_id in outcome_parents:
                adjustment_set.append(var_id)
        
        return adjustment_set
    
    def _estimate_effect_do_calculus(
        self,
        model: CausalModel,
        treatment_id: str,
        outcome_id: str,
        adjustment_set: List[str]
    ) -> Dict[str, Any]:
        """Estimate effect using do-calculus.
        
        Args:
            model: Causal model.
            treatment_id: Treatment variable ID.
            outcome_id: Outcome variable ID.
            adjustment_set: Variables to adjust for.
            
        Returns:
            Effect estimate dictionary.
        """
        # Find direct path strength
        direct_strength = 0.0
        for edge in model.edges.values():
            if edge["source"] == treatment_id and edge["target"] == outcome_id:
                direct_strength = edge["strength"]
                break
        
        # Calculate total effect (including indirect paths)
        total_effect = self._calculate_total_effect(model, treatment_id, outcome_id)
        
        # Confidence based on edge strength and model certainty
        confidence = min(0.95, abs(direct_strength) * 1.5)
        
        return {
            "size": total_effect,
            "confidence": confidence,
            "direct_effect": direct_strength,
            "indirect_effect": total_effect - direct_strength
        }
    
    def _estimate_effect_backdoor(
        self,
        model: CausalModel,
        treatment_id: str,
        outcome_id: str,
        adjustment_set: List[str]
    ) -> Dict[str, Any]:
        """Estimate effect using backdoor criterion.
        
        Args:
            model: Causal model.
            treatment_id: Treatment variable ID.
            outcome_id: Outcome variable ID.
            adjustment_set: Variables to adjust for.
            
        Returns:
            Effect estimate dictionary.
        """
        # Simplified backdoor adjustment
        direct_strength = 0.0
        for edge in model.edges.values():
            if edge["source"] == treatment_id and edge["target"] == outcome_id:
                direct_strength = edge["strength"]
                break
        
        # Adjust for confounders
        adjustment_factor = 1.0
        for confounder_id in adjustment_set:
            for edge in model.edges.values():
                if edge["source"] == confounder_id and edge["target"] == treatment_id:
                    adjustment_factor *= (1 - edge["strength"])
        
        adjusted_effect = direct_strength * adjustment_factor
        
        return {
            "size": adjusted_effect,
            "confidence": min(0.9, abs(adjusted_effect) * 1.3),
            "adjustment_factor": adjustment_factor
        }
    
    def _estimate_effect_front_door(
        self,
        model: CausalModel,
        treatment_id: str,
        outcome_id: str,
        adjustment_set: List[str]
    ) -> Dict[str, Any]:
        """Estimate effect using front-door criterion.
        
        Args:
            model: Causal model.
            treatment_id: Treatment variable ID.
            outcome_id: Outcome variable ID.
            adjustment_set: Variables to adjust for.
            
        Returns:
            Effect estimate dictionary.
        """
        # Find mediators (variables on the path from treatment to outcome)
        mediators = []
        for var_id in model.variables.keys():
            if var_id != treatment_id and var_id != outcome_id:
                # Check if var is on a path from treatment to outcome
                treatment_children = model.get_children(treatment_id)
                outcome_parents = model.get_parents(outcome_id)
                
                if var_id in treatment_children and var_id in outcome_parents:
                    mediators.append(var_id)
        
        # Calculate mediated effect
        total_effect = 0.0
        for mediator_id in mediators:
            # Treatment -> Mediator
            tm_strength = 0.0
            for edge in model.edges.values():
                if edge["source"] == treatment_id and edge["target"] == mediator_id:
                    tm_strength = edge["strength"]
                    break
            
            # Mediator -> Outcome
            mo_strength = 0.0
            for edge in model.edges.values():
                if edge["source"] == mediator_id and edge["target"] == outcome_id:
                    mo_strength = edge["strength"]
                    break
            
            total_effect += tm_strength * mo_strength
        
        return {
            "size": total_effect,
            "confidence": min(0.85, abs(total_effect) * 1.4),
            "mediators": mediators
        }
    
    def _calculate_total_effect(
        self,
        model: CausalModel,
        treatment_id: str,
        outcome_id: str
    ) -> float:
        """Calculate total causal effect including all paths.
        
        Args:
            model: Causal model.
            treatment_id: Treatment variable ID.
            outcome_id: Outcome variable ID.
            
        Returns:
            Total effect strength.
        """
        # Use adjacency matrix to find all paths
        adj_matrix = model.get_adjacency_matrix()
        
        # Get variable indices
        var_ids = list(model.variables.keys())
        id_to_idx = {var_id: idx for idx, var_id in enumerate(var_ids)}
        
        if treatment_id not in id_to_idx or outcome_id not in id_to_idx:
            return 0.0
        
        treatment_idx = id_to_idx[treatment_id]
        outcome_idx = id_to_idx[outcome_id]
        
        # Calculate total effect using matrix powers (simplified)
        total_effect = 0.0
        
        # Direct effect
        total_effect += adj_matrix[outcome_idx, treatment_idx]
        
        # Indirect effects (up to 3 steps)
        current_power = adj_matrix.copy()
        for step in range(2, 4):
            current_power = np.dot(current_power, adj_matrix)
            total_effect += current_power[outcome_idx, treatment_idx] * (0.5 ** (step - 1))
        
        return total_effect
