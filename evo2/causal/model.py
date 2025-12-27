"""Causal Model implementation for Evo2."""

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import uuid


@dataclass
class CausalModelConfig:
    """Configuration for the causal model."""
    num_variables: int = 10
    max_parents: int = 3
    learning_rate: float = 0.01
    edge_strength_threshold: float = 0.1
    update_frequency: int = 10
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_variables <= 0:
            raise ValueError("num_variables must be positive")
        if self.max_parents <= 0:
            raise ValueError("max_parents must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.edge_strength_threshold < 1:
            raise ValueError("edge_strength_threshold must be between 0 and 1")
        if self.update_frequency <= 0:
            raise ValueError("update_frequency must be positive")


class CausalModel:
    """Causal model for representing and learning causal relationships.
    
    This class implements a causal model that can learn causal relationships
    from data and perform causal inference.
    """
    
    def __init__(self, config: Optional[CausalModelConfig] = None):
        """Initialize the causal model.
        
        Args:
            config: Optional model configuration.
        """
        self.config = config or CausalModelConfig()
        
        # Model structure
        self.variables: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        
        # Learning state
        self._update_count = 0
        self._edge_strengths: Dict[str, float] = {}
        self._variable_stats: Dict[str, Dict[str, float]] = {}
        
        self.logger = logging.getLogger("CausalModel")
    
    def add_variable(self, name: str, var_type: str = "continuous", **kwargs) -> str:
        """Add a variable to the model.
        
        Args:
            name: Variable name.
            var_type: Variable type (continuous, binary, categorical).
            **kwargs: Additional variable properties.
            
        Returns:
            Variable ID.
        """
        if len(self.variables) >= self.config.num_variables:
            raise ValueError(f"Maximum number of variables ({self.config.num_variables}) reached")
        
        var_id = str(uuid.uuid4())
        
        self.variables[var_id] = {
            "name": name,
            "type": var_type,
            "created_at": self._get_timestamp(),
            **kwargs
        }
        
        # Initialize variable statistics
        self._variable_stats[var_id] = {
            "mean": 0.0,
            "std": 1.0,
            "min": float('inf'),
            "max": float('-inf'),
            "count": 0
        }
        
        self.logger.info(f"Added variable {name} (ID: {var_id})")
        return var_id
    
    def add_edge(self, source_id: str, target_id: str, strength: float = 0.5, **kwargs) -> str:
        """Add a causal edge to the model.
        
        Args:
            source_id: Source variable ID.
            target_id: Target variable ID.
            strength: Edge strength (0-1).
            **kwargs: Additional edge properties.
            
        Returns:
            Edge ID.
        """
        if source_id not in self.variables:
            raise ValueError(f"Source variable {source_id} not found")
        if target_id not in self.variables:
            raise ValueError(f"Target variable {target_id} not found")
        
        # Check for cycles
        if self._would_create_cycle(source_id, target_id):
            raise ValueError("Adding this edge would create a cycle")
        
        # Check parent limit
        current_parents = self.get_parents(target_id)
        if len(current_parents) >= self.config.max_parents:
            raise ValueError(f"Target variable {target_id} has maximum number of parents")
        
        edge_id = str(uuid.uuid4())
        
        self.edges[edge_id] = {
            "source": source_id,
            "target": target_id,
            "strength": max(0.0, min(1.0, strength)),
            "created_at": self._get_timestamp(),
            **kwargs
        }
        
        self._edge_strengths[edge_id] = strength
        
        self.logger.info(f"Added edge {source_id} -> {target_id} (strength: {strength})")
        return edge_id
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the model.
        
        Args:
            edge_id: Edge ID to remove.
            
        Returns:
            True if edge was removed, False if not found.
        """
        if edge_id in self.edges:
            edge = self.edges[edge_id]
            del self.edges[edge_id]
            del self._edge_strengths[edge_id]
            
            self.logger.info(f"Removed edge {edge['source']} -> {edge['target']}")
            return True
        
        return False
    
    def get_parents(self, var_id: str) -> List[str]:
        """Get parent variables of a variable.
        
        Args:
            var_id: Variable ID.
            
        Returns:
            List of parent variable IDs.
        """
        parents = []
        for edge_id, edge in self.edges.items():
            if edge["target"] == var_id:
                parents.append(edge["source"])
        return parents
    
    def get_children(self, var_id: str) -> List[str]:
        """Get child variables of a variable.
        
        Args:
            var_id: Variable ID.
            
        Returns:
            List of child variable IDs.
        """
        children = []
        for edge_id, edge in self.edges.items():
            if edge["source"] == var_id:
                children.append(edge["target"])
        return children
    
    def update(self, data: Dict[str, np.ndarray]) -> None:
        """Update the model with new data.
        
        Args:
            data: Dictionary mapping variable IDs to data arrays.
        """
        self._update_count += 1
        
        # Update variable statistics
        self._update_variable_stats(data)
        
        # Update edge strengths based on data
        if self._update_count % self.config.update_frequency == 0:
            self._update_edge_strengths(data)
        
        self.logger.debug(f"Model updated with {len(data)} variables")
    
    def _update_variable_stats(self, data: Dict[str, np.ndarray]) -> None:
        """Update variable statistics.
        
        Args:
            data: Data dictionary.
        """
        for var_id, values in data.items():
            if var_id not in self.variables:
                continue
            
            stats = self._variable_stats[var_id]
            
            # Update running statistics
            new_count = stats["count"] + len(values)
            if stats["count"] == 0:
                stats["mean"] = np.mean(values)
                stats["std"] = np.std(values)
            else:
                # Online update of mean and std
                old_mean = stats["mean"]
                stats["mean"] = (stats["count"] * old_mean + np.sum(values)) / new_count
                
                # Update std (simplified)
                old_var = stats["std"] ** 2
                new_var = ((stats["count"] - 1) * old_var + np.sum((values - old_mean) ** 2)) / (new_count - 1)
                stats["std"] = np.sqrt(max(new_var, 0))
            
            stats["count"] = new_count
            stats["min"] = min(stats["min"], np.min(values))
            stats["max"] = max(stats["max"], np.max(values))
    
    def _update_edge_strengths(self, data: Dict[str, np.ndarray]) -> None:
        """Update edge strengths based on data correlations.
        
        Args:
            data: Data dictionary.
        """
        for edge_id, edge in self.edges.items():
            source_id = edge["source"]
            target_id = edge["target"]
            
            if source_id not in data or target_id not in data:
                continue
            
            source_values = data[source_id]
            target_values = data[target_id]
            
            if len(source_values) != len(target_values):
                continue
            
            # Calculate correlation as a simple measure of causal strength
            correlation = np.corrcoef(source_values, target_values)[0, 1]
            
            if not np.isnan(correlation):
                # Update edge strength with learning rate
                current_strength = self._edge_strengths[edge_id]
                new_strength = current_strength + self.config.learning_rate * (abs(correlation) - current_strength)
                new_strength = max(0.0, min(1.0, new_strength))
                
                self._edge_strengths[edge_id] = new_strength
                edge["strength"] = new_strength
                
                # Remove weak edges
                if new_strength < self.config.edge_strength_threshold:
                    self.remove_edge(edge_id)
    
    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Check if adding an edge would create a cycle.
        
        Args:
            source_id: Source variable ID.
            target_id: Target variable ID.
            
        Returns:
            True if cycle would be created.
        """
        # Simple cycle detection using DFS
        visited = set()
        
        def dfs(current_id: str) -> bool:
            if current_id == source_id:
                return True
            if current_id in visited:
                return False
            
            visited.add(current_id)
            for child_id in self.get_children(current_id):
                if dfs(child_id):
                    return True
            return False
        
        return dfs(target_id)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the model to a dictionary.
        
        Returns:
            Serialized model representation.
        """
        return {
            "variables": self.variables,
            "edges": self.edges,
            "edge_strengths": self._edge_strengths,
            "variable_stats": self._variable_stats,
            "config": asdict(self.config),
            "update_count": self._update_count
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'CausalModel':
        """Deserialize a model from a dictionary.
        
        Args:
            data: Serialized model data.
            
        Returns:
            Deserialized CausalModel instance.
        """
        config = CausalModelConfig(**data["config"])
        model = cls(config)
        
        model.variables = data["variables"]
        model.edges = data["edges"]
        model._edge_strengths = data["edge_strengths"]
        model._variable_stats = data["variable_stats"]
        model._update_count = data["update_count"]
        
        return model
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get the adjacency matrix of the causal graph.
        
        Returns:
            Adjacency matrix as numpy array.
        """
        n_vars = len(self.variables)
        if n_vars == 0:
            return np.zeros((0, 0))
        
        # Create variable ID to index mapping
        var_ids = list(self.variables.keys())
        id_to_idx = {var_id: idx for idx, var_id in enumerate(var_ids)}
        
        # Build adjacency matrix
        adj_matrix = np.zeros((n_vars, n_vars))
        
        for edge in self.edges.values():
            source_idx = id_to_idx[edge["source"]]
            target_idx = id_to_idx[edge["target"]]
            adj_matrix[target_idx, source_idx] = edge["strength"]
        
        return adj_matrix
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the causal model.
        
        Returns:
            Summary dictionary.
        """
        return {
            "num_variables": len(self.variables),
            "num_edges": len(self.edges),
            "variables": {var_id: var["name"] for var_id, var in self.variables.items()},
            "edges": list(self.edges.keys()),
            "learning_rate": self.config.learning_rate,
            "update_count": self._update_count
        }
    
    def get_variable_statistics(self, variable_id: str) -> Optional[Dict[str, float]]:
        """Get statistics for a variable.
        
        Args:
            variable_id: ID of the variable.
            
        Returns:
            Statistics dictionary or None if not found.
        """
        if variable_id not in self.variables:
            return None
        
        stats = self._variable_stats.get(variable_id, {})
        return {
            "mean": stats.get("mean", 0.0),
            "std": stats.get("std", 0.0),
            "min": stats.get("min", 0.0),
            "max": stats.get("max", 0.0),
            "count": stats.get("count", 0)
        }
    
    def _get_timestamp(self) -> float:
        """Get current timestamp.
        
        Returns:
            Current timestamp as float.
        """
        import time
        return time.time()
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"CausalModel(variables={len(self.variables)}, edges={len(self.edges)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CausalModel(num_variables={len(self.variables)}, "
                f"num_edges={len(self.edges)}, "
                f"update_count={self._update_count})")
