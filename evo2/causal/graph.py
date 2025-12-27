"""Causal Graph visualization and analysis for Evo2."""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
from .model import CausalModel


@dataclass
class GraphConfig:
    """Configuration for causal graph visualization."""
    layout: str = "spring"  # spring, circular, hierarchical
    node_size: int = 300
    edge_width: float = 2.0
    font_size: int = 12
    show_labels: bool = True
    show_edge_strength: bool = True
    color_scheme: str = "default"  # default, causal, importance
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_layouts = ["spring", "circular", "hierarchical"]
        if self.layout not in valid_layouts:
            raise ValueError(f"layout must be one of {valid_layouts}")
        if self.node_size <= 0:
            raise ValueError("node_size must be positive")
        if self.edge_width <= 0:
            raise ValueError("edge_width must be positive")
        if self.font_size <= 0:
            raise ValueError("font_size must be positive")


class CausalGraph:
    """Causal graph visualization and analysis.
    
    This class provides visualization and analysis capabilities for causal models,
    including graph layout algorithms and metric calculations.
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        """Initialize the causal graph.
        
        Args:
            config: Optional graph configuration.
        """
        self.config = config or GraphConfig()
        
        # Graph structure
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        
        # Layout information
        self.positions: Dict[str, Tuple[float, float]] = {}
        
        self.logger = logging.getLogger("CausalGraph")
    
    @classmethod
    def from_model(cls, model: CausalModel, config: Optional[GraphConfig] = None) -> 'CausalGraph':
        """Create a graph from a causal model.
        
        Args:
            model: Causal model to convert.
            config: Optional graph configuration.
            
        Returns:
            CausalGraph instance.
        """
        graph = cls(config)
        
        # Add nodes
        for var_id, var_info in model.variables.items():
            graph.add_node(var_id, var_info["name"], var_info["type"])
        
        # Add edges
        for edge_id, edge_info in model.edges.items():
            graph.add_edge(
                edge_info["source"],
                edge_info["target"],
                edge_info["strength"]
            )
        
        # Calculate layout
        graph._calculate_layout()
        
        return graph
    
    def add_node(self, node_id: str, label: str, node_type: str = "default", **kwargs) -> None:
        """Add a node to the graph.
        
        Args:
            node_id: Node identifier.
            label: Node label.
            node_type: Node type.
            **kwargs: Additional node properties.
        """
        self.nodes[node_id] = {
            "label": label,
            "type": node_type,
            "color": self._get_node_color(node_type),
            **kwargs
        }
    
    def add_edge(self, source_id: str, target_id: str, strength: float = 1.0, **kwargs) -> None:
        """Add an edge to the graph.
        
        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            strength: Edge strength.
            **kwargs: Additional edge properties.
        """
        edge_id = f"{source_id}->{target_id}"
        self.edges[edge_id] = {
            "source": source_id,
            "target": target_id,
            "strength": strength,
            "width": self._get_edge_width(strength),
            "color": self._get_edge_color(strength),
            **kwargs
        }
    
    def _calculate_layout(self) -> None:
        """Calculate node positions based on layout algorithm."""
        if self.config.layout == "spring":
            self._spring_layout()
        elif self.config.layout == "circular":
            self._circular_layout()
        elif self.config.layout == "hierarchical":
            self._hierarchical_layout()
    
    def _spring_layout(self) -> None:
        """Spring layout algorithm."""
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            return
        
        # Initialize random positions
        node_ids = list(self.nodes.keys())
        positions = np.random.rand(n_nodes, 2) * 10
        
        # Spring layout simulation
        for iteration in range(50):
            forces = np.zeros((n_nodes, 2))
            
            # Repulsive forces between all nodes
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        diff = positions[i] - positions[j]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            forces[i] += diff / (dist ** 3) * 0.1
            
            # Attractive forces for connected nodes
            for edge in self.edges.values():
                source_idx = node_ids.index(edge["source"])
                target_idx = node_ids.index(edge["target"])
                
                diff = positions[target_idx] - positions[source_idx]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    force = diff * edge["strength"] * 0.01
                    forces[source_idx] += force
                    forces[target_idx] -= force
            
            # Update positions
            positions += forces
            
            # Center the graph
            positions -= np.mean(positions, axis=0)
        
        # Store positions
        for i, node_id in enumerate(node_ids):
            self.positions[node_id] = (positions[i, 0], positions[i, 1])
    
    def _circular_layout(self) -> None:
        """Circular layout algorithm."""
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            return
        
        node_ids = list(self.nodes.keys())
        radius = 5.0
        
        for i, node_id in enumerate(node_ids):
            angle = 2 * np.pi * i / n_nodes
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            self.positions[node_id] = (x, y)
    
    def _hierarchical_layout(self) -> None:
        """Hierarchical layout algorithm."""
        # Simple topological layout
        if not self.nodes:
            return
        
        # Find levels using topological sorting
        levels = self._calculate_levels()
        
        # Position nodes by level
        level_width = 10.0
        level_height = 3.0
        
        for level, nodes in enumerate(levels):
            for i, node_id in enumerate(nodes):
                x = (i - len(nodes) / 2) * level_width / max(len(nodes), 1)
                y = -level * level_height
                self.positions[node_id] = (x, y)
    
    def _calculate_levels(self) -> List[List[str]]:
        """Calculate hierarchical levels of nodes."""
        if not self.nodes:
            return []
        
        # Simple level assignment based on topological order
        levels = []
        remaining_nodes = set(self.nodes.keys())
        
        while remaining_nodes:
            current_level = []
            
            # Find nodes with no parents in remaining_nodes
            for node_id in remaining_nodes:
                parents = self._get_parents(node_id)
                if not any(parent in remaining_nodes for parent in parents):
                    current_level.append(node_id)
            
            if not current_level:
                # Cycle detected, add remaining nodes
                current_level = list(remaining_nodes)
            
            levels.append(current_level)
            remaining_nodes -= set(current_level)
        
        return levels
    
    def _get_parents(self, node_id: str) -> List[str]:
        """Get parent nodes of a node."""
        parents = []
        for edge in self.edges.values():
            if edge["target"] == node_id:
                parents.append(edge["source"])
        return parents
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for graph visualization.
        
        Returns:
            Dictionary containing visualization data.
        """
        nodes_data = []
        for node_id, node_info in self.nodes.items():
            pos = self.positions.get(node_id, (0, 0))
            nodes_data.append({
                "id": node_id,
                "label": node_info["label"],
                "type": node_info["type"],
                "x": pos[0],
                "y": pos[1],
                "color": node_info["color"],
                "size": self.config.node_size
            })
        
        edges_data = []
        for edge_id, edge_info in self.edges.items():
            source_pos = self.positions.get(edge_info["source"], (0, 0))
            target_pos = self.positions.get(edge_info["target"], (0, 0))
            
            edges_data.append({
                "id": edge_id,
                "source": edge_info["source"],
                "target": edge_info["target"],
                "strength": edge_info["strength"],
                "width": edge_info["width"],
                "color": edge_info["color"],
                "source_x": source_pos[0],
                "source_y": source_pos[1],
                "target_x": target_pos[0],
                "target_y": target_pos[1]
            })
        
        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "layout": self.config.layout,
            "config": {
                "node_size": self.config.node_size,
                "edge_width": self.config.edge_width,
                "font_size": self.config.font_size,
                "show_labels": self.config.show_labels,
                "show_edge_strength": self.config.show_edge_strength
            }
        }
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate graph metrics.
        
        Returns:
            Dictionary containing graph metrics.
        """
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        
        if n_nodes == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0.0,
                "average_path_length": 0.0,
                "clustering_coefficient": 0.0
            }
        
        # Density
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0
        
        # Average path length (simplified)
        avg_path_length = self._calculate_average_path_length()
        
        # Clustering coefficient (simplified)
        clustering = self._calculate_clustering_coefficient()
        
        return {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "density": density,
            "average_path_length": avg_path_length,
            "clustering_coefficient": clustering
        }
    
    def _calculate_average_path_length(self) -> float:
        """Calculate average shortest path length."""
        if len(self.nodes) <= 1:
            return 0.0
        
        # Build adjacency matrix
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Initialize adjacency matrix
        adj = np.zeros((n, n))
        for edge in self.edges.values():
            source_idx = id_to_idx[edge["source"]]
            target_idx = id_to_idx[edge["target"]]
            adj[source_idx, target_idx] = 1
            adj[target_idx, source_idx] = 1  # Undirected for path calculation
        
        # Floyd-Warshall algorithm
        dist = np.where(adj > 0, 1, np.inf)
        np.fill_diagonal(dist, 0)
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        # Calculate average path length (excluding infinite distances)
        finite_distances = dist[np.isfinite(dist) & (dist > 0)]
        return np.mean(finite_distances) if len(finite_distances) > 0 else 0.0
    
    def _calculate_clustering_coefficient(self) -> float:
        """Calculate clustering coefficient."""
        if len(self.nodes) < 3:
            return 0.0
        
        clustering_coeffs = []
        
        for node_id in self.nodes.keys():
            neighbors = self._get_neighbors(node_id)
            k = len(neighbors)
            
            if k < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count edges between neighbors
            neighbor_edges = 0
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if self._are_connected(neighbor1, neighbor2):
                        neighbor_edges += 1
            
            # Clustering coefficient
            possible_edges = k * (k - 1) / 2
            clustering = neighbor_edges / possible_edges if possible_edges > 0 else 0.0
            clustering_coeffs.append(clustering)
        
        return np.mean(clustering_coeffs)
    
    def _get_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring nodes."""
        neighbors = []
        for edge in self.edges.values():
            if edge["source"] == node_id:
                neighbors.append(edge["target"])
            elif edge["target"] == node_id:
                neighbors.append(edge["source"])
        return neighbors
    
    def _are_connected(self, node1_id: str, node2_id: str) -> bool:
        """Check if two nodes are connected."""
        for edge in self.edges.values():
            if ((edge["source"] == node1_id and edge["target"] == node2_id) or
                (edge["source"] == node2_id and edge["target"] == node1_id)):
                return True
        return False
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        color_map = {
            "continuous": "#1f77b4",
            "binary": "#ff7f0e",
            "categorical": "#2ca02c",
            "default": "#7f7f7f"
        }
        return color_map.get(node_type, color_map["default"])
    
    def _get_edge_width(self, strength: float) -> float:
        """Get edge width based on strength."""
        return max(0.5, min(5.0, strength * self.config.edge_width))
    
    def _get_edge_color(self, strength: float) -> str:
        """Get edge color based on strength."""
        if strength > 0.7:
            return "#d62728"  # Strong positive
        elif strength > 0.3:
            return "#ff7f0e"  # Medium positive
        elif strength > 0:
            return "#ffbb78"  # Weak positive
        else:
            return "#1f77b4"  # Blue for negative/zero
