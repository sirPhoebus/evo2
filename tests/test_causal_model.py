"""Tests for Causal Model implementation."""

import pytest
import numpy as np
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from evo2.causal.model import CausalModel, CausalModelConfig
from evo2.causal.inference import CausalInference, InferenceConfig
from evo2.causal.graph import CausalGraph, GraphConfig


class TestCausalModel:
    """Test suite for CausalModel."""

    def test_model_initialization(self):
        """Test CausalModel initialization."""
        config = CausalModelConfig(
            num_variables=5,
            max_parents=3,
            learning_rate=0.01
        )
        
        model = CausalModel(config)
        
        assert model.config.num_variables == 5
        assert model.config.max_parents == 3
        assert model.config.learning_rate == 0.01
        assert len(model.variables) == 0
        assert len(model.edges) == 0

    def test_model_configuration_validation(self):
        """Test model configuration validation."""
        # Valid config
        config = CausalModelConfig(num_variables=10, max_parents=2)
        assert config.num_variables == 10
        
        # Invalid configs
        with pytest.raises(ValueError):
            CausalModelConfig(num_variables=0)
        
        with pytest.raises(ValueError):
            CausalModelConfig(max_parents=0)
        
        with pytest.raises(ValueError):
            CausalModelConfig(learning_rate=-0.1)

    def test_add_variable(self):
        """Test adding variables to the model."""
        config = CausalModelConfig(num_variables=5)
        model = CausalModel(config)
        
        # Add variable
        var_id = model.add_variable("test_var", "continuous")
        
        assert var_id is not None
        assert len(model.variables) == 1
        assert model.variables[var_id]["name"] == "test_var"
        assert model.variables[var_id]["type"] == "continuous"

    def test_add_edge(self):
        """Test adding edges to the model."""
        config = CausalModelConfig(num_variables=5)
        model = CausalModel(config)
        
        # Add variables
        var1_id = model.add_variable("var1", "continuous")
        var2_id = model.add_variable("var2", "continuous")
        
        # Add edge
        edge_id = model.add_edge(var1_id, var2_id, strength=0.8)
        
        assert edge_id is not None
        assert len(model.edges) == 1
        assert model.edges[edge_id]["source"] == var1_id
        assert model.edges[edge_id]["target"] == var2_id
        assert model.edges[edge_id]["strength"] == 0.8

    def test_remove_edge(self):
        """Test removing edges from the model."""
        config = CausalModelConfig(num_variables=5)
        model = CausalModel(config)
        
        # Add variables and edge
        var1_id = model.add_variable("var1", "continuous")
        var2_id = model.add_variable("var2", "continuous")
        edge_id = model.add_edge(var1_id, var2_id)
        
        assert len(model.edges) == 1
        
        # Remove edge
        success = model.remove_edge(edge_id)
        
        assert success is True
        assert len(model.edges) == 0

    def test_get_parents(self):
        """Test getting parent variables."""
        config = CausalModelConfig(num_variables=5)
        model = CausalModel(config)
        
        # Add variables
        var1_id = model.add_variable("var1", "continuous")
        var2_id = model.add_variable("var2", "continuous")
        var3_id = model.add_variable("var3", "continuous")
        
        # Add edges: var1 -> var3, var2 -> var3
        model.add_edge(var1_id, var3_id)
        model.add_edge(var2_id, var3_id)
        
        # Get parents of var3
        parents = model.get_parents(var3_id)
        
        assert len(parents) == 2
        assert var1_id in parents
        assert var2_id in parents

    def test_get_children(self):
        """Test getting child variables."""
        config = CausalModelConfig(num_variables=5)
        model = CausalModel(config)
        
        # Add variables
        var1_id = model.add_variable("var1", "continuous")
        var2_id = model.add_variable("var2", "continuous")
        var3_id = model.add_variable("var3", "continuous")
        
        # Add edges: var1 -> var2, var1 -> var3
        model.add_edge(var1_id, var2_id)
        model.add_edge(var1_id, var3_id)
        
        # Get children of var1
        children = model.get_children(var1_id)
        
        assert len(children) == 2
        assert var2_id in children
        assert var3_id in children

    def test_model_update(self):
        """Test updating the model with new data."""
        config = CausalModelConfig(num_variables=5, learning_rate=0.1, update_frequency=1)
        model = CausalModel(config)
        
        # Add variables
        var1_id = model.add_variable("var1", "continuous")
        var2_id = model.add_variable("var2", "continuous")
        
        # Add edge
        edge_id = model.add_edge(var1_id, var2_id, strength=0.5)
        
        initial_strength = model.edges[edge_id]["strength"]
        
        # Update with data
        data = {
            var1_id: np.array([1.0, 2.0, 3.0]),
            var2_id: np.array([1.5, 2.5, 3.5])
        }
        
        model.update(data)
        
        # Edge strength should have changed
        new_strength = model.edges[edge_id]["strength"]
        assert new_strength != initial_strength

    def test_model_serialization(self):
        """Test model serialization and deserialization."""
        config = CausalModelConfig(num_variables=5)
        model = CausalModel(config)
        
        # Add variables and edges
        var1_id = model.add_variable("var1", "continuous")
        var2_id = model.add_variable("var2", "continuous")
        model.add_edge(var1_id, var2_id, strength=0.7)
        
        # Serialize
        serialized = model.serialize()
        
        assert isinstance(serialized, dict)
        assert "variables" in serialized
        assert "edges" in serialized
        assert "config" in serialized
        
        # Deserialize
        new_model = CausalModel.deserialize(serialized)
        
        assert len(new_model.variables) == 2
        assert len(new_model.edges) == 1


class TestCausalInference:
    """Test suite for CausalInference."""

    def test_inference_initialization(self):
        """Test CausalInference initialization."""
        config = InferenceConfig(
            method="do_calculus",
            confidence_threshold=0.8
        )
        
        inference = CausalInference(config)
        
        assert inference.config.method == "do_calculus"
        assert inference.config.confidence_threshold == 0.8

    def test_causal_effect_estimation(self):
        """Test causal effect estimation."""
        config = InferenceConfig()
        inference = CausalInference(config)
        
        # Create simple model
        model_config = CausalModelConfig(num_variables=3)
        model = CausalModel(model_config)
        
        # Add variables and edges: X -> Y, Z -> Y
        x_id = model.add_variable("X", "continuous")
        y_id = model.add_variable("Y", "continuous")
        z_id = model.add_variable("Z", "continuous")
        
        model.add_edge(x_id, y_id, strength=0.5)
        model.add_edge(z_id, y_id, strength=0.3)
        
        # Estimate effect of X on Y
        effect = inference.estimate_effect(model, x_id, y_id)
        
        assert isinstance(effect, dict)
        assert "effect_size" in effect
        assert "confidence" in effect
        assert "method" in effect

    def test_do_calculation(self):
        """Test do-calculus operations."""
        config = InferenceConfig()
        inference = CausalInference(config)
        
        # Create model
        model_config = CausalModelConfig(num_variables=2)
        model = CausalModel(model_config)
        
        x_id = model.add_variable("X", "continuous")
        y_id = model.add_variable("Y", "continuous")
        model.add_edge(x_id, y_id, strength=0.6)
        
        # Calculate do(X=1)
        result = inference.do_calculation(model, x_id, 1.0)
        
        assert isinstance(result, dict)
        assert "intervention" in result
        assert "expected_outcomes" in result

    def test_counterfactual_analysis(self):
        """Test counterfactual analysis."""
        config = InferenceConfig()
        inference = CausalInference(config)
        
        # Create model
        model_config = CausalModelConfig(num_variables=3)
        model = CausalModel(model_config)
        
        # Add variables: Treatment -> Outcome, Confounder -> Both
        t_id = model.add_variable("Treatment", "binary")
        o_id = model.add_variable("Outcome", "binary")
        c_id = model.add_variable("Confounder", "continuous")
        
        model.add_edge(t_id, o_id, strength=0.4)
        model.add_edge(c_id, t_id, strength=0.3)
        model.add_edge(c_id, o_id, strength=0.2)
        
        # Counterfactual: what if treatment was different?
        factual = {t_id: 1, o_id: 1, c_id: 0.5}
        counterfactual = inference.counterfactual(model, t_id, 0, factual)
        
        assert isinstance(counterfactual, dict)
        assert "predictions" in counterfactual


class TestCausalGraph:
    """Test suite for CausalGraph."""

    def test_graph_initialization(self):
        """Test CausalGraph initialization."""
        config = GraphConfig(
            layout="spring",
            node_size=300,
            edge_width=2.0
        )
        
        graph = CausalGraph(config)
        
        assert graph.config.layout == "spring"
        assert graph.config.node_size == 300
        assert graph.config.edge_width == 2.0

    def test_graph_from_model(self):
        """Test creating graph from causal model."""
        # Create model
        model_config = CausalModelConfig(num_variables=3)
        model = CausalModel(model_config)
        
        # Add variables and edges
        var1_id = model.add_variable("var1", "continuous")
        var2_id = model.add_variable("var2", "continuous")
        var3_id = model.add_variable("var3", "continuous")
        
        model.add_edge(var1_id, var2_id, strength=0.7)
        model.add_edge(var2_id, var3_id, strength=0.5)
        
        # Create graph
        graph_config = GraphConfig()
        graph = CausalGraph.from_model(model, graph_config)
        
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_graph_visualization(self):
        """Test graph visualization."""
        # Create simple model
        model_config = CausalModelConfig(num_variables=2)
        model = CausalModel(model_config)
        
        var1_id = model.add_variable("var1", "continuous")
        var2_id = model.add_variable("var2", "continuous")
        model.add_edge(var1_id, var2_id, strength=0.8)
        
        # Create graph
        graph_config = GraphConfig()
        graph = CausalGraph.from_model(model, graph_config)
        
        # Generate visualization data
        viz_data = graph.get_visualization_data()
        
        assert isinstance(viz_data, dict)
        assert "nodes" in viz_data
        assert "edges" in viz_data
        assert "layout" in viz_data

    def test_graph_metrics(self):
        """Test graph metrics calculation."""
        # Create model with more complex structure
        model_config = CausalModelConfig(num_variables=4)
        model = CausalModel(model_config)
        
        # Add variables
        var_ids = []
        for i in range(4):
            var_ids.append(model.add_variable(f"var{i}", "continuous"))
        
        # Add edges to create a diamond structure
        model.add_edge(var_ids[0], var_ids[1], strength=0.5)
        model.add_edge(var_ids[0], var_ids[2], strength=0.3)
        model.add_edge(var_ids[1], var_ids[3], strength=0.6)
        model.add_edge(var_ids[2], var_ids[3], strength=0.4)
        
        # Create graph
        graph_config = GraphConfig()
        graph = CausalGraph.from_model(model, graph_config)
        
        # Calculate metrics
        metrics = graph.calculate_metrics()
        
        assert "num_nodes" in metrics
        assert "num_edges" in metrics
        assert "density" in metrics
        assert "average_path_length" in metrics
