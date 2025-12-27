"""Dashboard for visualizing simulation progress and metrics."""

import os
import time
from typing import Dict, Any

class Dashboard:
    """A CLI-based dashboard for Evo2 simulation monitoring."""
    
    def __init__(self):
        self.colors = {
            "header": "\033[95m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "bold": "\033[1m",
            "underline": "\033[4m",
            "reset": "\033[0m"
        }
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def render(self, state: Dict[str, Any]):
        """Render the dashboard with current state.
        
        Args:
            state: Dictionary containing agent and simulation state.
        """
        self.clear_screen()
        
        # Header
        print(f"{self.colors['header']}{self.colors['bold']}=== Evo2 Meta-RL Scientist - Simulation Dashboard ==={self.colors['reset']}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Agent: {state.get('agent_name', 'Unknown')}")
        print("-" * 60)
        
        # Iteration Progress
        current = state.get('current_iteration', 0)
        total = state.get('max_iterations', 100)
        progress = (current / total) * 20
        progress_bar = "█" * int(progress) + "░" * (20 - int(progress))
        print(f"Iteration: {current}/{total} [{progress_bar}] {int(current/total*100)}%")
        
        print("\n" + f"{self.colors['cyan']}--- Research Progress ---{self.colors['reset']}")
        print(f"Experiments Run:    {self.colors['green']}{state.get('total_experiments_run', 0)}{self.colors['reset']}")
        print(f"Learning Events:    {self.colors['green']}{state.get('total_learning_events', 0)}{self.colors['reset']}")
        print(f"Causal Discoveries: {self.colors['yellow']}{state.get('performance_metrics', {}).get('causal_discoveries', 0)}{self.colors['reset']}")
        
        print("\n" + f"{self.colors['blue']}--- Causal Model State ---{self.colors['reset']}")
        model = state.get('causal_model', {})
        print(f"Variables:          {model.get('variables', 0)}")
        print(f"Causal Edges:       {model.get('edges', 0)}")
        print(f"Graph Complexity:   {model.get('summary', {}).get('edge_density', 0.0):.2f}")
        
        print("\n" + f"{self.colors['yellow']}--- Last Reasoning (Thought) ---{self.colors['reset']}")
        thought = state.get('last_thought', {})
        if thought:
            hypotheses = thought.get('hypotheses', [])
            if hypotheses:
                print(f"Latest Hypothesis:  {hypotheses[0].get('description', 'N/A')}")
                print(f"Confidence:         {self.colors['bold']}{hypotheses[0].get('confidence', 0.0):.2f}{self.colors['reset']}")
            
            lit = state.get('last_learning', {}).get('events', [])
            if lit:
                 print(f"Recent Insight:     {lit[-1].get('experiment_id', 'N/A')} processed.")
            
            # Show Recently Discovered Variables
            all_vars = model.get('summary', {}).get('variables', {})
            if all_vars:
                # Show last 5 added variables
                recent_vars = list(all_vars.values())[-5:]
                print(f"New Frontiers:      {', '.join(recent_vars)}")
        else:
            print("Reasoning:          Waiting for first thought...")

        print("\n" + "-" * 60)
        print(f"{self.colors['bold']}Status: {self.colors['green']}OPERATIONAL{self.colors['reset']}")
