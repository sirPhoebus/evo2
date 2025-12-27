from typing import Dict, Any, List, Optional
import time
from ..agent.integrated import IntegratedAgent, IntegratedAgentConfig
from ..experiments.framework import Experiment, SimpleExperiment, ExperimentConfig
from .observer import YOLOObserver
from .interface import GameInterface

class GameAgent(IntegratedAgent):
    """Specialized agent for playing games via scientific discovery."""
    
    def __init__(self, config: IntegratedAgentConfig):
        super().__init__(config)
        self.config.agent_name = "Evo2_Gamer"
        
        # Initialize Game Components
        self.observer = YOLOObserver()
        self.game_interface = GameInterface()
        
    def _create_experiment(self, plan: Dict[str, Any]) -> SimpleExperiment:
        """Override to create Game Actions instead of generic experiments."""
        
        # Name the experiment based on the plan
        exp_name = f"action_{self.current_iteration}_{int(time.time()*1000)}"
        
        exp_config = ExperimentConfig(
            name=exp_name,
            description=plan["description"],
            max_duration=5.0, # Game actions are short
            parameters=plan
        )
        
        def game_action_function(experiment):
            # 1. Observe State (Pre-action)
            pre_state = self.observer.observe()
            
            # 2. Execute Action
            inputs = []
            
            # Translate hypothesis to inputs
            # e.g. "Test Jump" -> ["A"]
            if "Jump" in plan["description"] or "A" in plan.get("variables", []):
                inputs.append("A")
            elif "Right" in plan["description"]:
                inputs.append("RIGHT")
            elif "Left" in plan["description"]:
                inputs.append("LEFT")
            else:
                 # Default exploration
                 import random
                 inputs.append(random.choice(self.game_interface.controls))
            
            action_result = self.game_interface.execute_action({"type": "input", "inputs": inputs})
            
            # 3. Observe State (Post-action)
            time.sleep(0.5) # Wait for physics
            post_state = self.observer.observe()
            
            # 4. Return Delta as Data
            data = {}
            # Combine pre and post for causal analysis
            for k, v in pre_state.items():
                data[f"pre_{k}"] = v
            for k, v in post_state.items():
                data[f"post_{k}"] = v
                
            # Add action indicator
            for ctrl in self.game_interface.controls:
                data[f"did_{ctrl}"] = 1.0 if ctrl in inputs else 0.0
                
            self.logger.warning(f"DEBUG: Game Action returning: {list(data.keys())}")
            return data
            
        return SimpleExperiment(exp_config, game_action_function)
