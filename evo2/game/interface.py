from typing import Dict, Any, List, Optional
import logging
import time

class GameInterface:
    """Interface for executed agent actions in a game environment."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("GameInterface")
        
        # Valid controls
        self.controls = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
        
    def execute_action(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plan (series of inputs) and return the result.
        
        Args:
            action_plan: Dictionary defining the action, e.g.,
                        {"type": "sequence", "inputs": ["RIGHT", "A"]}
        
        Returns:
            Result dictionary.
        """
        action_type = action_plan.get("type", "input")
        
        if action_type == "input":
            inputs = action_plan.get("inputs", [])
            return self._send_inputs(inputs)
            
        elif action_type == "wait":
            duration = action_plan.get("duration", 0.5)
            time.sleep(duration)
            return {"status": "waited", "duration": duration}
            
        else:
            self.logger.warning(f"Unknown action type: {action_type}")
            return {"status": "failed", "reason": "unknown_type"}
            
    def _send_inputs(self, inputs: List[str]) -> Dict[str, Any]:
        """Simulate sending input to controller."""
        valid_inputs = [i for i in inputs if i in self.controls]
        
        if not valid_inputs:
             return {"status": "no_op", "reason": "no_valid_inputs"}
             
        self.logger.info(f"Controller Input: {valid_inputs}")
        
        # In a real scenario, this would interface with a game (e.g., via keyboard emulation or API)
        # game.press(valid_inputs)
        
        # Update Mock World
        from .mock_world import MockGameWorld
        world = MockGameWorld()
        for input_key in valid_inputs:
             world.update(input_key)
        
        return {"status": "executed", "inputs": valid_inputs}
