import random
from typing import Dict, Any

class MockGameWorld:
    """Shared state for Game Mode testing."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MockGameWorld, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance
    
    def reset(self):
        self.state = {
            "player_x": 0.0,
            "player_y": 0.0,
            "enemy_x": 0.5,
            "has_key": False,
            "is_alive": True
        }
    
    def update(self, action: str):
        """Update world state based on action."""
        if action == "A": # Jump
            self.state["player_y"] += 0.5 
            if self.state["player_y"] > 1.0: self.state["player_y"] = 0.0 # Land
        elif action == "RIGHT":
            self.state["player_x"] += 0.1
        elif action == "LEFT":
            self.state["player_x"] -= 0.1
            
        # Physics decay
        self.state["player_y"] = max(0.0, self.state["player_y"] - 0.1)
        
        # Interactions
        if abs(self.state["player_x"] - self.state["enemy_x"]) < 0.1 and self.state["player_y"] < 0.1:
            self.state["is_alive"] = False # Die if touching enemy on ground
