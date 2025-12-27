import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from dataclasses import dataclass
import random
import time

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    class_id: int
    confidence: float
    label: str

class YOLOObserver:
    """Simulates a YOLO-based visual observer for game state extraction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("YOLOObserver")
        self.known_classes = ["player", "enemy", "pickup", "obstacle", "goal"]
        
        # State tracking
        self.last_observation = {}
        self.observation_history = []
        
    def observe(self, frame: np.ndarray = None) -> Dict[str, float]:
        """Process a frame (or simulate processing) to extract state variables."""
        # Use Mock World state
        from .mock_world import MockGameWorld
        world = MockGameWorld()
        
        # Add some noise to simulate vision noise
        noise = np.random.normal(0, 0.01)
        
        variables = {
            "player_x": world.state["player_x"] + noise,
            "player_y": world.state["player_y"] + noise,
            "has_enemy": 1.0, # Always present in mock
            "dist_to_enemy": abs(world.state["player_x"] - world.state["enemy_x"])
        }
        
        self.last_observation = variables
        self.observation_history.append(variables)
        
        return variables
    
    def _get_detections(self, frame: Any) -> List[BoundingBox]:
        """Mock detection logic for prototype."""
        # TODO: Integrate real YOLOv8 model here
        boxes = []
        
        # Always detect player
        boxes.append(BoundingBox(
            x=random.random(), y=random.random(),
            width=0.1, height=0.2,
            class_id=0, confidence=0.9, label="player"
        ))
        
        # Randomly detect explicit objects
        if random.random() > 0.3:
            boxes.append(BoundingBox(
                x=random.random(), y=random.random(),
                width=0.1, height=0.1,
                class_id=1, confidence=0.8, label="enemy"
            ))
            
        return boxes
        
    def _detections_to_variables(self, detections: List[BoundingBox]) -> Dict[str, float]:
        """Map raw bounding boxes to semantic variables for the causal model."""
        vars = {}
        
        player = next((d for d in detections if d.label == "player"), None)
        
        if player:
            vars["player_x"] = player.x
            vars["player_y"] = player.y
            
            # Calculate relations
            for d in detections:
                if d.label != "player":
                    # Distance to entity
                    dist = np.sqrt((d.x - player.x)**2 + (d.y - player.y)**2)
                    vars[f"dist_to_{d.label}"] = dist
        
        # Existence flags
        for cls in self.known_classes:
            found = any(d.label == cls for d in detections)
            vars[f"has_{cls}"] = 1.0 if found else 0.0
            
        return vars
