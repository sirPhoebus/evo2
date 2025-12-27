import logging
import time
from evo2.agent.integrated import IntegratedAgentConfig
from evo2.game.agent import GameAgent

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Setup Config
    config = IntegratedAgentConfig(
        agent_name="Test_Gamer",
        # Disable lit review for this simple test or assume manual is loaded
    )
    # Reduce iterations for test
    config.loop_config.max_iterations = 60
    config.enable_task_scheduling = False # Force sync for matching logic
    
    # 2. Init Agent
    agent = GameAgent(config)
    # agent.initialize() # IntegratedAgent initializes in __init__
    
    # 3. Inject Manual Knowledge (Mocking loading from file)
    # In a real run, this would come from /knowledge
    agent.literature_store.knowledge_base = [
        {
            "title": "Game Manual",
            "content": "Pressing did_A increases post_player_y. Increasing post_player_x allows moving right. did_RIGHT increases post_player_x. did_A avoids spikes.",
            "type": "manual",
            "topics": ["did_A", "post_player_y", "post_player_x", "did_RIGHT", "did_LEFT", "did_B", "post_enemy_x", "post_has_key", "post_is_alive"]
        }
    ]
    # Manually trigger regex discovery on this text (mocking what happens on load)
    # For now, we'll just seed the agent with some "Game Concepts" directly if the regex doesn't catch them
    # But let's see if the organic discovery picks up 'Jump', 'Spike', 'Player'
    
    # 4. Run Game Loop
    print("Starting Game Agent...")
    results = agent.run()
    
    # 5. Report
    print("\n=== Game Session Report ===")
    print(f"Actions Taken: {len(agent.experiment_history)}")
    
    print("\nDiscovered Game Mechanics (Causal Graph):")
    for uuid, edge in agent.causal_model.edges.items():
        src = agent.causal_model.variables[edge['source']]['name']
        tgt = agent.causal_model.variables[edge['target']]['name']
        print(f"  {src} -> {tgt} (strength: {edge.get('strength', '?'):.2f})")

if __name__ == "__main__":
    main()
