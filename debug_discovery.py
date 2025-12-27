import logging
from evo2.agent.integrated import IntegratedAgent, IntegratedAgentConfig

logging.basicConfig(level=logging.INFO)

config = IntegratedAgentConfig(agent_name="Debug")
agent = IntegratedAgent(config)
# agent.initialize() 

# Inject knowledge
agent.literature_store.knowledge_base = [
    {
        "title": "Game Manual",
        "content": "Content...",
        "type": "manual",
        "topics": ["did_A", "post_player_y"]
    }
]

# Force logic
print("Existing vars:", agent.causal_model.variables)
print("Manual topics:", agent.literature_store.get_all_topics())

# Call discovery
new_vars = agent.literature_store.discover_variables(list(agent.causal_model.variables.keys()))
print("Discovered vars:", new_vars)

# Call generate hypotheses
hypotheses = agent._generate_hypotheses({"num_variables": 0})
print("Generated Hypotheses:", hypotheses)
