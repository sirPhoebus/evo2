import sys
from pathlib import Path
from evo2.agent.integrated import IntegratedAgent, IntegratedAgentConfig

def visualize():
    config = IntegratedAgentConfig()
    agent = IntegratedAgent(config)
    
    checkpoint_path = Path("checkpoints/agent_v1.pth")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return
        
    try:
        agent = IntegratedAgent.load_checkpoint(str(checkpoint_path))
    except Exception as e:
        print(f"Loading failed: {e}")
        return
        
    cm = agent.causal_model
    print(f"\n=== ORGANIC RESEARCH GRAPH ===")
    print(f"Variables: {len(cm.variables)}")
    print(f"Edges:     {len(cm.edges)}")
    print("-" * 40)
    
    # List all variables discovered from literature
    var_names = sorted([v['name'] for v in cm.variables.values()])
    print("Discovered Concepts:")
    for i in range(0, len(var_names), 4):
        print(f"  {', '.join(var_names[i:i+4])}")
        
    print("\nTop Relationships:")
    edges = []
    for edge in cm.edges.values():
        src = cm.variables[edge['source']]['name']
        tgt = cm.variables[edge['target']]['name']
        strength = edge.get('strength', 0.0)
        edges.append((src, tgt, strength))
        
    edges.sort(key=lambda x: x[2], reverse=True)
    for src, tgt, s in edges[:15]:
        print(f"  {src:<20} -> {tgt:<20} (str: {s:.2f})")

if __name__ == "__main__":
    visualize()
    
# Mermaid helper
def print_mermaid(agent):
    edges = []
    cm = agent.causal_model
    for edge in cm.edges.values():
         src = cm.variables[edge['source']]['name'].replace(' ', '_')
         tgt = cm.variables[edge['target']]['name'].replace(' ', '_')
         edges.append(f"{src} --> {tgt}")
    print("\ngraph TD")
    for e in edges[:20]:
        print(f"  {e}")
