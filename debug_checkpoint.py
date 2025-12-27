from evo2.agent.integrated import IntegratedAgent
import sys

try:
    print("Loading checkpoint...")
    agent = IntegratedAgent.load_checkpoint("checkpoints/agent_v1.pth")
    cm = agent.causal_model
    print(f"Variables: {len(cm.variables)}")
    print(f"Edges: {len(cm.edges)}")
    
    print("\nEDGES FOUND:")
    for edge in cm.edges.values():
        src = cm.variables[edge['source']]['name']
        tgt = cm.variables[edge['target']]['name']
        strength = edge.get('strength', 0.0)
        print(f"{src} -> {tgt} ({strength:.2f})")
        
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
