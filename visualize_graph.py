import pickle
import sys
from pathlib import Path

# The ultimate numpy pickle compatibility hack
import sys
import types

# Create mock numpy core structure for pickle loading
def hack_numpy():
    try:
        import numpy
        # If numpy is present, try to alias _core to core
        if not hasattr(numpy, '_core') and hasattr(numpy, 'core'):
            sys.modules['numpy._core'] = numpy.core
            # Also need multiarray within _core
            if not hasattr(numpy.core, 'multiarray') and hasattr(numpy, 'core'):
                 # This part is tricky, usually numpy.core has multiarray
                 pass
    except ImportError:
        # If no numpy, create a fake one just for the pickle class identifiers
        np = types.ModuleType('numpy')
        sys.modules['numpy'] = np
        np.ndarray = type('ndarray', (), {})
        core = types.ModuleType('numpy.core')
        sys.modules['numpy.core'] = core
        _core = types.ModuleType('numpy._core')
        sys.modules['numpy._core'] = _core

hack_numpy()

def visualize_graph(checkpoint_path):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"Error: {checkpoint_path} not found")
        return

    try:
        with open(checkpoint, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    cm = data.get('causal_model')
    if not cm:
        print("No causal model found in checkpoint")
        return

    print(f"\n=== ORGANIC RESEARCH GRAPH ===")
    print(f"Discovered Concepts: {len(cm.variables)}")
    print(f"Validated Links:     {len(cm.edges)}")
    print("-" * 40)

    # Sort variables by name for clean display
    vars_list = sorted([v['name'] for v in cm.variables.values()])
    print(f"Known concepts: {', '.join(vars_list[:15])}")
    if len(vars_list) > 15:
        print(f"... and {len(vars_list)-15} more.")

    print("\n--- Top Causal Relationships ---")
    edges = []
    for edge in cm.edges.values():
        src = cm.variables[edge['source']]['name']
        tgt = cm.variables[edge['target']]['name']
        strength = edge.get('strength', 0.0)
        edges.append((src, tgt, strength))
    
    edges.sort(key=lambda x: x[2], reverse=True)
    for src, tgt, s in edges[:15]:
        print(f"{src:<20} --> {tgt:<20} | Strength: {s:.4f}")

    print("\n--- Recent Discoveries (Mermaid) ---")
    print("graph LR")
    for src, tgt, s in edges[:10]:
        s_clean = src.replace(" ", "_").replace("-", "_")
        t_clean = tgt.replace(" ", "_").replace("-", "_")
        print(f"    {s_clean} -->|{s:.2f}| {t_clean}")

if __name__ == "__main__":
    path = "checkpoints/agent_v1.pth"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    visualize_graph(path)
