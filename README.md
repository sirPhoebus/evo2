# Evo2 - Meta-RL Scientist

A sophisticated simulation where AI agents act as **Reinforcement Learning Researchers**. Evo2 simulates the full lifecycle of scientific research, including literature review, hypothesis generation, and experimentation.

## Core Architecture

Agents in Evo2 operate in a continuous loop:
1.  **Think**: Review literature from the `LiteratureStore` and generate/refine hypotheses based on internal Causal Models.
2.  **Act**: Design and execute experiments. Tasks are managed by an RNN-based **Task Execution Engine** and a parallel **Task Scheduler**.
3.  **Learn**: Update causal models and improve execution strategies based on experiment results.

## Key Features

- **Autonomous Literature Review**: Integrated `LiteratureStore` allowing agents to search and synthesize research papers.
- **Causal Modeling**: Internal causal graphs used for inference and discovery.
- **RNN Brain**:Recurrent Neural Network for task processing and decision making.
- **Simulation Dashboard**: Real-time visual monitoring of agent progress and metrics.
- **Operational Standards**:
    - **Deterministic Seeding**: Full reproducibility across runs.
    - **Checkpointing**: Disk-based saving and loading of agent states.

## Installation

```bash
# Recommendation: use a virtual environment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

## Usage

Evo2 provides a robust CLI via `main.py`.

### Start a New Simulation
Run a 10-iteration simulation with visual feedback:
```bash
python -m evo2.main --iterations 10 --visual
```

### Options
- `--iterations N`: Number of cycles to run (default: 10).
- `--visual`: Enable the CLI-based simulation dashboard.
- `--checkpoint PATH`: Load a previously saved agent state.
- `--save-path PATH`: Path to save the final agent state (default: `checkpoints/agent_v1.pth`).
- `--log-level [DEBUG|INFO|WARNING]`: Set verbosity.

## License

MIT License
