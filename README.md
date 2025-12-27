# Evo2 - Meta-RL Scientist

A sophisticated simulation where AI agents act as **Reinforcement Learning Researchers**. Evo2 simulates the full lifecycle of scientific research, including literature review, hypothesis generation, and experimentation.

## Core Architecture

Agents in Evo2 operate in a continuous loop:
1.  **Think**: Review literature from the `LiteratureStore` (/knowledge) and generate hypotheses based on internal Causal Models.
2.  **Act**: Design and execute experiments managed by an RNN-based **Task Execution Engine**.
3.  **Learn**: Update causal models and improve execution strategies based on experiment results.

## Key Features

- **Organic Variable Discovery**: The agent discovers new research variables dynamically by reading literature, allowing for unbounded knowledge growth.
- **Continuous Learning**: Agents persist their state and automatically resume training, "stacking" knowledge over time.
- **Autonomous Literature Review**: Integrated `LiteratureStore` allows agents to search and synthesize research papers from local text files.
- **Causal Modeling**: Sophisticated internal causal graphs used for inference and discovery.
- **Scientific Evaluator**: A comprehensive evaluation suite that assesses the agent's "scientific competence" (reasoning, literature usage, experiment design).
- **Simulation Dashboard**: Real-time visual monitoring of agent progress, new discoveries, and metrics.

## Installation

```bash
# Recommendation: use a virtual environment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

## Usage

### 1. Run the Simulation (Continuous Learning)
By default, the simulation will load the existing agent from `checkpoints/agent_v1.pth` and continue training.

```bash
# Run 100 iterations with visual dashboard
python -m evo2.main --iterations 100 --visual
```

### 2. Start Fresh
To ignore existing checkpoints and start a brand new agent:

```bash
python -m evo2.main --fresh --iterations 100 --visual
```

### 3. Evaluate Scientific Competence
Run the comprehensive evaluation suite to assess the agent's scientific thinking:

```bash
python run_evaluation.py --checkpoint checkpoints/agent_v1.pth
```
Output will be saved to `evaluation_report.md`.

### 4. Visualize Knowledge Graph
View the discovered causal topics and relationships:

```bash
python visualize_graph.py checkpoints/agent_v1.pth
```

## Configuration
- **Knowledge Base**: Place `.txt` files in the `/knowledge` directory. Prefix filenames with `opinion_` or `summary_` for better classification.
- **Checkpointing**: Agents save state automatically to `checkpoints/`.

## License

MIT License
