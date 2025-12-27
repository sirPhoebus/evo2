# Evo2 - Meta-RL Scientist

A sophisticated simulation where AI agents act as Reinforcement Learning Researchers.

## Overview

Evo2 simulates the full lifecycle of scientific research, including literature review, hypothesis generation, and experimentation. AI agents autonomously read literature, maintain causal models, and learn to execute parallel experiments through a recurrent neural network "Brain."

## Features

- **Autonomous Literature Review**: Agents can read, synthesize, and utilize existing research
- **Causal Modeling**: Agents maintain and update internal causal models
- **Parallel Experimentation**: RNN-based task execution allowing for concurrent processing
- **Rich Data Visualization**: Comprehensive dashboards for monitoring agent progress
- **Modular Architecture**: Extensible plugin system for new environments and architectures

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from evo2 import Agent

# Create an agent
agent = Agent(config="config.yaml")

# Run the agent loop
agent.run()
```

## Development

This project follows a strict Test-Driven Development workflow. See `workflow.md` for detailed development guidelines.

## License

MIT License
