"""Main entry point for Evo2 Meta-RL Scientist."""

import logging
import sys
import time
from pathlib import Path

from evo2.agent import Agent


import argparse
from evo2.agent.integrated import IntegratedAgent, IntegratedAgentConfig
from evo2.utils.dashboard import Dashboard

def setup_logging(level: str = "INFO", silent: bool = False) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout) if not silent else logging.NullHandler(),
            logging.FileHandler("evo2.log"),
        ],
    )

def main() -> int:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Evo2 Meta-RL Scientist")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--name", type=str, default="Evo2_Scientist", help="Agent name")
    parser.add_argument("--checkpoint", type=str, help="Path to load checkpoint from")
    parser.add_argument("--save-path", type=str, default="checkpoints/agent_v1.pth", help="Path to save final agent")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--visual", action="store_true", help="Enable simulation dashboard")
    args = parser.parse_args()

    setup_logging(args.log_level, silent=args.visual)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Initializing {args.name}")
        
        if args.checkpoint:
            logger.info(f"Loading agent from {args.checkpoint}")
            agent = IntegratedAgent.load_checkpoint(args.checkpoint)
        else:
            config = IntegratedAgentConfig(agent_name=args.name)
            agent = IntegratedAgent(config)
            
        logger.info(f"Starting simulation for {args.iterations} iterations")
        
        dashboard = Dashboard() if args.visual else None
        
        # If visual mode, we run the loop manually to update dashboard
        if args.visual:
            for i in range(args.iterations):
                agent.current_iteration = i + 1
                agent.agent_loop.run_iteration()
                
                # Get current state and augment with max_iterations
                state = agent.get_state()
                state['max_iterations'] = args.iterations
                state['last_thought'] = agent.knowledge_base.get("last_thought", {})
                state['last_learning'] = agent.knowledge_base.get("last_learning", {})
                
                dashboard.render(state)
                time.sleep(0.1)  # Slow down for visual effect
            
            summary = agent._generate_summary([]) # Final summary
        else:
            summary = agent.run(max_iterations=args.iterations)
        
        # Save final state
        agent.save_checkpoint(args.save_path)
        
        logger.info("Simulation completed successfully")
        logger.info(f"Experiments run: {summary['run_summary']['total_experiments']}")
        logger.info(f"Causal discoveries: {summary['performance_metrics']['causal_discoveries']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to start Evo2: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
