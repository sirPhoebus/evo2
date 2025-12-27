"""Main entry point for Evo2 Meta-RL Scientist."""

import logging
import sys
from pathlib import Path

from evo2.agent import Agent


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("evo2.log"),
        ],
    )


def main() -> int:
    """Main entry point for the application.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Evo2 Meta-RL Scientist")
        
        # TODO: Initialize and run agent based on configuration
        logger.info("Evo2 started successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to start Evo2: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
