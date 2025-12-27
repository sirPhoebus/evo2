import argparse
import json
import logging
from pathlib import Path
from evo2.evaluation.evaluator import AgentEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evo2 Agent Evaluator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/agent_v1.pth", help="Path to the agent checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_report.md", help="Path to save the report")
    parser.add_argument("--json", action="store_true", help="Output JSON results to stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("EvaluatorRunner")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    evaluator = AgentEvaluator(str(checkpoint_path))
    results = evaluator.run_full_evaluation()
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    report_md = evaluator.generate_report_markdown(results)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    logger.info(f"Evaluation complete. Report saved to {args.output}")
    print("\n--- Evaluation Summary ---")
    print(f"Overall Score: {results['overall_score']:.2f}/100")
    print(f"Variables Discovered: {results['causal_model']['num_variables']}")
    print(f"Causal Links Established: {results['causal_model']['num_edges']}")
    print(f"Literature Grounding: {results['literature_alignment']['agent_grounding']:.2%}")

if __name__ == "__main__":
    main()
