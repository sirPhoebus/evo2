import sys
import re
from collections import Counter
from pathlib import Path

def debug_extraction(file_path):
    print(f"Reading {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    print("\n--- Content Snippet ---")
    print(content[:200])
    print("-----------------------\n")
    
    # Updated patterns
    patterns = [
        r'\b[a-z]{4,}(?:_[a-z]+)+\b',  # snake_case (e.g., learning_rate)
        r'\b[A-Z][a-z]{2,}[A-Z][a-zA-Z0-9-]+\b', # CamelCase (e.g., AgentOrchestra)
        r'\b[A-Z]{3,}\b',  # Acronyms (e.g., GAIA, MCP, PPO)
        r'\b[A-Z][a-z]{3,}\b', # Simple Capitalized words (e.g. Agent, Reward) -> NEW
    ]
    
    stop_words = {
        "this", "that", "with", "from", "their", "those", "these", "result", "framework", "system", "method", "paper", "value", "metric",
        "intro", "conclusion", "summary", "chapter", "section", "table", "figure", "case", "study", "work", "time", "year", "data",
        "model", "approach", "based", "using", "used", "such", "each", "both", "most", "many", "some", "other", "into", "over", "than", "then"
    }
    
    word_counts = Counter()
    for p_idx, pattern in enumerate(patterns):
        words = re.findall(pattern, content)
        print(f"Pattern {p_idx} matches: {words[:5]}")
        word_counts.update(words)
        
    topics = [k for k, count in word_counts.items() if k.lower() not in stop_words and count >= 1]
    
    print(f"\nFinal Extracted Topics ({len(topics)}):")
    print(topics)

if __name__ == "__main__":
    debug_extraction("knowledge/summary_RL.txt")
