"""LiteratureStore for managing and searching research papers from local knowledge folder."""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

class LiteratureStore:
    """A store for research literature that reads from local text files.
    
    Allows agents to 'read' papers based on topics or keywords, 
    distinguishing between summaries and opinions.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Default to the knowledge folder in the repo
            self.base_dir = Path(__file__).parent.parent.parent / "knowledge"
            
        self.corpus: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("LiteratureStore")
        
        if self.base_dir.exists():
            self._load_files()
        else:
            self.logger.warning(f"Knowledge directory not found at {self.base_dir}")
        
    def _load_files(self):
        """Scan and parse files in the base directory."""
        self.logger.info(f"Loading literature from {self.base_dir}")
        
        # Scan for .txt files
        for file_path in self.base_dir.glob("*.txt"):
            try:
                paper_data = self._parse_file(file_path)
                if paper_data:
                    self.corpus.append(paper_data)
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path}: {e}")
                
        self.logger.info(f"Loaded {len(self.corpus)} literature items")
        
    def _parse_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a single knowledge file.
        
        Args:
            file_path: Path to the .txt file.
            
        Returns:
            Dictionary containing paper metadata and content.
        """
        filename = file_path.name
        
        # Determine type
        if filename.startswith("opinion_"):
            doc_type = "opinion"
        elif filename.startswith("summary_"):
            doc_type = "summary"
        else:
            doc_type = "unknown"
            
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        metadata = {
            "file_path": str(file_path),
            "filename": filename,
            "type": doc_type,
            "topics": [] # Will populate from content if possible
        }
        
        # Extract metadata from header
        content_start_line = 0
        for i, line in enumerate(lines):
            if line.startswith("Timestamp:"):
                metadata["timestamp"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source PDF:"):
                metadata["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source Summary:"):
                metadata["source_summary"] = line.split(":", 1)[1].strip()
            elif "==============" in line:
                content_start_line = i + 1
                break
                
        # The title can be derived from the source or filename
        title = metadata.get("source") or metadata.get("source_summary") or filename
        if title.endswith(".pdf"):
            title = title[:-4]
        if doc_type == "opinion":
            title = f"{title} (Opinion)"
        metadata["title"] = title
        
        # Combine the rest as summary/content
        raw_content = "".join(lines[content_start_line:]).strip()
        metadata["summary"] = raw_content
        
        # Enhanced heuristic for variable/topic extraction
        # Look for words like 'learning_rate', 'AgentOrchestra', 'MCP-Manager', etc.
        import re
        
        # Potential variable patterns: snake_case, CamelCase with special chars, or specific RL terms
        patterns = [
            r'\b[a-z]{4,}(?:_[a-z]+)+\b',  # snake_case (e.g., learning_rate)
            r'\b[A-Z][a-z]{2,}[A-Z][a-zA-Z0-9-]+\b', # CamelCase (e.g., AgentOrchestra, MultiAgent)
            r'\b[A-Z]{3,}\b',  # Acronyms (e.g., GAIA, MCP, PPO)
        ]
        
        from collections import Counter
        word_counts = Counter()
        for pattern in patterns:
            words = re.findall(pattern, raw_content)
            word_counts.update(words)
            
        # Filter out common non-research words and short words
        stop_words = {"this", "that", "with", "from", "their", "those", "these", "result", "framework", "system", "method", "paper", "Value", "Metric"}
        metadata["topics"] = [k for k, count in word_counts.items() if k.lower() not in stop_words and count >= 1]
        
        return metadata
        
    def discover_variables(self, existing_vars: List[str], limit: int = 5) -> List[str]:
        """Discover new potential variables from the corpus.
        
        Args:
            existing_vars: List of variable names already in the model.
            limit: Maximum number of new variables to suggest.
            
        Returns:
            List of suggested new variable names.
        """
        all_topics = self.get_all_topics()
        existing_lower = [v.lower() for v in existing_vars]
        
        candidates = []
        for topic in all_topics:
            if topic.lower() not in existing_lower:
                candidates.append(topic)
                
        self.logger.info(f"Discovered {len(candidates)} new potential variables from literature.")
        if candidates:
            self.logger.debug(f"Top candidates: {candidates[:5]}")
            
        # Return a diverse set
        import random
        random.shuffle(candidates)
        return candidates[:limit]
        
    def query(self, topics: List[str], max_results: int = 2) -> List[Dict[str, Any]]:
        """Query the corpus for papers matching the given topics.
        
        Args:
            topics: List of topics or keywords to search for.
            max_results: Maximum number of papers to return.
            
        Returns:
            List of matching papers.
        """
        results = []
        topics_lower = [t.lower() for t in topics]
        
        for paper in self.corpus:
            match_score = 0
            
            # Topic match
            for topic in paper["topics"]:
                if topic.lower() in topics_lower:
                    match_score += 2
            
            # Content match
            content = (paper["title"] + " " + paper["summary"]).lower()
            for topic in topics_lower:
                if topic in content:
                    match_score += 1
            
            if match_score > 0:
                results.append((match_score, paper))
        
        # Sort by match score descending
        results.sort(key=lambda x: x[0], reverse=True)
        
        final_results = [p for _, p in results[:max_results]]
        self.logger.info(f"Query for {topics} returned {len(final_results)} items")
        
        return final_results

    def get_all_topics(self) -> List[str]:
        """Get all detected topics in the corpus."""
        all_topics = set()
        for paper in self.corpus:
            for topic in paper["topics"]:
                all_topics.add(topic)
        return list(all_topics)
