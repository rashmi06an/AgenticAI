import pandas as pd
import os
import time
from typing import List, Dict, Any
from rag_web_agent.web_agent import WebSearchAgent
from rag_web_agent.rag_agent import RAGAgent
from rag_web_agent.config import EXPERIMENTS_FILE, DEFAULT_CONFIG

class ExperimentRunner:
    def __init__(self):
        self.web_agent = WebSearchAgent()
        # RAG agent will be initialized with specific configs

    def run_query(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Web Search First
        web_result = self.web_agent.run(query)
        
        if web_result["is_sufficient"]:
            result = {
                "query": query,
                "answer": web_result["answer"],
                "answer_len": len(web_result["answer"]),
                "retrieval_k": 0,
                "num_sources": 5, # Serper top 5
                "self_confidence": web_result["confidence"],
                "self_quality": 4, # Heuristic
                "mode_used": "web",
                "time_taken": round(time.time() - start_time, 2)
            }
        else:
            # 2. Fallback to RAG
            rag_agent = RAGAgent(config)
            rag_result = rag_agent.query(query)
            
            result = {
                "query": query,
                "answer": rag_result["answer"],
                "answer_len": len(rag_result["answer"]),
                "retrieval_k": config.get("k", 3),
                "num_sources": len(rag_result.get("sources", [])),
                "self_confidence": rag_result["confidence"],
                "self_quality": 4 if "Not found" not in rag_result["answer"] else 1,
                "mode_used": "docs",
                "time_taken": round(time.time() - start_time, 2)
            }
        
        # Add config params to result
        result.update(config)
        return result

    def run_experiments(self, queries: List[str], configs: List[Dict[str, Any]]):
        all_results = []
        
        for query in queries:
            for config in configs:
                print(f"Running query: '{query}' with config: {config}")
                result = self.run_query(query, config)
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(EXPERIMENTS_FILE), exist_ok=True)
        
        # Append if exists, else create
        if os.path.exists(EXPERIMENTS_FILE):
             df.to_csv(EXPERIMENTS_FILE, mode='a', header=False, index=False)
        else:
             df.to_csv(EXPERIMENTS_FILE, index=False)
        
        print(f"Experiments saved to {EXPERIMENTS_FILE}")

if __name__ == "__main__":
    # Sample queries and configurations for testing
    test_queries = [
        "What is the latest advancement in Quantum Computing in 2024?",
        "How do RAG systems handle document indexing?",
    ]
    
    test_configs = [
        {"chunk_size": 500, "chunk_overlap": 50, "k": 3, "chain_type": "stuff"},
        {"chunk_size": 1000, "chunk_overlap": 100, "k": 5, "chain_type": "map_reduce"}
    ]
    
    runner = ExperimentRunner()
    runner.run_experiments(test_queries, test_configs)
