import os
import requests
import json
from typing import Dict, Any, Tuple
from rag_web_agent.config import SERPER_API_KEY, WEB_CHUNKS_MIN_WORDS, CONFIDENCE_THRESHOLD

class WebSearchAgent:
    def __init__(self, temperature: float = 0.7):
        self.api_key = SERPER_API_KEY
        self.search_url = "https://google.serper.dev/search"

    def search(self, query: str) -> Dict[str, Any]:
        """Call Serper API and return raw JSON results."""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        payload = json.dumps({"q": query, "autocorrect": True})
        
        try:
            response = requests.request("POST", self.search_url, headers=headers, data=payload)
            return response.json()
        except Exception as e:
            print(f"Error during web search: {e}")
            return {}

    def extract_answer(self, results: Dict[str, Any]) -> Tuple[bool, str, float, int]:
        """
        Extract answer from Serper results without an LLM.
        Priority: 1. Answer Box, 2. Knowledge Graph, 3. Top Organic Snippet.
        """
        answer = ""
        confidence = 0.0
        
        # 1. Check Answer Box (Direct answer from Google)
        if "answerBox" in results:
            ab = results["answerBox"]
            answer = ab.get("answer") or ab.get("snippet") or ab.get("title", "")
            confidence = 0.95
        
        # 2. Check Knowledge Graph
        elif "knowledgeGraph" in results:
            kg = results["knowledgeGraph"]
            answer = kg.get("description", "")
            confidence = 0.85
        
        # 3. Fallback to top organic snippet
        elif "organic" in results and len(results["organic"]) > 0:
            top_result = results["organic"][0]
            answer = top_result.get("snippet", "")
            confidence = 0.6 # Lower confidence for raw snippets
        
        word_count = len(answer.split())
        is_sufficient = word_count >= WEB_CHUNKS_MIN_WORDS or confidence >= CONFIDENCE_THRESHOLD
        
        if not answer:
            return False, "NOT_ENOUGH_WEB_CONTEXT", 0.0, 0
            
        return is_sufficient, answer, confidence, word_count

    def run(self, query: str) -> Dict[str, Any]:
        """Entry point for web agent."""
        results = self.search(query)
        is_sufficient, answer, confidence, word_count = self.extract_answer(results)
        
        # Combine snippets for context field (compatibility with UI/RAG flow)
        snippets = []
        if "organic" in results:
            for res in results["organic"][:3]:
                snippets.append(res.get("snippet", ""))
        context = "\n".join(snippets)
        
        return {
            "mode": "web",
            "is_sufficient": is_sufficient,
            "answer": answer,
            "confidence": confidence,
            "word_count": word_count,
            "context": context
        }
