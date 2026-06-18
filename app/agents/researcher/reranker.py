"""BGE Reranker service using CrossEncoder for scoring search results."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class BGEReranker:
    """Reranker using BAAI/bge-reranker-base to score search results or retrieved chunks."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        logger.info("Initializing BGE Reranker with model: %s", model_name)
        # CrossEncoder handles scoring query-document pairs
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank list of documents using the CrossEncoder model.

        Each document should have a 'text' field.
        """
        if not documents:
            return []

        # Create pairs [query, doc_text]
        pairs = [[query, doc.get("text", "")] for doc in documents]
        
        try:
            # Predict scores (higher is more relevant)
            scores = self.model.predict(pairs)
            
            # Map scores to documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            # Sort by score descending
            ranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            return ranked_docs[:top_k]
        except Exception as e:
            logger.error("Reranking failed: %s. Returning top_k from original list.", e)
            return documents[:top_k]
