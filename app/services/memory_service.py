"""Memory Service using FAISS vector store and BGE Embeddings."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.core.config import Settings

logger = logging.getLogger(__name__)


class BGEMemoryEmbeddings(Embeddings):
    """Local embedding service using BAAI/bge-base-en-v1.5 via sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        logger.info("Initializing BGE Embeddings with model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of documents using the local BGE model."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the recommended retrieval prefix for BGE models."""
        instruction = "Represent this sentence for searching relevant passages: "
        embedding = self.model.encode(instruction + text, normalize_embeddings=True)
        return embedding.tolist()


class ResearchMemoryService:
    """FAISS memory service for storing research evidence chunks with semantic retrieval & reranking."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = BGEMemoryEmbeddings(settings.embedding_model)
        from app.agents.researcher.reranker import BGEReranker
        self.reranker = BGEReranker(settings.reranker_model)
        self.index_path = settings.faiss_index_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )
        self._vectorstore: FAISS | None = None

    def _get_vectorstore(self) -> FAISS | None:
        if self._vectorstore is not None:
            return self._vectorstore

        if os.path.exists(self.index_path) and os.path.exists(
            os.path.join(self.index_path, "index.faiss")
        ):
            try:
                logger.info("Loading existing FAISS index from: %s", self.index_path)
                self._vectorstore = FAISS.load_local(
                    self.index_path, self.embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.error("Failed to load local FAISS index: %s", e)
        return self._vectorstore

    async def ingest_evidence(self, evidence_items: List[Any]) -> None:
        """Chunk, embed, and ingest evidence items into the FAISS memory."""
        if not evidence_items:
            return

        texts = []
        metadatas = []
        for item in evidence_items:
            # item could be a dict or a Pydantic model
            text = getattr(item, "text", "") if hasattr(item, "text") else item.get("text", "")
            if not text:
                continue

            metadata = {
                "id": getattr(item, "id", "") if hasattr(item, "id") else item.get("id", ""),
                "source_id": getattr(item, "source_id", "")
                if hasattr(item, "source_id")
                else item.get("source_id", ""),
                "task_id": getattr(item, "task_id", "")
                if hasattr(item, "task_id")
                else item.get("task_id", ""),
                "relevance_score": float(
                    getattr(item, "relevance_score", 0.5)
                    if hasattr(item, "relevance_score")
                    else item.get("relevance_score", 0.5)
                ),
            }

            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append(metadata.copy())

        if not texts:
            return

        vectorstore = self._get_vectorstore()
        if vectorstore is None:
            logger.info("Creating new FAISS index at: %s", self.index_path)
            self._vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            logger.info("Adding %d chunks to existing FAISS index", len(texts))
            self._vectorstore.add_texts(texts, metadatas=metadatas)

        # Persist index to disk
        try:
            os.makedirs(self.index_path, exist_ok=True)
            self._vectorstore.save_local(self.index_path)
            logger.info("FAISS index saved successfully to: %s", self.index_path)
        except Exception as e:
            logger.error("Failed to save FAISS index: %s", e)

    async def retrieve_relevant(
        self, query: str, top_k: int | None = None, rerank_top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k chunks based on semantic similarity, then rerank using CrossEncoder."""
        vectorstore = self._get_vectorstore()
        if vectorstore is None:
            logger.info("No FAISS index available for retrieval.")
            return []

        k = top_k or self.settings.faiss_top_k
        rk = rerank_top_k or self.settings.rerank_top_k

        try:
            # similarity_search_with_relevance_scores returns List[Tuple[Document, float]]
            docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            logger.error("FAISS search failed: %s", e)
            return []

        candidates = []
        for doc, score in docs_and_scores:
            candidates.append(
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )

        # Apply BGE Reranker
        reranked = self.reranker.rerank(query, candidates, top_k=rk)
        return reranked

    def clear(self) -> None:
        """Clear memory cache and delete local index on disk."""
        self._vectorstore = None
        if os.path.exists(self.index_path):
            try:
                shutil.rmtree(self.index_path)
                logger.info("Deleted FAISS index directory: %s", self.index_path)
            except Exception as e:
                logger.error("Failed to delete FAISS index directory: %s", e)
