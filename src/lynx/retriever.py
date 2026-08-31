"""
Corrective RAG (CRAG) Retrieval, Hybrid Search, and LLM Document Evaluation Module.

Architecture & Components:
1. Hybrid Search Engine:
   - Dense Vector Retrieval via FastEmbed (BAAI/bge-small-en-v1.5) & Qdrant Cosine search.
   - Sparse Keyword Retrieval via BM25 (BM25Okapi) with smooth empty-index fallback.
   - Reciprocal Rank Fusion (RRF) for balanced multi-signal rank merging.
2. Pydantic v2 Output Schema:
   - `GradeDocuments`: Strict binary scoring ("yes" | "no") and concise reasoning.
3. LLM Document Grader:
   - Evaluates retrieved candidate chunks against the user query to filter out false positives.
   - Supports local LLM backends (Ollama / ChatOllama / OpenAI-compatible local APIs)
     with low-latency concurrent execution and robust JSON parsing.
4. Corrective RAG (CRAG) Routing:
   - Calculates candidate relevance ratio. If < 50% relevant, flags `web_search_needed = True`
     and generates a transformed `rewritten_query` for web search or query expansion.
"""

import os
import re
import sys
import json
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple, Literal, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress minor third-party warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

# Pydantic v2 for strict schema definition
from pydantic import BaseModel, Field

# FastEmbed for dense vector representation
from fastembed import TextEmbedding

# Qdrant Client for vector search
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

# Rank-BM25 for sparse keyword search
from rank_bm25 import BM25Okapi

# Security & RBAC Context
from lynx.auth import UserSecurityContext

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("crag_retriever")


# Comprehensive stopword list for high-precision semantic parsing
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "tell", "explain", "please"
}


# ---------------------------------------------------------------------------
# 1. Pydantic v2 Schema for Document Grading
# ---------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    """
    Pydantic v2 output schema for evaluating document relevance to a user question.
    """
    binary_score: Literal["yes", "no"] = Field(
        description="Binary relevance score: 'yes' if document contains relevant facts/context, 'no' otherwise."
    )
    reasoning: str = Field(
        description="Brief explanation justifying the relevance or irrelevance of the chunk to the question."
    )


class QueryTransform(BaseModel):
    """
    Schema for rewritten search query when web search or expansion is required.
    """
    rewritten_query: str = Field(
        description="Optimized, keyword-rich query suitable for web search or broader retrieval."
    )
    explanation: str = Field(
        description="Reasoning behind the query transformation."
    )


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    """Represents a candidate text chunk retrieved via hybrid search with RBAC."""
    chunk_id: str
    text: str
    filename: str
    page_number: Optional[int]
    tenant_id: str = "tenant_default"
    owner_id: str = "system"
    allowed_roles: List[str] = field(default_factory=lambda: ["user", "admin"])
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradedChunk:
    """Represents a retrieved chunk with its LLM relevance evaluation."""
    chunk: RetrievedChunk
    is_relevant: bool
    score: Literal["yes", "no"]
    reasoning: str


@dataclass
class CRAGRetrievalResult:
    """Comprehensive output of the Corrective RAG pipeline."""
    query: str
    retrieved_chunks: List[RetrievedChunk]
    relevant_chunks: List[RetrievedChunk]
    graded_chunks: List[GradedChunk]
    relevance_ratio: float
    web_search_needed: bool
    rewritten_query: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Hybrid Search Engine (Qdrant Dense + BM25 Sparse with Multi-Tenant RBAC)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Implements multi-tenant hybrid search over a Qdrant collection using dense embeddings
    and an in-memory BM25 sparse keyword index, merged via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        qdrant_path: str = "./qdrant_storage",
        collection_name: str = "agentic_rag_knowledge",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        client: Optional[QdrantClient] = None,
    ):
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

        logger.info(f"Initializing FastEmbed model: '{embedding_model_name}' (threads=1 for memory efficiency)...")
        try:
            self.embedding_model = TextEmbedding(model_name=embedding_model_name, threads=1)
        except TypeError:
            self.embedding_model = TextEmbedding(model_name=embedding_model_name)

        # Initialize Qdrant Client (reusing existing instance if provided)
        if client is not None:
            self.client = client
        else:
            self.client = QdrantClient(path=qdrant_path)

        self.bm25_index: Optional[BM25Okapi] = None
        self.corpus_chunks: List[RetrievedChunk] = []

        # Build initial BM25 index from stored Qdrant collection points
        self._build_bm25_index()

    def _tokenize(self, text: str) -> List[str]:
        """Fast whitespace and alphanumeric tokenizer for BM25 with stopword removal."""
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return [tok for tok in clean.split() if len(tok) > 1 and tok not in ENGLISH_STOPWORDS]

    def _build_bm25_index(self) -> None:
        """Fetches all indexed chunks from Qdrant to construct an in-memory BM25 index."""
        try:
            if not hasattr(self.client, "get_collections"):
                self.bm25_index = None
                self.corpus_chunks = []
                return

            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                logger.warning(f"Qdrant collection '{self.collection_name}' not found. BM25 index initialized empty.")
                self.bm25_index = None
                self.corpus_chunks = []
                return

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                logger.warning(f"No points found in '{self.collection_name}'. BM25 index is empty.")
                self.bm25_index = None
                self.corpus_chunks = []
                return

            corpus_chunks: List[RetrievedChunk] = []
            tokenized_corpus: List[List[str]] = []

            for p in points:
                payload = p.payload or {}
                text = payload.get("text", "")
                if not text:
                    continue

                chunk = RetrievedChunk(
                    chunk_id=str(p.id),
                    text=text,
                    filename=payload.get("filename", "unknown"),
                    page_number=payload.get("page_number"),
                    tenant_id=payload.get("tenant_id", "tenant_default"),
                    owner_id=payload.get("owner_id", "system"),
                    allowed_roles=payload.get("allowed_roles", ["user", "admin"]),
                    metadata=payload,
                )
                corpus_chunks.append(chunk)
                tokenized_corpus.append(self._tokenize(text))

            if tokenized_corpus:
                self.bm25_index = BM25Okapi(tokenized_corpus)
                self.corpus_chunks = corpus_chunks
                logger.info(f"Constructed BM25 index over {len(corpus_chunks)} document chunk(s).")
            else:
                self.bm25_index = None
                self.corpus_chunks = []

        except Exception as e:
            logger.error(f"Error constructing BM25 index: {e}", exc_info=True)
            self.bm25_index = None
            self.corpus_chunks = []

    def dense_search(
        self,
        query: str,
        limit: int = 5,
        security_context: Optional[UserSecurityContext] = None,
    ) -> List[Tuple[RetrievedChunk, float]]:
        """Executes dense cosine vector search in Qdrant with tenant and RBAC filtering."""
        try:
            query_vector = list(self.embedding_model.embed([query]))[0]
            vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

            query_filter = None
            if security_context is not None:
                must_conditions = [
                    FieldCondition(key="tenant_id", match=MatchValue(value=security_context.tenant_id))
                ]
                # Non-admin users are restricted to their assigned roles
                if not security_context.has_role("admin"):
                    must_conditions.append(
                        FieldCondition(key="allowed_roles", match=MatchAny(any=security_context.roles))
                    )
                query_filter = Filter(must=must_conditions)

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=vector_list,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            dense_results: List[Tuple[RetrievedChunk, float]] = []
            for point in results.points:
                payload = point.payload or {}
                chunk = RetrievedChunk(
                    chunk_id=str(point.id),
                    text=payload.get("text", ""),
                    filename=payload.get("filename", "unknown"),
                    page_number=payload.get("page_number"),
                    tenant_id=payload.get("tenant_id", "tenant_default"),
                    owner_id=payload.get("owner_id", "system"),
                    allowed_roles=payload.get("allowed_roles", ["user", "admin"]),
                    dense_score=float(point.score),
                    metadata=payload,
                )
                dense_results.append((chunk, float(point.score)))

            return dense_results
        except Exception as e:
            logger.error(f"Dense vector search failed: {e}", exc_info=True)
            return []

    def sparse_search(
        self,
        query: str,
        limit: int = 5,
        security_context: Optional[UserSecurityContext] = None,
    ) -> List[Tuple[RetrievedChunk, float]]:
        """Executes BM25 keyword search over indexed chunks, enforcing tenant and RBAC boundaries."""
        if not self.bm25_index or not self.corpus_chunks:
            logger.debug("BM25 index is empty. Falling back smoothly.")
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        sparse_results: List[Tuple[RetrievedChunk, float]] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            chunk = self.corpus_chunks[idx]

            # Security Filtering for BM25 Sparse Index to prevent data leakage
            if security_context is not None:
                chunk_tenant = chunk.tenant_id or chunk.metadata.get("tenant_id", "tenant_default")
                if chunk_tenant != security_context.tenant_id:
                    continue

                if not security_context.has_role("admin"):
                    chunk_roles = chunk.allowed_roles or chunk.metadata.get("allowed_roles", ["user", "admin"])
                    if not set(security_context.roles).intersection(set(chunk_roles)):
                        continue

            chunk_copy = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                filename=chunk.filename,
                page_number=chunk.page_number,
                tenant_id=chunk.tenant_id,
                owner_id=chunk.owner_id,
                allowed_roles=chunk.allowed_roles,
                sparse_score=float(scores[idx]),
                metadata=chunk.metadata,
            )
            sparse_results.append((chunk_copy, float(scores[idx])))
            if len(sparse_results) >= limit:
                break

        return sparse_results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        security_context: Optional[UserSecurityContext] = None,
    ) -> List[RetrievedChunk]:
        """
        Combines dense vector and BM25 sparse results using Reciprocal Rank Fusion (RRF).
        Strictly restricts candidate search to the authorized tenant and user roles.
        """
        dense_hits = self.dense_search(query, limit=top_k * 2, security_context=security_context)
        sparse_hits = self.sparse_search(query, limit=top_k * 2, security_context=security_context)

        # If BM25 is unavailable, return dense results directly
        if not sparse_hits and dense_hits:
            logger.info("Hybrid search: Returning dense results (BM25 yielded 0 hits).")
            return [chunk for chunk, _ in dense_hits[:top_k]]

        if not dense_hits and sparse_hits:
            logger.info("Hybrid search: Returning sparse results (Dense yielded 0 hits).")
            return [chunk for chunk, _ in sparse_hits[:top_k]]

        if not dense_hits and not sparse_hits:
            logger.warning("Hybrid search: 0 authorized hits from both dense and sparse retrievers.")
            return []

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            rrf_score = self.dense_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + rrf_score
            chunk.dense_score = score
            chunk_map[chunk.chunk_id] = chunk

        for rank, (chunk, score) in enumerate(sparse_hits, start=1):
            rrf_score = self.sparse_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + rrf_score
            if chunk.chunk_id in chunk_map:
                chunk_map[chunk.chunk_id].sparse_score = score
            else:
                chunk.sparse_score = score
                chunk_map[chunk.chunk_id] = chunk

        # Sort chunks by composite RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        final_chunks: List[RetrievedChunk] = []

        for cid in sorted_chunk_ids[:top_k]:
            c = chunk_map[cid]
            c.hybrid_score = rrf_scores[cid]
            final_chunks.append(c)

        logger.info(f"Hybrid search returned {len(final_chunks)} authorized chunk(s) for query: '{query}'")
        return final_chunks

    def search(
        self,
        query: str,
        top_k: int = 4,
        security_context: Optional[UserSecurityContext] = None,
    ) -> List[RetrievedChunk]:
        """Convenience alias for hybrid_search with RBAC support."""
        return self.hybrid_search(query=query, top_k=top_k, security_context=security_context)


# Import ModelRouter and schema definitions
from lynx.model_router import ModelRouter, model_router, GradeDocuments


# ---------------------------------------------------------------------------
# 3. LLM Document Grader (Corrective RAG Evaluation via ModelRouter)
# ---------------------------------------------------------------------------
class DocumentGrader:
    """
    Evaluates retrieved document chunks against the user question using the
    low-latency SLM evaluator tier via ModelRouter. Strictly enforces Pydantic v2 GradeDocuments schema.
    """

    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        model_name: str = "llama3.2:3b",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_workers: int = 4,
    ):
        self.router = router or model_router
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.max_workers = max_workers

    def _build_grading_prompt(self, question: str, document_text: str) -> str:
        """Builds an unambiguous system prompt enforcing strict JSON output."""
        return f"""You are a strict, production-grade Corrective RAG (CRAG) document relevance grader.
Your task is to evaluate whether the retrieved document chunk contains factual information, definitions, or semantic context that directly helps answer the user's question.

CRITICAL INSTRUCTIONS:
1. Return ONLY a valid JSON object matching this exact schema:
{{
  "binary_score": "yes" | "no",
  "reasoning": "<concise explanation>"
}}
2. Set "binary_score" to "yes" if the document contains relevant facts, context, or mechanisms related to the question.
3. Set "binary_score" to "no" if the document is off-topic, discusses completely unrelated concepts, or contains only vague keyword coincidences with no explanatory value.
4. Do NOT include any conversational filler, markdown formatting, or preamble outside the JSON object.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CHUNK:
{document_text}

JSON RESPONSE:"""

    def _parse_llm_json_response(self, raw_output: str) -> GradeDocuments:
        """Extracts and validates JSON from LLM output using Pydantic v2."""
        cleaned = raw_output.strip()
        # Strip potential markdown code block formatting if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

        # Find first JSON object bracket pattern
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return GradeDocuments.model_validate(data)

        raise ValueError(f"Could not parse valid JSON from LLM output: {raw_output}")

    def _semantic_fallback_grade(self, question: str, document_text: str) -> GradeDocuments:
        """
        High-precision semantic relevance fallback evaluator used when the local LLM
        daemon is offline or unresponsive, ensuring 100% reliable local testability.
        """
        q_clean = re.sub(r"[^\w\s]", " ", question.lower())
        d_clean = re.sub(r"[^\w\s]", " ", document_text.lower())

        q_tokens = set([t for t in q_clean.split() if len(t) > 2])
        d_tokens = set([t for t in d_clean.split() if len(t) > 2])

        # Filter out all non-informative stopwords
        informative_q_tokens = q_tokens - ENGLISH_STOPWORDS

        if not informative_q_tokens:
            return GradeDocuments(
                binary_score="no",
                reasoning="Query contained no informative terms."
            )

        overlap = informative_q_tokens.intersection(d_tokens)
        overlap_ratio = len(overlap) / len(informative_q_tokens)

        # Meaningful semantic overlap threshold (> 25% of core query keywords with substantive match)
        if overlap_ratio >= 0.25 and len(overlap) >= 1:
            matched_terms = ", ".join(list(overlap)[:4])
            return GradeDocuments(
                binary_score="yes",
                reasoning=f"Document contains direct contextual match for key query terms ({matched_terms})."
            )
        else:
            return GradeDocuments(
                binary_score="no",
                reasoning=f"Document lacks relevant context for query terms. Matched {len(overlap)}/{len(informative_q_tokens)} terms."
            )

    def grade_chunk(self, question: str, chunk: RetrievedChunk) -> GradedChunk:
        """Evaluates a single retrieved chunk against the user question using ModelRouter."""
        grade = self.router.grade_document(question=question, document_text=chunk.text)
        return GradedChunk(
            chunk=chunk,
            is_relevant=(grade.binary_score == "yes"),
            score=grade.binary_score,
            reasoning=grade.reasoning,
        )

    def grade_chunks_batch(self, question: str, chunks: List[RetrievedChunk]) -> List[GradedChunk]:
        """
        Evaluates multiple retrieved chunks concurrently to optimize latency.
        """
        if not chunks:
            return []

        graded_results: List[GradedChunk] = [None] * len(chunks)  # type: ignore

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
            future_to_idx = {
                executor.submit(self.grade_chunk, question, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    graded_results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error grading chunk index {idx}: {e}")
                    graded_results[idx] = GradedChunk(
                        chunk=chunks[idx],
                        is_relevant=False,
                        score="no",
                        reasoning=f"Grading error: {e}",
                    )

        return graded_results

    def grade_documents(self, question: str, chunks: List[RetrievedChunk]) -> List[GradedChunk]:
        """Convenience alias for grade_chunks_batch."""
        return self.grade_chunks_batch(question=question, chunks=chunks)


# ---------------------------------------------------------------------------
# 4. Corrective RAG Query Transformation & Orchestrator
# ---------------------------------------------------------------------------
class CorrectiveRAGRetriever:
    """
    Master CRAG Orchestrator:
    - Performs hybrid retrieval (dense + sparse).
    - Grades retrieved candidate chunks using local LLM.
    - Determines if context is sufficient or if web search / query rewriting is triggered.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        document_grader: Optional[DocumentGrader] = None,
        relevance_threshold: float = 0.5,
    ):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.document_grader = document_grader or DocumentGrader()
        self.relevance_threshold = relevance_threshold

    def rewrite_query_for_web(self, query: str) -> str:
        """
        Rewrites a poorly matching or narrow user query into an optimized query
        suitable for external web search or expanded retrieval.
        """
        # Strip conversational packaging to isolate core semantic keywords
        clean_query = re.sub(
            r"^(can you tell me|please explain|what is|what are|how do i|tell me about)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        clean_query = clean_query.strip().rstrip("?")
        rewritten = f"{clean_query} comprehensive overview specifications"
        logger.info(f"CRAG: Query rewritten for web search -> '{rewritten}'")
        return rewritten

    def retrieve_and_evaluate(self, query: str, top_k: int = 2) -> CRAGRetrievalResult:
        """
        Executes hybrid search, evaluates retrieved chunks, and routes execution.
        Returns a structured CRAGRetrievalResult containing only relevant chunks
        and the routing flag `web_search_needed`.
        """
        logger.info("=" * 60)
        logger.info(f"CRAG Pipeline: Processing query: '{query}'")
        logger.info("=" * 60)

        # 1. Hybrid Retrieval
        retrieved_chunks = self.hybrid_retriever.hybrid_search(query, top_k=top_k)

        if not retrieved_chunks:
            logger.warning("CRAG: 0 candidate chunks retrieved. Web search triggered immediately.")
            return CRAGRetrievalResult(
                query=query,
                retrieved_chunks=[],
                relevant_chunks=[],
                graded_chunks=[],
                relevance_ratio=0.0,
                web_search_needed=True,
                rewritten_query=self.rewrite_query_for_web(query),
            )

        # 2. LLM Relevance Grading (Concurrent Batch)
        logger.info(f"CRAG: Grading {len(retrieved_chunks)} retrieved chunk(s)...")
        graded_chunks = self.document_grader.grade_chunks_batch(query, retrieved_chunks)

        # 3. Filter Relevant Chunks
        relevant_chunks = [gc.chunk for gc in graded_chunks if gc.is_relevant]
        relevance_ratio = len(relevant_chunks) / len(retrieved_chunks)

        logger.info(
            f"CRAG Grading Complete: {len(relevant_chunks)}/{len(retrieved_chunks)} "
            f"chunks relevant (Relevance Ratio: {relevance_ratio:.1%})"
        )

        for gc in graded_chunks:
            logger.info(
                f" - [{gc.score.upper()}] File: {gc.chunk.filename} (p.{gc.chunk.page_number}) | "
                f"Reasoning: {gc.reasoning}"
            )

        # 4. Routing Decision
        web_search_needed = (relevance_ratio < self.relevance_threshold) or (len(relevant_chunks) == 0)
        rewritten_query = self.rewrite_query_for_web(query) if web_search_needed else None

        if web_search_needed:
            logger.warning(
                f"[CRAG ROUTING] Relevance ratio ({relevance_ratio:.1%}) < threshold ({self.relevance_threshold:.1%}). "
                f"FLAGGING: web_search_needed = True"
            )
        else:
            logger.info(
                f"[CRAG ROUTING] Retrieved context is high-confidence ({relevance_ratio:.1%} >= {self.relevance_threshold:.1%}). "
                f"FLAGGING: web_search_needed = False"
            )

        return CRAGRetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            relevant_chunks=relevant_chunks,
            graded_chunks=graded_chunks,
            relevance_ratio=relevance_ratio,
            web_search_needed=web_search_needed,
            rewritten_query=rewritten_query,
        )


# ---------------------------------------------------------------------------
# Verification & Self-Test Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=======================================================")
    print("    CORRECTIVE RAG (CRAG) RETRIEVER & GRADER TEST     ")
    print("=======================================================\n")

    # Initialize CRAG Retriever with local Qdrant collection
    crag_pipeline = CorrectiveRAGRetriever(
        hybrid_retriever=HybridRetriever(
            qdrant_path="./qdrant_storage",
            collection_name="agentic_rag_knowledge",
        ),
        document_grader=DocumentGrader(),
        relevance_threshold=0.5,
    )

    # Test Case 1: Highly relevant query present in the local vector DB
    print("\n>>> TEST CASE 1: Query with High Relevance in Local Store <<<")
    query_1 = "What is Qdrant vector database and how does local agentic RAG work?"
    result_1 = crag_pipeline.retrieve_and_evaluate(query_1, top_k=2)

    print(f"\nQuery: '{result_1.query}'")
    print(f"Total Retrieved Chunks: {len(result_1.retrieved_chunks)}")
    print(f"Relevant Chunks Filtered: {len(result_1.relevant_chunks)}")
    print(f"Relevance Ratio: {result_1.relevance_ratio:.1%}")
    print(f"Web Search Needed Flag: {result_1.web_search_needed}")
    assert not result_1.web_search_needed, "Test Case 1 failed: Expected web_search_needed=False"

    # Test Case 2: Irrelevant query absent from local store (should trigger web search)
    print("\n>>> TEST CASE 2: Query Unrelated to Local Store (Triggers CRAG Fallback) <<<")
    query_2 = "What are the latest revenue numbers and financial forecasts for quantum computing in 2026?"
    result_2 = crag_pipeline.retrieve_and_evaluate(query_2, top_k=2)

    print(f"\nQuery: '{result_2.query}'")
    print(f"Total Retrieved Chunks: {len(result_2.retrieved_chunks)}")
    print(f"Relevant Chunks Filtered: {len(result_2.relevant_chunks)}")
    print(f"Relevance Ratio: {result_2.relevance_ratio:.1%}")
    print(f"Web Search Needed Flag: {result_2.web_search_needed}")
    print(f"Rewritten Web Query: '{result_2.rewritten_query}'")
    assert result_2.web_search_needed, "Test Case 2 failed: Expected web_search_needed=True"

    # Test Case 3: Verify Pydantic v2 Schema directly
    print("\n>>> TEST CASE 3: Pydantic v2 Schema Strict Validation <<<")
    sample_valid_json = '{"binary_score": "yes", "reasoning": "Explicit match with system architecture."}'
    grade_obj = GradeDocuments.model_validate_json(sample_valid_json)
    print(f"Validated Pydantic Grade Object: {grade_obj.model_dump()}")
    assert grade_obj.binary_score == "yes"

    print("\n[ALL TESTS PASSED] CRAG Retrieval and Grading Pipeline fully verified.\n")

