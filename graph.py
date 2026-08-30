"""
Corrective RAG (CRAG) & Self-RAG Agentic Workflow Engine with Heterogeneous Model Routing.

Architecture & Heterogeneous Routing:
1. Low-Latency SLM Evaluator Tier (`model_router.evaluator_llm`):
   - `grade_documents_node`: Fast binary chunk relevance scoring (`GradeDocuments`).
   - `rewrite_query_node`: Fast keyword query expansion (`RewrittenQuery`).
   - `hallucination_grader_node`: Fast Self-RAG grounding verification (`GradeHallucinations`).
2. High-Capacity Synthesizer Tier (`model_router.synthesizer_llm`):
   - `generate_node`: Synthesizes detailed grounded responses with citations (Groq 70B / Local 14B / Ollama).
   - Dynamic Fallback: Transparently falls back to local Ollama upon Groq 429 rate limit or missing API key.
3. LangGraph Cyclic Flow:
   - Hybrid Retrieval -> SLM Grade -> (Relevance < 50% ? SLM Rewrite -> Re-retrieve : Generate)
   - Max 2 retries -> Automated DuckDuckGo Web Fallback.
   - Generate -> SLM Hallucination Grader -> (Ungrounded ? Regenerate with feedback : End).

Author: High-Performance AI Inference Architect
"""

import os
import re
import sys
import json
import logging
import warnings
from typing import List, Dict, Any, Optional, Literal, TypedDict, Set
from dataclasses import asdict

# Suppress minor third-party deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

# Pydantic v2 for structured evaluation schemas
from pydantic import BaseModel, Field

# LangChain message abstractions
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# LangGraph state machine components
from langgraph.graph import StateGraph, START, END

# Import verified CRAG hybrid retriever, web search, and heterogeneous model router
from retriever import HybridRetriever, DocumentGrader, RetrievedChunk
from web_search import web_search_client, WebSearchEngine
from model_router import ModelRouter, model_router, GradeDocuments, GradeHallucinations, RewrittenQuery

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("crag_agent")


from auth import UserSecurityContext
from observability import trace_agent_node, setup_observability

# Initialize observability telemetry provider
setup_observability()


# ---------------------------------------------------------------------------
# 1. Agent State TypedDict
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    Complete state representation passed across all nodes in the LangGraph graph.
    """
    question: str
    current_query: str
    messages: List[BaseMessage]
    documents: List[Dict[str, Any]]
    generation: str
    retrieval_retry_count: int
    generation_retry_count: int
    route_status: str
    web_search_needed: bool
    web_search_executed: bool
    hallucination_grade: Optional[str]
    hallucination_feedback: Optional[str]
    citations: List[str]
    security_context: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# 2. Agentic Workflow Nodes
# ---------------------------------------------------------------------------
class CRAGWorkflowEngine:
    """
    Encapsulates all node logic, heterogeneous model routing, web search fallbacks,
    and state transitions.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        grader: Optional[DocumentGrader] = None,
        web_search: Optional[WebSearchEngine] = None,
        router: Optional[ModelRouter] = None,
        max_retrieval_retries: int = 2,
        max_generation_retries: int = 2,
    ):
        self.router = router or model_router
        self.retriever = retriever or HybridRetriever()
        self.grader = grader or DocumentGrader(router=self.router)
        self.web_search = web_search or web_search_client
        self.max_retrieval_retries = max_retrieval_retries
        self.max_generation_retries = max_generation_retries

    # -----------------------------------------------------------------------
    # Node 1: Retrieve Node (with Multi-Tenant RBAC & OTel Spans)
    # -----------------------------------------------------------------------
    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes hybrid dense vector and sparse BM25 retrieval over Qdrant
        strictly constrained to the user's tenant_id and authorized RBAC roles.
        """
        query = state.get("current_query") or state["question"]
        raw_sec = state.get("security_context") or {}
        sec_ctx = None
        if raw_sec:
            try:
                sec_ctx = UserSecurityContext.model_validate(raw_sec)
            except Exception as e:
                logger.warning(f"Failed to validate security_context: {e}")

        tenant_id = sec_ctx.tenant_id if sec_ctx else "default"
        user_id = sec_ctx.user_id if sec_ctx else "anon"
        tenant_info = f" [Tenant: {tenant_id}, Roles: {sec_ctx.roles}]" if sec_ctx else " [Tenant: Default]"
        logger.info(f"--- [NODE: RETRIEVE] Fetching context for: '{query}'{tenant_info} ---")

        with trace_agent_node(
            "retrieve_node",
            inputs={"query": query},
            attributes={
                "tenant.id": tenant_id,
                "user.id": user_id,
                "retrieval.retry_count": state.get("retrieval_retry_count", 0),
            },
        ) as span:
            retrieved_chunks = self.retriever.search(query=query, top_k=3, security_context=sec_ctx)
            doc_dicts = [asdict(chunk) for chunk in retrieved_chunks]

            if span:
                span.set_attribute("retrieved.chunk_count", len(doc_dicts))

            logger.info(f"[NODE: RETRIEVE] Retrieved {len(doc_dicts)} authorized candidate chunk(s).")
            return {
                "documents": doc_dicts,
                "route_status": "graded",
            }

    # -----------------------------------------------------------------------
    # Node 2: Grade Documents Node (CRAG Filter via SLM Evaluator)
    # -----------------------------------------------------------------------
    def grade_documents_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Evaluates relevance of each retrieved chunk using the SLM evaluator tier.
        Flags web search or query rewrite if relevance < 50%.
        """
        question = state["question"]
        raw_docs = state.get("documents", [])
        raw_sec = state.get("security_context") or {}
        tenant_id = raw_sec.get("tenant_id", "default")
        user_id = raw_sec.get("user_id", "anon")

        logger.info(f"--- [NODE: GRADE DOCUMENTS] Evaluating {len(raw_docs)} chunks via SLM against: '{question}' ---")

        with trace_agent_node(
            "grade_documents_node",
            inputs={"question": question, "candidate_chunk_count": len(raw_docs)},
            attributes={"tenant.id": tenant_id, "user.id": user_id},
        ) as span:
            if not raw_docs:
                logger.warning("[NODE: GRADE DOCUMENTS] 0 chunks retrieved. Routing to rewrite/web search.")
                if span:
                    span.set_attribute("grading.status", "no_chunks")
                return {
                    "documents": [],
                    "route_status": "rewrite",
                    "web_search_needed": True,
                }

            chunks = [
                RetrievedChunk(
                    chunk_id=d.get("chunk_id", ""),
                    text=d.get("text", ""),
                    filename=d.get("filename", "unknown"),
                    page_number=d.get("page_number", 1),
                    hybrid_score=d.get("hybrid_score", d.get("score", 0.0)),
                    metadata=d.get("metadata", {}),
                )
                for d in raw_docs
            ]

            graded_chunks = self.grader.grade_documents(question=question, chunks=chunks)

            relevant_docs = []
            for graded in graded_chunks:
                fn = graded.chunk.filename
                pn = graded.chunk.page_number
                reason = graded.reasoning
                logger.info(f" - [{graded.score.upper()}] File: {fn} (p.{pn}) | {reason}")

                if graded.score == "yes":
                    relevant_docs.append(asdict(graded.chunk))

            relevance_ratio = len(relevant_docs) / len(chunks) if chunks else 0.0
            logger.info(f"[NODE: GRADE DOCUMENTS] Relevant Chunks: {len(relevant_docs)}/{len(chunks)} ({relevance_ratio:.1%})")

            if span:
                span.set_attribute("relevant.chunk_count", len(relevant_docs))
                span.set_attribute("relevance.ratio", round(relevance_ratio, 3))

            if relevance_ratio >= 0.50:
                logger.info("[NODE: GRADE DOCUMENTS] Sufficient context validated. Flagged for generation.")
                return {
                    "documents": relevant_docs,
                    "route_status": "generate",
                    "web_search_needed": False,
                }
            else:
                logger.info("[NODE: GRADE DOCUMENTS] Context insufficient. Flagged for query rewrite/web fallback.")
                return {
                    "documents": relevant_docs,
                    "route_status": "rewrite",
                    "web_search_needed": True,
                }

    # -----------------------------------------------------------------------
    # Node 3: Rewrite Query Node (SLM Evaluator)
    # -----------------------------------------------------------------------
    def rewrite_query_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Transforms conversational user question into an optimized vector query via SLM.
        """
        question = state["question"]
        current_retry = state.get("retrieval_retry_count", 0) + 1
        raw_sec = state.get("security_context") or {}
        tenant_id = raw_sec.get("tenant_id", "default")
        user_id = raw_sec.get("user_id", "anon")

        logger.info(f"--- [NODE: REWRITE QUERY] (Attempt {current_retry}/{self.max_retrieval_retries} via SLM) ---")

        with trace_agent_node(
            "rewrite_query_node",
            inputs={"original_question": question, "retry_attempt": current_retry},
            attributes={"tenant.id": tenant_id, "user.id": user_id, "retry_count": current_retry},
        ) as span:
            rewrite_result = self.router.rewrite_query(question=question, attempt=current_retry)
            new_query = rewrite_result.optimized_query

            if span:
                span.set_attribute("rewritten.query", new_query)
                span.set_attribute("rewrite.intent", rewrite_result.intent)

            logger.info(f"[NODE: REWRITE QUERY] Transformed query -> '{new_query}' (Intent: {rewrite_result.intent})")
            return {
                "current_query": new_query,
                "retrieval_retry_count": current_retry,
            }

    # -----------------------------------------------------------------------
    # Node 4: Web Search Node (Automated Fallback)
    # -----------------------------------------------------------------------
    def web_search_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes live DuckDuckGo web search when local context is exhausted.
        """
        question = state["question"]
        existing_docs = state.get("documents", [])
        raw_sec = state.get("security_context") or {}
        tenant_id = raw_sec.get("tenant_id", "default")
        user_id = raw_sec.get("user_id", "anon")

        logger.info(f"--- [NODE: WEB SEARCH] Fallback to live search for: '{question}' ---")

        with trace_agent_node(
            "web_search_node",
            inputs={"query": question, "existing_context_chunks": len(existing_docs)},
            attributes={"tenant.id": tenant_id, "user.id": user_id},
        ) as span:
            web_chunks = self.web_search.search(query=question, max_results=3)
            combined_docs = existing_docs + web_chunks

            if span:
                span.set_attribute("web_search.chunks_retrieved", len(web_chunks))
                span.set_attribute("total.context_count", len(combined_docs))

            logger.info(f"[NODE: WEB SEARCH] Retrieved {len(web_chunks)} live web chunk(s). Total context: {len(combined_docs)}.")
            return {
                "documents": combined_docs,
                "web_search_executed": True,
                "web_search_needed": False,
                "route_status": "generate",
            }

    # -----------------------------------------------------------------------
    # Node 5: Generate Node (High-Capacity Synthesizer Tier)
    # -----------------------------------------------------------------------
    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesizes grounded answer using high-capacity LLM (Groq / 70B / 14B / Ollama)
        with structured in-line citations and dynamic 429 rate limit fallback.
        """
        question = state["question"]
        docs = state.get("documents", [])
        feedback = state.get("hallucination_feedback")
        current_gen_retry = state.get("generation_retry_count", 0)
        raw_sec = state.get("security_context") or {}
        tenant_id = raw_sec.get("tenant_id", "default")
        user_id = raw_sec.get("user_id", "anon")

        tier_badge = "Groq LPU" if self.router.is_groq_active else "High-Capacity Local"
        logger.info(
            f"--- [NODE: GENERATE] Synthesizing answer via {tier_badge} "
            f"(Gen Retry: {current_gen_retry}/{self.max_generation_retries}) ---"
        )

        with trace_agent_node(
            "generate_node",
            inputs={"question": question, "context_chunk_count": len(docs), "has_feedback": bool(feedback)},
            attributes={
                "tenant.id": tenant_id,
                "user.id": user_id,
                "model.tier": tier_badge,
                "generation.retry_count": current_gen_retry,
            },
        ) as span:
            citations_list = []
            for doc in docs:
                if doc.get("is_web", False):
                    title = doc.get("title", "Web Result")
                    url = doc.get("source_url", "https://duckduckgo.com")
                    citations_list.append(f"[Web Source: {title}]({url})")
                else:
                    fn = doc.get("filename", "unknown")
                    pn = doc.get("page_number", 1)
                    citations_list.append(f"[Source: {fn}, Page: {pn}]")

            raw_generation = self.router.synthesize_answer(
                question=question,
                documents=docs,
                feedback=feedback,
            )

            citations_set = list(set(citations_list))
            if span:
                span.set_attribute("generation.char_length", len(raw_generation))
                span.set_attribute("citations.count", len(citations_set))

            logger.info("[NODE: GENERATE] Answer synthesis complete.")
            return {
                "generation": raw_generation,
                "citations": citations_set,
                "hallucination_feedback": None,
            }

    # -----------------------------------------------------------------------
    # Node 6: Hallucination Grader Node (Self-RAG Grounding via SLM)
    # -----------------------------------------------------------------------
    def hallucination_grader_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Validates factual grounding of generation against context docs via SLM Evaluator.
        """
        docs = state.get("documents", [])
        generation = state.get("generation", "")
        current_gen_retry = state.get("generation_retry_count", 0)
        raw_sec = state.get("security_context") or {}
        tenant_id = raw_sec.get("tenant_id", "default")
        user_id = raw_sec.get("user_id", "anon")

        logger.info("--- [NODE: HALLUCINATION GRADER] Self-RAG Grounding Verification via SLM ---")

        with trace_agent_node(
            "hallucination_grader_node",
            inputs={"generation_snippet": generation[:180], "context_chunk_count": len(docs)},
            attributes={"tenant.id": tenant_id, "user.id": user_id},
        ) as span:
            if not generation or not docs:
                logger.info("[NODE: HALLUCINATION GRADER] Pass-through for empty generation/context.")
                if span:
                    span.set_attribute("grounding.status", "pass_through")
                return {
                    "hallucination_grade": "yes",
                    "hallucination_feedback": None,
                    "generation_retry_count": current_gen_retry,
                }

            grade_res = self.router.grade_hallucination(generation=generation, documents=docs)
            score = grade_res.binary_score
            reasoning = grade_res.reasoning

            if span:
                span.set_attribute("grounding.grade", score)
                span.set_attribute("grounding.reasoning", reasoning)

            logger.info(f"[NODE: HALLUCINATION GRADER] Grounded: '{score.upper()}' | {reasoning}")

            return {
                "hallucination_grade": score,
                "hallucination_feedback": reasoning if score == "no" else None,
                "generation_retry_count": current_gen_retry + (1 if score == "no" else 0),
            }


# ---------------------------------------------------------------------------
# 3. Conditional Edge Routing Functions
# ---------------------------------------------------------------------------
def decide_to_generate_or_rewrite(state: AgentState) -> str:
    """
    Evaluates state after document grading:
    - If relevance >= 50% -> 'generate_node'
    - If relevance < 50% and retries < 2 -> 'rewrite_query_node'
    - If relevance < 50% and retries >= 2 -> 'web_search_node' (Automated Web Fallback)
    """
    route_status = state.get("route_status", "generate")
    retrieval_retries = state.get("retrieval_retry_count", 0)

    if route_status == "generate":
        logger.info("[ROUTER] Documents validated. Routing -> 'generate_node'")
        return "generate_node"

    if retrieval_retries < 2:
        logger.info(f"[ROUTER] Routing -> 'rewrite_query_node' (Retry {retrieval_retries + 1}/2)")
        return "rewrite_query_node"
    else:
        logger.warning(
            f"[ROUTER: RETRY LIMIT REACHED] Local retrieval exhausted ({retrieval_retries}/2). "
            f"Routing -> 'web_search_node' (Live Web Fallback)"
        )
        return "web_search_node"


def decide_to_finalize_or_regenerate(state: AgentState) -> str:
    """
    Evaluates state after Self-RAG hallucination checking:
    - If grounded ('yes') or generation_retries >= 2 -> END
    - If hallucinated ('no') and generation_retries < 2 -> 'generate_node'
    """
    hallucination_grade = state.get("hallucination_grade", "yes")
    gen_retries = state.get("generation_retry_count", 0)

    if hallucination_grade == "yes":
        logger.info("[ROUTER: CRAG COMPLETE] Grounded response approved. Routing -> END.")
        return END

    if gen_retries < 2:
        logger.warning(f"[ROUTER: HALLUCINATION DETECTED] Re-generating with grounding feedback (Retry {gen_retries}/2)")
        return "generate_node"
    else:
        logger.warning("[ROUTER: MAX GEN RETRIES] Max generation retries reached. Routing -> END.")
        return END


# ---------------------------------------------------------------------------
# 4. State Machine Graph Builder
def create_crag_graph(
    engine: Optional[CRAGWorkflowEngine] = None,
    retriever: Optional[HybridRetriever] = None,
    grader: Optional[DocumentGrader] = None,
    web_search: Optional[WebSearchEngine] = None,
    router: Optional[ModelRouter] = None,
    max_retrieval_retries: int = 2,
    max_generation_retries: int = 2,
):
    """
    Compiles the cyclic LangGraph CRAG & Self-RAG state machine with Heterogeneous Model Routing.
    """
    if engine is not None:
        workflow_engine = engine
    else:
        workflow_engine = CRAGWorkflowEngine(
            retriever=retriever,
            grader=grader,
            web_search=web_search,
            router=router,
            max_retrieval_retries=max_retrieval_retries,
            max_generation_retries=max_generation_retries,
        )

    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("retrieve_node", workflow_engine.retrieve_node)
    workflow.add_node("grade_documents_node", workflow_engine.grade_documents_node)
    workflow.add_node("rewrite_query_node", workflow_engine.rewrite_query_node)
    workflow.add_node("web_search_node", workflow_engine.web_search_node)
    workflow.add_node("generate_node", workflow_engine.generate_node)
    workflow.add_node("hallucination_grader_node", workflow_engine.hallucination_grader_node)

    # 2. Define Deterministic Edges
    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "grade_documents_node")
    workflow.add_edge("rewrite_query_node", "retrieve_node")
    workflow.add_edge("web_search_node", "generate_node")
    workflow.add_edge("generate_node", "hallucination_grader_node")

    # 3. Define Conditional Edges
    workflow.add_conditional_edges(
        "grade_documents_node",
        decide_to_generate_or_rewrite,
        {
            "generate_node": "generate_node",
            "rewrite_query_node": "rewrite_query_node",
            "web_search_node": "web_search_node",
        },
    )

    workflow.add_conditional_edges(
        "hallucination_grader_node",
        decide_to_finalize_or_regenerate,
        {
            END: END,
            "generate_node": "generate_node",
        },
    )

    app = workflow.compile()
    logger.info("LangGraph CRAG State Machine compiled successfully with Heterogeneous Model Routing.")
    return app


# ---------------------------------------------------------------------------
# 5. Executable Demonstration Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=======================================================================")
    print("      LANGGRAPH CRAG AGENT WITH HETEROGENEOUS MODEL ROUTING            ")
    print("=======================================================================\n")

    router = ModelRouter()
    engine = CRAGWorkflowEngine(router=router)
    graph = create_crag_graph(engine)

    test_query = "What vector database and indexing strategy is utilized for local storage?"
    print(f"Executing Query: '{test_query}'\n")

    initial_state: AgentState = {
        "question": test_query,
        "current_query": test_query,
        "messages": [HumanMessage(content=test_query)],
        "documents": [],
        "generation": "",
        "retrieval_retry_count": 0,
        "generation_retry_count": 0,
        "route_status": "init",
        "web_search_needed": False,
        "web_search_executed": False,
        "hallucination_grade": None,
        "hallucination_feedback": None,
        "citations": [],
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("FINAL GENERATED ANSWER:")
    print("=" * 70)
    print(final_state["generation"])
    print("\nCITATIONS:")
    for cit in final_state.get("citations", []):
        print(f" - {cit}")
    print("=" * 70 + "\n")
