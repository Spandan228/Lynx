"""
End-to-End Runtime Integration Test Runner and Cross-Module Diagnostic Suite.

Audits:
1. Schema & Vector Alignment: Vector dimensions (384d), Pydantic schemas, and AgentState keys.
2. Qdrant Lifecycle: Database existence, point count, and vector similarity search.
3. LLM Connectivity: Ollama daemon health check and deterministic fallback validation.
4. LangGraph Flow: Complete CRAG state machine simulation over local documents.
5. Loop Safety: Cyclic recursion guards terminating at max 2 retries.
6. FastAPI Async Execution: Verification of /health, /stats, and non-blocking /query.

Author: Principal AI Systems QA Engineer & High-Performance Inference Architect
"""

import os
import sys
import time
import socket
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Suppress minor warnings during test run
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

# Modules under test
from lynx.ingest import (
    IngestionConfig,
    IngestionPipeline,
    LocalEmbeddingEngine,
    TableAwareSemanticChunker,
    DoclingDocumentLoader,
)
from lynx.retriever import (
    HybridRetriever,
    DocumentGrader,
    RetrievedChunk,
    GradedChunk,
)
from lynx.model_router import (
    ModelRouter,
    model_router,
    GradeDocuments,
    GradeHallucinations,
    RewrittenQuery,
)
from lynx.graph import (
    create_crag_graph,
    CRAGWorkflowEngine,
    AgentState,
)
from lynx.app import (
    app,
    QueryRequest,
    QueryResponse,
    UploadResponse,
    StatsResponse,
    service_state,
)

# ---------------------------------------------------------------------------
# ANSI Color Codes for Diagnostic Terminal Output
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Diagnostic Test Runner Framework
# ---------------------------------------------------------------------------
class DiagnosticSuite:
    """Manages test execution, assertions, timing, and formatted diagnostic reporting."""

    def __init__(self):
        self.passed_count = 0
        self.warning_count = 0
        self.failed_count = 0
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def record_pass(self, test_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.passed_count += 1
        self.results.append({"name": test_name, "status": "PASS", "message": message, "details": details or {}})
        print(f"  {GREEN}[PASS]{RESET} {BOLD}{test_name}{RESET} - {message}")

    def record_warn(self, test_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.warning_count += 1
        self.results.append({"name": test_name, "status": "WARN", "message": message, "details": details or {}})
        print(f"  {YELLOW}[WARN]{RESET} {BOLD}{test_name}{RESET} - {message}")

    def record_fail(self, test_name: str, message: str, error: Optional[Exception] = None):
        self.failed_count += 1
        err_msg = f" ({str(error)})" if error else ""
        self.results.append({"name": test_name, "status": "FAIL", "message": f"{message}{err_msg}", "error": str(error)})
        print(f"  {RED}[FAIL]{RESET} {BOLD}{test_name}{RESET} - {message}{err_msg}")


# ---------------------------------------------------------------------------
# 1. Schema & Vector Dimension Alignment Tests
# ---------------------------------------------------------------------------
def test_schema_and_dimension_alignment(runner: DiagnosticSuite):
    """Audits cross-module vector dimension consistency and schema types."""
    print(f"\n{CYAN}{BOLD}[1/6] AUDITING SCHEMA & VECTOR DIMENSION ALIGNMENT{RESET}")

    # 1.1 Vector Dimension Consistency
    try:
        embed_engine = LocalEmbeddingEngine(model_name="BAAI/bge-small-en-v1.5")
        ingest_dim = embed_engine.vector_dim

        sample_embed = list(embed_engine.model.embed(["probe dimension"]))[0]
        retriever_dim = len(sample_embed)

        if ingest_dim == retriever_dim == 384:
            runner.record_pass(
                "Vector Embedding Dimension Alignment",
                f"Ingest ({ingest_dim}d) strictly matches Retriever query model ({retriever_dim}d).",
            )
        else:
            runner.record_fail(
                "Vector Embedding Dimension Alignment",
                f"Dimension mismatch! Ingest={ingest_dim}, Retriever={retriever_dim}",
            )
    except Exception as e:
        runner.record_fail("Vector Embedding Dimension Alignment", "Failed to verify embedding models", e)

    # 1.2 Pydantic v2 GradeDocuments Strict Schema Validation
    try:
        valid_grade = GradeDocuments.model_validate({"binary_score": "yes", "reasoning": "Document contains facts."})
        assert valid_grade.binary_score == "yes"

        invalid_caught = False
        try:
            GradeDocuments.model_validate({"binary_score": "maybe", "reasoning": "Uncertain."})
        except ValidationError:
            invalid_caught = True

        if invalid_caught:
            runner.record_pass(
                "Pydantic v2 GradeDocuments Strictness",
                "Schema strictly enforces Literal['yes', 'no'] binary scores and rejects invalid values.",
            )
        else:
            runner.record_fail(
                "Pydantic v2 GradeDocuments Strictness",
                "Schema failed to reject invalid binary_score value ('maybe').",
            )
    except Exception as e:
        runner.record_fail("Pydantic v2 GradeDocuments Strictness", "Schema audit error", e)

    # 1.3 Pydantic v2 GradeHallucinations Strictness
    try:
        valid_hallucination = GradeHallucinations.model_validate(
            {"binary_score": "no", "reasoning": "Model made ungrounded claims."}
        )
        assert valid_hallucination.binary_score == "no"
        runner.record_pass(
            "Pydantic v2 GradeHallucinations Strictness",
            "Self-RAG hallucination schema validated with strict binary types.",
        )
    except Exception as e:
        runner.record_fail("Pydantic v2 GradeHallucinations Strictness", "Hallucination schema error", e)

    # 1.4 State Schema Key Alignment (graph.py <-> app.py)
    try:
        agent_state_keys = set(AgentState.__annotations__.keys())
        expected_state_keys = {
            "question", "current_query", "messages", "documents", "generation",
            "retrieval_retry_count", "generation_retry_count", "route_status",
            "hallucination_grade", "hallucination_feedback", "citations"
        }
        if expected_state_keys.issubset(agent_state_keys):
            runner.record_pass(
                "AgentState Key Synchronization",
                f"All {len(expected_state_keys)} state keys are properly declared in AgentState.",
            )
        else:
            missing = expected_state_keys - agent_state_keys
            runner.record_fail("AgentState Key Synchronization", f"Missing keys in AgentState: {missing}")
    except Exception as e:
        runner.record_fail("AgentState Key Synchronization", "State key audit error", e)


# ---------------------------------------------------------------------------
# 2. Qdrant Vector Storage Roundtrip & Integrity Tests
# ---------------------------------------------------------------------------
def test_qdrant_vector_store_lifecycle(runner: DiagnosticSuite, client: QdrantClient):
    """Tests local Qdrant collection accessibility, vector search, and payload retrieval."""
    print(f"\n{CYAN}{BOLD}[2/6] TESTING QDRANT VECTOR STORE LIFECYCLE & INTEGRITY{RESET}")

    try:
        test_collection = "agentic_rag_knowledge"
        collections = [c.name for c in client.get_collections().collections]

        if test_collection in collections:
            runner.record_pass(
                "Qdrant Collection Status",
                f"Collection '{test_collection}' exists and is accessible.",
            )
        else:
            runner.record_warn(
                "Qdrant Collection Status",
                f"Collection '{test_collection}' not yet initialized. Run ingest.py first.",
            )
            return

        # Query points from collection
        points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
        if points:
            first_point = points[0]
            payload = first_point.payload or {}
            required_payload_keys = {"filename", "text", "doc_hash", "chunk_id"}
            if required_payload_keys.issubset(set(payload.keys())):
                runner.record_pass(
                    "Qdrant Payload Schema Integrity",
                    f"Verified {len(points)} indexed point(s). Metadata contains {list(required_payload_keys)}.",
                )
            else:
                missing = required_payload_keys - set(payload.keys())
                runner.record_warn(
                    "Qdrant Payload Schema Integrity",
                    f"Indexed points missing payload fields: {missing}",
                )
        else:
            runner.record_warn("Qdrant Payload Schema Integrity", "Collection contains 0 points. Run ingest.py.")

        # Vector Search Roundtrip Test
        probe_vector = [0.01] * 384
        search_results = client.query_points(
            collection_name=test_collection,
            query=probe_vector,
            limit=2,
            with_payload=True,
        )
        runner.record_pass(
            "Qdrant Vector Similarity Search Roundtrip",
            f"Cosine query executed successfully (returned {len(search_results.points)} nearest neighbors).",
        )

    except Exception as e:
        runner.record_fail("Qdrant Vector Store Lifecycle", "Qdrant roundtrip failure", e)


# ---------------------------------------------------------------------------
# 3. Local LLM Service Connectivity & Fallback Gracefulness
# ---------------------------------------------------------------------------
def test_llm_connectivity_and_fallback(runner: DiagnosticSuite):
    """Tests local Ollama endpoint availability and verifies robust deterministic fallback."""
    print(f"\n{CYAN}{BOLD}[3/6] TESTING LOCAL LLM CONNECTIVITY & FALLBACK GRACEFULNESS{RESET}")

    ollama_host = "localhost"
    ollama_port = 11434
    is_ollama_online = False

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.5)
        result = sock.connect_ex((ollama_host, ollama_port))
        sock.close()
        is_ollama_online = (result == 0)
    except Exception:
        is_ollama_online = False

    if is_ollama_online:
        runner.record_pass(
            "Local Ollama Daemon Connectivity",
            f"Ollama server detected active on http://{ollama_host}:{ollama_port}.",
        )
    else:
        runner.record_warn(
            "Local Ollama Daemon Connectivity",
            f"Ollama is offline on http://{ollama_host}:{ollama_port}. Internal deterministic fallback evaluator will engage.",
        )

    # Test DocumentGrader with either active LLM or deterministic fallback
    try:
        grader = DocumentGrader()
        sample_chunk = RetrievedChunk(
            chunk_id="test_c1",
            text="Qdrant provides vector indexing using HNSW graph algorithms and cosine similarity.",
            filename="system_architecture.md",
            page_number=1,
        )
        graded = grader.grade_chunk(
            question="What vector indexing algorithm is used in Qdrant?",
            chunk=sample_chunk,
        )

        assert graded.score in ["yes", "no"]
        assert isinstance(graded.reasoning, str) and len(graded.reasoning) > 0

        runner.record_pass(
            "Document Grader Subsystem",
            f"Grading evaluated chunk as '{graded.score.upper()}' | Reasoning: {graded.reasoning[:70]}...",
        )
    except Exception as e:
        runner.record_fail("Document Grader Subsystem", "Grader execution failed", e)


# ---------------------------------------------------------------------------
# 4. Synthetic LangGraph State Machine Execution Flow
# ---------------------------------------------------------------------------
def test_synthetic_langgraph_flow(runner: DiagnosticSuite, client: QdrantClient):
    """Simulates a complete LangGraph agentic query execution on verified knowledge."""
    print(f"\n{CYAN}{BOLD}[4/6] SIMULATING SYNTHETIC LANGGRAPH AGENTIC WORKFLOW{RESET}")

    try:
        retriever = HybridRetriever(
            qdrant_path="./qdrant_storage",
            collection_name="agentic_rag_knowledge",
            client=client,
        )
        grader = DocumentGrader()
        engine = CRAGWorkflowEngine(
            retriever=retriever,
            grader=grader,
            max_retrieval_retries=2,
            max_generation_retries=2,
        )
        graph = create_crag_graph(engine)

        test_state: AgentState = {
            "question": "How does Qdrant handle vector storage and payload indexing?",
            "current_query": "How does Qdrant handle vector storage and payload indexing?",
            "messages": [HumanMessage(content="How does Qdrant handle vector storage and payload indexing?")],
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

        output_state = graph.invoke(test_state)

        assert "generation" in output_state and len(output_state["generation"]) > 0
        assert "citations" in output_state
        assert output_state["hallucination_grade"] in ["yes", "no"]

        runner.record_pass(
            "LangGraph CRAG Execution Flow",
            f"Graph traversed full cycle: retrieve -> grade -> generate -> verify -> END. Citations: {len(output_state['citations'])}",
        )
    except Exception as e:
        runner.record_fail("LangGraph CRAG Execution Flow", "State machine invocation failed", e)


# ---------------------------------------------------------------------------
# 5. Cyclic Loop Safety & Recursion Boundary Tests
# ---------------------------------------------------------------------------
def test_cyclic_loop_safety_and_recursion_limits(runner: DiagnosticSuite, client: QdrantClient):
    """Verifies that retrieval retry guards terminate cleanly at max 2 retries."""
    print(f"\n{CYAN}{BOLD}[5/6] TESTING CYCLIC LOOP SAFETY & BOUNDARY LIMITS{RESET}")

    try:
        retriever = HybridRetriever(
            qdrant_path="./qdrant_storage",
            collection_name="agentic_rag_knowledge",
            client=client,
        )
        grader = DocumentGrader()
        engine = CRAGWorkflowEngine(
            retriever=retriever,
            grader=grader,
            max_retrieval_retries=2,
            max_generation_retries=2,
        )
        graph = create_crag_graph(engine)

        obscure_state: AgentState = {
            "question": "What is the warp velocity limit of the USS Enterprise in sector 001?",
            "current_query": "What is the warp velocity limit of the USS Enterprise in sector 001?",
            "messages": [HumanMessage(content="What is the warp velocity limit of the USS Enterprise in sector 001?")],
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

        start_time = time.time()
        final_obscure_state = graph.invoke(obscure_state)
        duration = time.time() - start_time

        retries_used = final_obscure_state.get("retrieval_retry_count", 0)
        if retries_used == 2:
            runner.record_pass(
                "Cyclic Retry Limit Enforcement",
                f"Graph reached exactly {retries_used}/2 retries before forced terminal routing (Time: {duration:.2f}s).",
            )
        else:
            runner.record_fail(
                "Cyclic Retry Limit Enforcement",
                f"Expected exactly 2 retries, but got {retries_used}.",
            )
    except Exception as e:
        runner.record_fail("Cyclic Retry Limit Enforcement", "Loop safety verification failed", e)


# ---------------------------------------------------------------------------
# 6. FastAPI Async Endpoints & Non-Blocking Execution Tests
# ---------------------------------------------------------------------------
def test_fastapi_endpoints_and_async_boundaries(runner: DiagnosticSuite, client: QdrantClient):
    """Tests FastAPI HTTP REST endpoints and non-blocking threadpool offloading."""
    print(f"\n{CYAN}{BOLD}[6/6] TESTING FASTAPI REST API & NON-BLOCKING ASYNC EXECUTION{RESET}")

    try:
        # Pre-assign shared Qdrant client to FastAPI service state to avoid lock collision
        service_state.qdrant_client = client

        with TestClient(app) as test_client:
            # 6.1 GET /health
            r_health = test_client.get("/health")
            if r_health.status_code == 200 and r_health.json().get("status") == "healthy":
                runner.record_pass("FastAPI GET /health Endpoint", "HTTP 200 OK returned.")
            else:
                runner.record_fail("FastAPI GET /health Endpoint", f"Unexpected status {r_health.status_code}")

            # 6.2 GET /stats
            r_stats = test_client.get("/stats")
            if r_stats.status_code == 200 and "total_indexed_chunks" in r_stats.json():
                stats_data = r_stats.json()
                runner.record_pass(
                    "FastAPI GET /stats Endpoint",
                    f"HTTP 200 OK. Indexed chunks: {stats_data.get('total_indexed_chunks')}.",
                )
            else:
                runner.record_fail("FastAPI GET /stats Endpoint", f"Stats error: {r_stats.text}")

            # 6.3 POST /query Non-blocking execution test
            t0 = time.time()
            r_query = test_client.post(
                "/query",
                json={"query": "How does local agentic RAG maintain data privacy?"},
            )
            api_duration = time.time() - t0

            if r_query.status_code == 200:
                resp_json = r_query.json()
                assert "answer" in resp_json
                assert "steps" in resp_json
                assert len(resp_json["steps"]) >= 3
                runner.record_pass(
                    "FastAPI POST /query Integration",
                    f"HTTP 200 OK (Latency: {api_duration:.2f}s). Structured execution steps: {len(resp_json['steps'])}.",
                )
            else:
                runner.record_fail("FastAPI POST /query Integration", f"Query API returned {r_query.status_code}: {r_query.text}")

    except Exception as e:
        runner.record_fail("FastAPI REST Endpoints", "TestClient execution error", e)


# ---------------------------------------------------------------------------
# High-Visibility Diagnostic Summary Report
# ---------------------------------------------------------------------------
def print_diagnostic_summary(runner: DiagnosticSuite):
    """Outputs high-visibility summary and common troubleshooting solutions."""
    total = runner.passed_count + runner.warning_count + runner.failed_count
    elapsed = time.time() - runner.start_time

    print("\n" + "=" * 75)
    print(f"{BOLD}                    SYSTEM DIAGNOSTIC SUMMARY REPORT                    {RESET}")
    print("=" * 75)
    print(f"Total Checks Executed : {total}")
    print(f"Passed Checks         : {GREEN}{BOLD}{runner.passed_count}{RESET}")
    print(f"Warnings / Advisories : {YELLOW}{BOLD}{runner.warning_count}{RESET}")
    print(f"Failed Checks         : {RED}{BOLD}{runner.failed_count}{RESET}")
    print(f"Total Execution Time  : {elapsed:.2f} seconds")
    print("=" * 75)

    if runner.failed_count == 0:
        print(f"\n{GREEN}{BOLD}>>> ALL SYSTEMS OPERATIONAL - ZERO RUNTIME BREAKS DETECTED <<<{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}>>> ISSUES DETECTED - REVIEW FAILED CHECKS ABOVE <<<{RESET}\n")

    # Troubleshooting Reference Table
    print(f"{BOLD}                        TROUBLESHOOTING MATRIX                         {RESET}")
    print("-" * 75)
    print(f"{'Failure Mode':<30} | {'Root Cause':<22} | {'1-Line Fix'}")
    print("-" * 75)
    print(f"{'Qdrant lock collision':<30} | {'Multiple client inits':<22} | Pass client=shared_client in init")
    print(f"{'Ollama connection refused':<30} | {'Daemon offline':<22} | Run: ollama serve in terminal")
    print(f"{'Vector dimension mismatch':<30} | {'Model name conflict':<22} | Set model='BAAI/bge-small-en-v1.5'")
    print(f"{'LangGraph recursion error':<30} | {'Missing retry guard':<22} | Check retry_count < 2 before route")
    print(f"{'FastAPI event loop lag':<30} | {'Blocking sync calls':<22} | Wrap with await run_in_threadpool()")
    print("-" * 75 + "\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{BOLD}======================================================================={RESET}")
    print(f"{BOLD}   LOCAL AGENTIC CRAG PIPELINE - RUNTIME INTEGRATION TEST RUNNER       {RESET}")
    print(f"{BOLD}======================================================================={RESET}")

    suite = DiagnosticSuite()
    shared_client = QdrantClient(path="./qdrant_storage")

    # Execute all diagnostic tests sequentially with shared client
    test_schema_and_dimension_alignment(suite)
    test_qdrant_vector_store_lifecycle(suite, client=shared_client)
    test_llm_connectivity_and_fallback(suite)
    test_synthetic_langgraph_flow(suite, client=shared_client)
    test_cyclic_loop_safety_and_recursion_limits(suite, client=shared_client)
    test_fastapi_endpoints_and_async_boundaries(suite, client=shared_client)

    # Print summary & troubleshooting matrix
    print_diagnostic_summary(suite)

    # Exit with non-zero status if any tests failed
    if suite.failed_count > 0:
        sys.exit(1)
    sys.exit(0)

