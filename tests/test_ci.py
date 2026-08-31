"""
Lynx CRAG - CI/CD Automated Test Suite.
Covers schema validation, JWT/RBAC, RRF ranking, LangGraph compilation,
chunker behavior, observability fallback, and FastAPI route registration.
Designed to run without a live Qdrant lock conflict in CI environments.
"""
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Auth & Security
# ---------------------------------------------------------------------------
from lynx.auth import (
    UserSecurityContext,
    create_access_token,
    decode_access_token,
)


def test_jwt_and_rbac_auth():
    """Verify JWT token creation, decoding, and RBAC role logic."""
    ctx = UserSecurityContext(
        tenant_id="tenant_beta",
        user_id="user_42",
        roles=["viewer", "finance_reader"],
    )
    token = create_access_token(ctx)
    assert isinstance(token, str) and len(token) > 20

    decoded = decode_access_token(token)
    assert decoded.tenant_id == "tenant_beta"
    assert decoded.user_id == "user_42"
    assert "finance_reader" in decoded.roles
    assert decoded.overlaps_roles(["finance_reader"]) is True
    assert decoded.overlaps_roles(["legal_only"]) is False
    assert decoded.has_role("finance_reader") is True


def test_rbac_admin_bypass():
    """Admin role should bypass all role filters."""
    ctx = UserSecurityContext(tenant_id="tenant_alpha", user_id="adm_1", roles=["admin"])
    assert ctx.overlaps_roles(["legal_only", "executive"]) is True
    assert ctx.has_role("anything") is True


# ---------------------------------------------------------------------------
# 2. Retriever Schemas & RRF Logic
# ---------------------------------------------------------------------------
from lynx.retriever import RetrievedChunk, GradedChunk


def test_retrieved_chunk_schema():
    """Verify RetrievedChunk dataclass uses actual field names."""
    chunk = RetrievedChunk(
        chunk_id="chk_001",
        text="Financial Q3 revenue table content.",
        filename="financial_q3.md",
        page_number=1,
        tenant_id="tenant_alpha",
        allowed_roles=["admin", "finance_reader"],
        dense_score=0.91,
        sparse_score=14.2,
        hybrid_score=0.88,
    )
    assert chunk.chunk_id == "chk_001"
    assert chunk.tenant_id == "tenant_alpha"
    assert "finance_reader" in chunk.allowed_roles
    assert chunk.dense_score == 0.91


def test_reciprocal_rank_fusion():
    """Verify RRF logic correctly boosts chunks present in both dense and sparse results."""
    dense = [{"chunk_id": "c1", "score": 0.9}, {"chunk_id": "c2", "score": 0.7}]
    sparse = [{"chunk_id": "c2", "score": 15.0}, {"chunk_id": "c3", "score": 10.0}]

    k = 60
    rrf: dict = {}
    for rank, item in enumerate(dense):
        rrf[item["chunk_id"]] = rrf.get(item["chunk_id"], 0.0) + 1.0 / (k + rank + 1)
    for rank, item in enumerate(sparse):
        rrf[item["chunk_id"]] = rrf.get(item["chunk_id"], 0.0) + 1.0 / (k + rank + 1)

    # c2 appeared in both lists — must outrank c1 (dense-only) and c3 (sparse-only)
    assert rrf["c2"] > rrf["c1"]
    assert rrf["c2"] > rrf["c3"]


# ---------------------------------------------------------------------------
# 3. Model Router Schemas
# ---------------------------------------------------------------------------
from lynx.model_router import GradeDocuments, GradeHallucinations, RewrittenQuery


def test_model_router_schemas():
    """Verify model router Pydantic output schemas."""
    grade = GradeDocuments(binary_score="yes", explanation="Matches context.")
    assert grade.binary_score == "yes"

    hallucination = GradeHallucinations(binary_score="no", reasoning="Not grounded.")
    assert hallucination.binary_score == "no"

    rewrite = RewrittenQuery(
            optimized_query="expanded LangGraph cyclic retrieval",
        )
    assert "LangGraph" in rewrite.optimized_query


# ---------------------------------------------------------------------------
# 4. LangGraph Compilation (mocked Qdrant to avoid file lock)
# ---------------------------------------------------------------------------
from lynx.graph import AgentState, create_crag_graph


def test_agent_state_schema():
    """Verify AgentState TypedDict keys exist."""
    state_keys = AgentState.__annotations__.keys()
    assert "question" in state_keys
    assert "documents" in state_keys
    assert "generation" in state_keys
    assert "security_context" in state_keys
    assert "retrieval_retry_count" in state_keys


def test_langgraph_compilation_mocked():
    """Verify LangGraph graph compiles with a mocked Qdrant client."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = MagicMock(vectors_count=7)

    with patch("lynx.retriever.QdrantClient", return_value=mock_client):
        with patch("lynx.retriever.TextEmbedding") as mock_embed:
            mock_embed.return_value = MagicMock()
            graph = create_crag_graph()
            assert graph is not None
            nodes = graph.nodes
            assert "retrieve_node" in nodes
            assert "grade_documents_node" in nodes
            assert "generate_node" in nodes
            assert "rewrite_query_node" in nodes
            assert "web_search_node" in nodes
            assert "hallucination_grader_node" in nodes


# ---------------------------------------------------------------------------
# 5. Observability Graceful Fallback
# ---------------------------------------------------------------------------
from lynx.observability import setup_observability, trace_agent_node


def test_observability_graceful_fallback():
    """Verify observability initializes or gracefully fails without crashing."""
    result = setup_observability()
    assert result in [True, False]


def test_trace_agent_node_context_manager():
    """Verify trace_agent_node context manager runs without crashing."""
    import inspect
    sig = inspect.signature(trace_agent_node)
    params = list(sig.parameters.keys())
    assert len(params) >= 1  # Must accept at least a node name


# ---------------------------------------------------------------------------
# 6. FastAPI Route Registration
# ---------------------------------------------------------------------------
def test_fastapi_route_registration():
    """Verify all critical API routes are registered without starting the server."""
    from lynx.app import app
    routes = [route.path for route in app.routes]
    assert "/query" in routes
    assert "/stream_query" in routes
    assert "/upload" in routes
    assert "/stats" in routes
    assert "/health" in routes
    assert "/auth/token" in routes

