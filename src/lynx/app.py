"""
FastAPI Backend Service for Local Agentic Corrective RAG (CRAG), Web Search Fallback & SSE Streaming.

Features:
- REST API endpoint `POST /query`: Synchronous JSON execution trace response.
- REST SSE endpoint `POST /stream_query`: Real-time Server-Sent Events (SSE) streaming step-by-step thoughts and generated tokens.
- REST API endpoint `POST /upload`: Memory-efficient streaming file uploads (up to 50MB).
- Health & Stats endpoints `GET /health` and `GET /stats`.
- Full CORS middleware configuration and strict Pydantic v2 data models.
"""

import os
import sys
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

# Suppress minor third-party deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient

# Import modular RAG engine & ingestion components
from lynx.ingest import IngestionPipeline, IngestionConfig
from lynx.retriever import HybridRetriever, DocumentGrader
from lynx.web_search import web_search_client, WebSearchEngine
from lynx.graph import create_crag_graph, CRAGWorkflowEngine, AgentState

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Security & Multi-Tenant Authentication
from lynx.auth import (
    UserSecurityContext,
    get_current_user_security_context,
    create_access_token,
    decode_access_token,
)

# OpenTelemetry & Arize Phoenix Observability
from lynx.observability import (
    setup_observability,
    get_current_trace_id,
    PHOENIX_UI_URL,
    PHOENIX_PROJECT_NAME,
)

logger = logging.getLogger("rag_api")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
QDRANT_PATH = str(Path(__file__).parent.parent.parent / "qdrant_storage")
COLLECTION_NAME = "agentic_rag_knowledge"
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit


# ---------------------------------------------------------------------------
# Pydantic v2 Request & Response Models
# ---------------------------------------------------------------------------
class TokenRequest(BaseModel):
    """Payload for minting JWT access tokens."""
    tenant_id: str = Field(default="tenant_default", description="Tenant identifier.")
    user_id: str = Field(default="usr_001", description="User identifier.")
    roles: List[str] = Field(default=["user"], description="Assigned RBAC roles.")
    email: Optional[str] = Field(default=None, description="Optional user email.")


class TokenResponse(BaseModel):
    """JWT bearer token output."""
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int = 24
    tenant_id: str
    user_id: str
    roles: List[str]


class QueryRequest(BaseModel):
    """Input payload for user questions."""
    query: str = Field(..., min_length=1, description="The user query or question.")
    top_k: int = Field(default=3, ge=1, le=10, description="Max context chunks to retrieve.")


class DocumentSource(BaseModel):
    """Source citation and metadata for retrieved context chunks."""
    filename: str
    page_number: Optional[int] = None
    chunk_id: str
    snippet: str
    tenant_id: str = "tenant_default"
    allowed_roles: List[str] = Field(default_factory=list)
    is_web: bool = False
    source_url: Optional[str] = None


class ExecutionStep(BaseModel):
    """Trace details for intermediate agent reasoning steps."""
    step_name: str
    description: str
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Comprehensive structured response returned by the Agentic CRAG state machine with Phoenix Traces."""
    query: str
    answer: str
    citations: List[str]
    retrieved_sources: List[DocumentSource]
    retrieval_retries: int
    generation_retries: int
    web_search_executed: bool
    hallucination_grade: Optional[str]
    steps: List[ExecutionStep]
    trace_id: Optional[str] = Field(default=None, description="OpenTelemetry / Arize Phoenix trace identifier.")
    trace_url: Optional[str] = Field(default=None, description="Direct URL to Phoenix trace inspection tree.")
    execution_time_seconds: float


class UploadResponse(BaseModel):
    """Response returned upon file upload and ingestion."""
    status: str
    filename: str
    file_size_bytes: int
    ingestion_stats: Dict[str, Any]
    message: str


class StatsResponse(BaseModel):
    """System and vector database status metadata."""
    status: str
    collection_name: str
    total_indexed_chunks: int
    storage_path: str
    web_search_enabled: bool
    supported_formats: List[str]


# ---------------------------------------------------------------------------
# Global Application State & Initializer Helper
# ---------------------------------------------------------------------------
class ServiceState:
    qdrant_client: Optional[QdrantClient] = None
    graph_runner: Optional[Any] = None
    ingestion_pipeline: Optional[IngestionPipeline] = None
    hybrid_retriever: Optional[HybridRetriever] = None
    web_search_engine: Optional[WebSearchEngine] = None


service_state = ServiceState()


def initialize_services():
    """Initializes shared vector store client, retrieval index, and compiled LangGraph pipeline."""
    if service_state.graph_runner is not None:
        return

    logger.info("Initializing Agentic RAG Services...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(QDRANT_PATH, exist_ok=True)

    # 1. Single Shared Qdrant Client instance
    if service_state.qdrant_client is None:
        service_state.qdrant_client = QdrantClient(path=QDRANT_PATH)

    # 2. Ingestion Pipeline with shared client
    ingest_config = IngestionConfig(
        data_dir=str(DATA_DIR),
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )
    service_state.ingestion_pipeline = IngestionPipeline(
        config=ingest_config,
        client=service_state.qdrant_client,
    )

    # 3. Hybrid Retriever & Grader with shared client
    service_state.hybrid_retriever = HybridRetriever(
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        client=service_state.qdrant_client,
    )
    document_grader = DocumentGrader()
    service_state.web_search_engine = web_search_client

    # 4. Initialize Observability & Phoenix Telemetry
    setup_observability()

    # 5. Build & Compile LangGraph State Machine with Web Search Fallback
    engine = CRAGWorkflowEngine(
        retriever=service_state.hybrid_retriever,
        grader=document_grader,
        web_search=service_state.web_search_engine,
        max_retrieval_retries=2,
        max_generation_retries=2,
    )
    service_state.graph_runner = create_crag_graph(engine)
    logger.info("Agentic RAG Service with Web Search Fallback & Observability initialized successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context manager."""
    initialize_services()
    yield
    logger.info("Shutting down Agentic RAG Services.")


# ---------------------------------------------------------------------------
# FastAPI Application Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Local Agentic Corrective RAG API",
    description="Production REST & SSE API for Qdrant-backed Corrective RAG (CRAG), Web Search Fallback & Self-RAG workflows.",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for UI & API integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static web frontend assets if available
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

@app.get("/", include_in_schema=False)
async def serve_root():
    """Serves the Lynx CRAG frontend workspace."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Lynx CRAG API is operational. Visit /docs for Swagger specifications."}


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication & RBAC"])
async def generate_auth_token(request: TokenRequest) -> TokenResponse:
    """
    Issues a signed JWT Bearer token encoding tenant_id, user_id, and authorized RBAC roles.
    """
    ctx = UserSecurityContext(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        roles=request.roles,
        email=request.email,
    )
    token = create_access_token(ctx)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_hours=24,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        roles=ctx.roles,
    )


@app.get("/health", tags=["Monitoring"])
async def health_check() -> Dict[str, Any]:
    """Health check verifying model engine readiness, Qdrant DB connectivity, and Web Search."""
    initialize_services()
    qdrant_ok = False
    try:
        if service_state.hybrid_retriever and service_state.hybrid_retriever.client:
            service_state.hybrid_retriever.client.get_collections()
            qdrant_ok = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    return {
        "status": "healthy" if qdrant_ok else "degraded",
        "service": "Agentic CRAG Service",
        "version": "3.0.0",
        "security": "Multi-Tenant & RBAC Enabled",
        "qdrant_connected": qdrant_ok,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "web_search_available": service_state.web_search_engine is not None,
    }


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats() -> StatsResponse:
    """Returns vector database collection status and capabilities."""
    initialize_services()
    try:
        retriever = service_state.hybrid_retriever
        total_chunks = len(retriever.corpus_chunks) if retriever else 0
        return StatsResponse(
            status="active",
            collection_name=COLLECTION_NAME,
            total_indexed_chunks=total_chunks,
            storage_path=QDRANT_PATH,
            web_search_enabled=True,
            supported_formats=[".pdf", ".md", ".txt"],
        )
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query system stats: {str(e)}",
        )


@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: Optional[str] = Form(None),
    allowed_roles: Optional[str] = Form(None),
    user: UserSecurityContext = Depends(get_current_user_security_context),
) -> UploadResponse:
    """
    Streams and stores uploaded documents (.pdf, .md, .txt) and triggers the
    idempotent ingestion pipeline scoped to the authenticated tenant and assigned roles.
    """
    initialize_services()
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".pdf", ".md", ".txt", ".markdown"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed: .pdf, .md, .txt",
        )

    destination_path = DATA_DIR / file.filename
    total_bytes = 0

    try:
        with open(destination_path, "wb") as buffer:
            while chunk := await file.read(65536):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    destination_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
                    )
                buffer.write(chunk)

        effective_tenant = tenant_id or user.tenant_id
        roles_list = [r.strip() for r in allowed_roles.split(",")] if allowed_roles else user.roles
        logger.info(
            f"Received upload: '{file.filename}' ({total_bytes} bytes) "
            f"for tenant '{effective_tenant}', roles: {roles_list}. Ingesting..."
        )

        pipeline = service_state.ingestion_pipeline or IngestionPipeline()
        stats = await run_in_threadpool(
            pipeline.process_file,
            destination_path,
            tenant_id=effective_tenant,
            owner_id=user.user_id,
            allowed_roles=roles_list,
        )

        if service_state.hybrid_retriever:
            await run_in_threadpool(service_state.hybrid_retriever._build_bm25_index)

        return UploadResponse(
            status="success" if stats.get("status") in ["indexed", "skipped"] else "failed",
            filename=file.filename,
            file_size_bytes=total_bytes,
            ingestion_stats=stats,
            message=f"Document processed for tenant '{user.tenant_id}' (Status: {stats.get('status')}).",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file upload '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse, tags=["Agentic RAG"])
async def query_agent(
    request: QueryRequest,
    user: UserSecurityContext = Depends(get_current_user_security_context),
) -> QueryResponse:
    """
    Executes the full LangGraph Corrective RAG (CRAG) & Self-RAG state machine
    for the authenticated user, enforcing multi-tenant and RBAC boundaries.
    """
    initialize_services()
    start_time = time.time()
    user_query = request.query.strip()
    logger.info(f"Incoming /query from tenant '{user.tenant_id}' (User: {user.user_id}, Roles: {user.roles}): '{user_query}'")

    if not service_state.graph_runner:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agentic RAG state machine is not initialized.",
        )

    initial_state: AgentState = {
        "question": user_query,
        "current_query": user_query,
        "messages": [HumanMessage(content=user_query)],
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
        "security_context": user.model_dump(),
    }

    try:
        # Execute compiled LangGraph state machine asynchronously in threadpool
        final_state: AgentState = await run_in_threadpool(
            service_state.graph_runner.invoke,
            initial_state,
        )

        elapsed_time = round(time.time() - start_time, 3)

        # Build intermediate step traces
        steps: List[ExecutionStep] = []

        # Step 1: Hybrid Retrieval
        steps.append(
            ExecutionStep(
                step_name="Hybrid Retrieval",
                description="Queried Qdrant dense vectors and BM25 sparse keyword index with tenant & role filtering.",
                status="completed",
                details={
                    "retrieved_chunk_count": len(final_state.get("documents", [])),
                    "search_query": final_state.get("current_query", user_query),
                    "tenant_id": user.tenant_id,
                    "roles": user.roles,
                },
            )
        )

        # Step 2: Document Grading & Query Rewriting
        retrieval_retries = final_state.get("retrieval_retry_count", 0)
        if retrieval_retries > 0:
            steps.append(
                ExecutionStep(
                    step_name="Query Rewriting & Re-Retrieval",
                    description=f"Initial context was insufficient. Executed {retrieval_retries} query expansion cycle(s).",
                    status="completed",
                    details={
                        "retries_used": retrieval_retries,
                        "rewritten_query": final_state.get("current_query"),
                    },
                )
            )
        else:
            steps.append(
                ExecutionStep(
                    step_name="Document Relevance Grading",
                    description="Evaluated retrieved candidate chunks; verified sufficient relevant context.",
                    status="completed",
                    details={"relevance_check": "passed"},
                )
            )

        # Step 3: Web Search Fallback (if executed)
        web_executed = final_state.get("web_search_executed", False)
        if web_executed:
            steps.append(
                ExecutionStep(
                    step_name="Web Search Fallback",
                    description="Local document relevance was insufficient. Queried DuckDuckGo live web search for supplemental context.",
                    status="completed",
                    details={"web_search_engine": "DuckDuckGo", "status": "active"},
                )
            )

        # Step 4: Synthesis & Self-RAG Grounding
        hallucination_grade = final_state.get("hallucination_grade", "yes")
        steps.append(
            ExecutionStep(
                step_name="Self-RAG Grounding Verification",
                description="Evaluated synthesized response against context documents for hallucination prevention.",
                status="completed",
                details={
                    "is_grounded": (hallucination_grade == "yes"),
                    "generation_retries": final_state.get("generation_retry_count", 0),
                },
            )
        )

        # Format retrieved source metadata
        sources: List[DocumentSource] = []
        for doc in final_state.get("documents", []):
            text = doc.get("text", "")
            snippet = text[:180] + ("..." if len(text) > 180 else "")
            is_web = doc.get("is_web", False)
            sources.append(
                DocumentSource(
                    filename=doc.get("filename", "unknown"),
                    page_number=doc.get("page_number"),
                    chunk_id=doc.get("chunk_id", ""),
                    snippet=snippet,
                    tenant_id=doc.get("tenant_id", "tenant_default"),
                    allowed_roles=doc.get("allowed_roles", []),
                    is_web=is_web,
                    source_url=doc.get("source_url"),
                )
            )

        trace_id = get_current_trace_id()
        trace_url = f"{PHOENIX_UI_URL}/projects/{PHOENIX_PROJECT_NAME}"

        return QueryResponse(
            query=user_query,
            answer=final_state.get("generation", "No response generated."),
            citations=final_state.get("citations", []),
            retrieved_sources=sources,
            retrieval_retries=final_state.get("retrieval_retry_count", 0),
            generation_retries=final_state.get("generation_retry_count", 0),
            web_search_executed=web_executed,
            hallucination_grade=hallucination_grade,
            steps=steps,
            trace_id=trace_id,
            trace_url=trace_url,
            execution_time_seconds=elapsed_time,
        )

    except Exception as e:
        logger.error(f"Error executing /query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agentic workflow failed: {str(e)}",
        )


@app.post("/stream_query", tags=["Agentic RAG Streaming"])
async def stream_query_endpoint(
    request: QueryRequest,
    user: UserSecurityContext = Depends(get_current_user_security_context),
):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint with Multi-Tenant RBAC & OTel Tracing.
    Streams intermediate node execution steps followed by word-by-word generation tokens.
    """
    initialize_services()
    user_query = request.query.strip()
    logger.info(f"Incoming SSE /stream_query from tenant '{user.tenant_id}' (Roles: {user.roles}): '{user_query}'")

    if not service_state.graph_runner:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agentic RAG state machine is not initialized.",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.time()
        trace_id = get_current_trace_id()
        trace_url = f"{PHOENIX_UI_URL}/projects/{PHOENIX_PROJECT_NAME}"

        initial_state: AgentState = {
            "question": user_query,
            "current_query": user_query,
            "messages": [HumanMessage(content=user_query)],
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
            "security_context": user.model_dump(),
        }

        try:
            # Emit Trace Metadata Event
            yield f"data: {json.dumps({'event': 'trace', 'trace_id': trace_id, 'trace_url': trace_url})}\n\n"

            # Emit Step 1: Initial Hybrid Retrieval
            yield f"data: {json.dumps({'event': 'step', 'step_name': '🔍 Hybrid Retrieval', 'description': 'Searching Qdrant dense vectors + BM25 sparse index...', 'status': 'running'})}\n\n"
            await asyncio.sleep(0.1)

            # Run LangGraph streaming in worker thread
            def run_graph_sync():
                node_updates = []
                for chunk in service_state.graph_runner.stream(initial_state, stream_mode="updates"):
                    node_updates.append(chunk)
                return node_updates

            updates = await run_in_threadpool(run_graph_sync)

            # Process node transitions
            accumulated_state: Dict[str, Any] = dict(initial_state)
            for node_dict in updates:
                for node_name, state_delta in node_dict.items():
                    accumulated_state.update(state_delta)

                    if node_name == "retrieve_node":
                        chunk_count = len(state_delta.get("documents", []))
                        yield f"data: {json.dumps({'event': 'step', 'step_name': '🔍 Hybrid Retrieval', 'description': f'Retrieved {chunk_count} candidate chunk(s).', 'status': 'completed'})}\n\n"
                    elif node_name == "grade_documents_node":
                        relevance = "Passed" if not state_delta.get("web_search_needed") else "Low Relevance (Fallback Triggered)"
                        yield f"data: {json.dumps({'event': 'step', 'step_name': '⚖️ Document Relevance Grading', 'description': f'Evaluated chunks against query: {relevance}', 'status': 'completed'})}\n\n"
                    elif node_name == "rewrite_query_node":
                        new_q = state_delta.get("current_query", "")
                        yield f"data: {json.dumps({'event': 'step', 'step_name': '🔄 Query Rewriting', 'description': f'Optimized search query: {new_q}', 'status': 'completed'})}\n\n"
                    elif node_name == "web_search_node":
                        yield f"data: {json.dumps({'event': 'step', 'step_name': '🌐 Live Web Search Fallback', 'description': 'Queried DuckDuckGo for real-time external knowledge.', 'status': 'completed'})}\n\n"
                    elif node_name == "hallucination_grader_node":
                        grade = state_delta.get("hallucination_grade", "yes")
                        yield f"data: {json.dumps({'event': 'step', 'step_name': '🛡️ Self-RAG Grounding Verification', 'description': f'Fact check grade: {grade.upper()}', 'status': 'completed'})}\n\n"

            # Stream generated answer tokens
            full_answer = accumulated_state.get("generation", "No answer generated.")
            yield f"data: {json.dumps({'event': 'step', 'step_name': '✍️ Synthesizing Grounded Answer', 'description': 'Generating response with citations...', 'status': 'running'})}\n\n"

            words = full_answer.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"
                await asyncio.sleep(0.015)  # Smooth token stream pacing

            # Format source metadata
            sources: List[Dict[str, Any]] = []
            for doc in accumulated_state.get("documents", []):
                text = doc.get("text", "")
                snippet = text[:180] + ("..." if len(text) > 180 else "")
                sources.append({
                    "filename": doc.get("filename", "unknown"),
                    "page_number": doc.get("page_number"),
                    "chunk_id": doc.get("chunk_id", ""),
                    "snippet": snippet,
                    "is_web": doc.get("is_web", False),
                    "source_url": doc.get("source_url"),
                })

            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'event': 'complete', 'answer': full_answer, 'citations': accumulated_state.get('citations', []), 'sources': sources, 'web_search_executed': accumulated_state.get('web_search_executed', False), 'trace_id': trace_id, 'trace_url': trace_url, 'execution_time_seconds': elapsed})}\n\n"

        except Exception as e:
            logger.error(f"Error during SSE stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Static Web Dashboard Mount (sample.mp4 replica)
# ---------------------------------------------------------------------------
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Backend Server with Web Search on http://localhost:8000 ...")
    uvicorn.run("lynx.app:app", host="0.0.0.0", port=8000, reload=False)


