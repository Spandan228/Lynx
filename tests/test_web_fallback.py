"""
CLI Test Script: Verifies Web Search Fallback Pathway & SSE Streaming
for Queries Not Present in Local Vector Store.

Author: Principal AI Agent Architect
"""

import time
import json
import warnings
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient

# Suppress minor warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from lynx.retriever import HybridRetriever, DocumentGrader
from lynx.web_search import web_search_client
from lynx.graph import create_crag_graph, CRAGWorkflowEngine, AgentState
from lynx.app import app

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def test_graph_web_fallback():
    print(f"\n{CYAN}{BOLD}[1/2] TESTING LANGGRAPH AUTOMATED WEB SEARCH FALLBACK PATHWAY{RESET}")
    print("Testing with external query not indexed in local documents...")

    import tempfile, shutil
    temp_dir = tempfile.mkdtemp(prefix="test_qdrant_web_")
    try:
        client = QdrantClient(path=temp_dir)
        retriever = HybridRetriever(
            qdrant_path=temp_dir,
            collection_name="agentic_rag_knowledge",
            client=client,
        )
        grader = DocumentGrader()
        engine = CRAGWorkflowEngine(
            retriever=retriever,
            grader=grader,
            web_search=web_search_client,
            max_retrieval_retries=2,
            max_generation_retries=2,
        )
        graph = create_crag_graph(engine)

        external_query = "What is the primary scientific objective of the James Webb Space Telescope?"

        test_state: AgentState = {
            "question": external_query,
            "current_query": external_query,
            "messages": [HumanMessage(content=external_query)],
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
        final_state = graph.invoke(test_state)
        elapsed = time.time() - start_time

        print(f"\n{BOLD}Execution Result Summary:{RESET}")
        print(f"  - Web Search Executed : {GREEN if final_state.get('web_search_executed') else YELLOW}{final_state.get('web_search_executed')}{RESET}")
        print(f"  - Retrieval Retries   : {final_state.get('retrieval_retry_count')}/2")
        print(f"  - Total Citations     : {len(final_state.get('citations', []))}")
        print(f"  - Grounding Grade     : {final_state.get('hallucination_grade')}")
        print(f"  - Latency             : {elapsed:.2f}s")
        print(f"\n{BOLD}Synthesized Answer Preview:{RESET}")
        print(final_state.get("generation", "")[:350] + "...")
        print(f"\n{BOLD}Citations Gathered:{RESET}")
        for c in final_state.get("citations", []):
            print(f"  * {c}")

        assert final_state.get("web_search_executed") is True, "Web search fallback was not triggered!"
        assert len(final_state.get("citations", [])) > 0, "No citations generated from web results!"
        print(f"\n{GREEN}{BOLD}>>> LANGGRAPH WEB SEARCH FALLBACK VERIFIED SUCCESSFULLY <<<{RESET}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fastapi_sse_streaming():
    print(f"\n{CYAN}{BOLD}[2/2] TESTING FASTAPI SSE STREAMING (/stream_query){RESET}")

    with TestClient(app) as client:
        response = client.post(
            "/stream_query",
            json={"query": "Explain how hybrid vector search combines dense and sparse embeddings."},
        )
        assert response.status_code == 200, f"Streaming failed with status {response.status_code}"

        lines = response.text.strip().split("\n")
        events_received = 0
        token_count = 0

        for line in lines:
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                    events_received += 1
                    if payload.get("event") == "token":
                        token_count += 1
                except Exception:
                    pass

        print(f"  - Total SSE Events Received : {events_received}")
        print(f"  - Streamed Token Chunks     : {token_count}")
        assert events_received > 5, "Too few SSE events received!"
        print(f"\n{GREEN}{BOLD}>>> FASTAPI SSE STREAMING VERIFIED SUCCESSFULLY <<<{RESET}\n")


if __name__ == "__main__":
    test_graph_web_fallback()
    test_fastapi_sse_streaming()

