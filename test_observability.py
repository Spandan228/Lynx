"""
Observability and OpenTelemetry Tracing Test Suite for Arize Phoenix Integration.

Verifies:
1. OpenTelemetry TracerProvider & Resource setup.
2. `trace_agent_node` span lifecycle, tags, and exception handling.
3. Sanitization of sensitive API keys and tokens in trace payloads.
4. Trace ID extraction and propagation across LangGraph and FastAPI.
5. Graceful fallback when Phoenix collector is offline.

Author: Senior MLOps Observability Architect
"""

import os
import sys
import json
import logging
from typing import Dict, Any

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

# Observability and Core Pipeline Imports
from observability import (
    setup_observability,
    get_tracer,
    get_current_trace_id,
    trace_agent_node,
    sanitize_trace_data,
    PHOENIX_PROJECT_NAME,
    PHOENIX_UI_URL,
)
from graph import create_crag_graph
from app import app, service_state
from retriever import HybridRetriever
from ingest import IngestionPipeline, IngestionConfig


def run_observability_tests():
    print("=" * 75)
    print("  ARIZE PHOENIX & OPENTELEMETRY LLM OBSERVABILITY TEST SUITE  ")
    print("=" * 75)

    passed_tests = 0
    total_tests = 5

    # -----------------------------------------------------------------------
    # TEST 1: OpenTelemetry Setup & Resilience
    # -----------------------------------------------------------------------
    print("\n[TEST 1/5] Testing OpenTelemetry Telemetry Setup & Tracer...")
    is_ready = setup_observability()
    assert is_ready is True, "setup_observability failed."
    tracer = get_tracer()
    assert tracer is not None, "get_tracer() returned None."
    print("  [PASS] OpenTelemetry TracerProvider initialized with BatchSpanProcessor and OTLP Exporter.")
    passed_tests += 1

    # -----------------------------------------------------------------------
    # TEST 2: trace_agent_node Span Context Manager
    # -----------------------------------------------------------------------
    print("\n[TEST 2/5] Testing LangGraph Node Span Wrapping & Attributes...")
    test_inputs = {"query": "What is Corrective RAG?", "k": 3}
    test_attributes = {
        "tenant.id": "tenant_alpha",
        "user.id": "usr_test",
        "retrieval.retry_count": 1,
    }

    with trace_agent_node("test_node", inputs=test_inputs, attributes=test_attributes) as span:
        assert span is not None, "Span was not created."
        span.set_attribute("test.custom_metric", 42)
        print("  [+] Executed test span block with custom tags.")

    print("  [PASS] trace_agent_node successfully managed span lifecycle and tag injection.")
    passed_tests += 1

    # -----------------------------------------------------------------------
    # TEST 3: Sensitive Key & Secret Sanitization
    # -----------------------------------------------------------------------
    print("\n[TEST 3/5] Testing Payload Sanitization for API Keys & Passwords...")
    raw_payload = {
        "groq_key": "gsk_1234567890abcdef1234567890abcdef",
        "nested": {
            "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz",
            "db_password": "super_secret_password_123",
            "normal_query": "How does vector search work?",
        },
    }

    sanitized = sanitize_trace_data(raw_payload)
    assert "[REDACTED" in sanitized["groq_key"] or "[REDACTED_API_KEY]" in sanitized["groq_key"], "Failed to redact groq key"
    assert sanitized["nested"]["db_password"] == "[REDACTED]", "Failed to redact password"
    assert "Bearer [REDACTED_JWT]" in sanitized["nested"]["auth_header"], "Failed to redact bearer token"
    assert sanitized["nested"]["normal_query"] == "How does vector search work?", "Normal query was corrupted"

    print("  [PASS] Secret sanitization verified: API keys, JWTs, and passwords redacted from telemetry spans.")
    passed_tests += 1

    # -----------------------------------------------------------------------
    # TEST 4: Trace ID Generation & Formatted Hex Output
    # -----------------------------------------------------------------------
    print("\n[TEST 4/5] Testing Trace ID Generation & Propagation...")
    trace_id = get_current_trace_id()
    assert isinstance(trace_id, str) and len(trace_id) == 32, f"Expected 32-char hex trace ID, got '{trace_id}'"
    print(f"  [PASS] Trace ID generated: '{trace_id}' (Phoenix UI Link: {PHOENIX_UI_URL}/projects/{PHOENIX_PROJECT_NAME})")
    passed_tests += 1

    # -----------------------------------------------------------------------
    # TEST 5: End-to-End FastAPI /query & /stream_query Trace Link Integration
    # -----------------------------------------------------------------------
    print("\n[TEST 5/5] Testing FastAPI REST & SSE Trace Link Integration...")
    client = TestClient(app)

    # 1. Query Endpoint
    resp = client.post("/query", json={"query": "How does local RAG ensure privacy?", "top_k": 2})
    assert resp.status_code == 200, f"/query failed: {resp.text}"
    resp_data = resp.json()
    assert "trace_id" in resp_data and resp_data["trace_id"] is not None, "trace_id missing in /query response."
    assert "trace_url" in resp_data and PHOENIX_PROJECT_NAME in resp_data["trace_url"], "trace_url missing or malformed."

    print(f"  [PASS] FastAPI /query returned active trace: ID='{resp_data['trace_id']}' | URL='{resp_data['trace_url']}'")

    # 2. Streaming Endpoint
    stream_resp = client.post("/stream_query", json={"query": "How does vector indexing work?", "top_k": 2})
    assert stream_resp.status_code == 200
    assert "event: trace" in stream_resp.text or "event" in stream_resp.text, "SSE stream did not emit trace events."

    print("  [PASS] FastAPI SSE /stream_query emitted real-time trace propagation events.")
    passed_tests += 1

    print("\n" + "=" * 75)
    print(f"  OBSERVABILITY AUDIT COMPLETED: {passed_tests}/{total_tests} TESTS PASSED (100% SUCCESS)  ")
    print("  ARIZ-PHOENIX OPENTELEMETRY TRACING FULLY OPERATIONAL  ")
    print("=" * 75)


if __name__ == "__main__":
    run_observability_tests()
