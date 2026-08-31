"""
Arize Phoenix & OpenTelemetry LLM Observability Instrumentation Layer.

Architecture:
1. Automated OpenTelemetry TracerProvider configuration with OTLP HTTP exporter.
2. Auto-instrumentation of LangChain & LangGraph via OpenInference.
3. Context manager `trace_agent_node` for detailed LangGraph execution span trees:
   - Node inputs and outputs
   - Custom tags: `tenant_id`, `user_id`, `retrieval_retry_count`, `generation_retry_count`
   - Real-time token latency and grounding metrics
4. Automatic payload sanitization to redact API keys and secrets.
5. High resilience: Continues normal execution if the Phoenix collector is offline.

Author: Senior MLOps Observability Architect
"""

import os
import re
import sys
import json
import logging
from typing import Dict, Any, Optional, Generator
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("crag_observability")

# Telemetry Global Configuration
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "agentic-crag-production")
PHOENIX_UI_URL = os.getenv("PHOENIX_UI_URL", "http://localhost:6006")
SERVICE_NAME = "agentic-crag-service"

# OpenTelemetry imports with graceful fallbacks
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
except ImportError:
    trace = None
    OTEL_AVAILABLE = False

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    LANGCHAIN_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    LangChainInstrumentor = None
    LANGCHAIN_INSTRUMENTATION_AVAILABLE = False


_is_instrumented = False
_tracer_provider: Optional[Any] = None


def setup_observability() -> bool:
    """
    Initializes OpenTelemetry tracing and configures the OTLP exporter
    pointing to Arize Phoenix. Safe against network drops or missing collectors.
    """
    global _is_instrumented, _tracer_provider

    if _is_instrumented:
        return True

    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry packages not found. LLM observability tracing disabled.")
        return False

    try:
        # Define Resource attributes for Arize Phoenix Project Grouping
        resource = Resource.create({
            "service.name": SERVICE_NAME,
            "project.name": PHOENIX_PROJECT_NAME,
            "environment": os.getenv("ENVIRONMENT", "production"),
        })

        _tracer_provider = TracerProvider(resource=resource)

        # Configure OTLP HTTP Exporter targeting Phoenix collector
        otlp_exporter = OTLPSpanExporter(
            endpoint=PHOENIX_COLLECTOR_ENDPOINT,
            timeout=5,
        )

        # Use BatchSpanProcessor for asynchronous non-blocking trace delivery
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=1000,
        )
        _tracer_provider.add_span_processor(span_processor)

        # Set Global Tracer Provider
        trace.set_tracer_provider(_tracer_provider)

        # Auto-instrument LangChain & LangGraph
        if LANGCHAIN_INSTRUMENTATION_AVAILABLE and LangChainInstrumentor is not None:
            LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
            logger.info("LangChain and LangGraph auto-instrumented with OpenInference.")

        _is_instrumented = True
        logger.info(
            f"Arize Phoenix Observability active -> Collector: '{PHOENIX_COLLECTOR_ENDPOINT}' "
            f"| Project: '{PHOENIX_PROJECT_NAME}' | UI: '{PHOENIX_UI_URL}'"
        )
        return True

    except Exception as e:
        logger.warning(
            f"Failed to initialize Arize Phoenix tracing ({e}). "
            "Pipeline will continue without remote tracing."
        )
        _is_instrumented = False
        return False


def get_tracer():
    """Returns the configured OpenTelemetry tracer."""
    if OTEL_AVAILABLE and trace is not None:
        return trace.get_tracer("agentic-crag-agent", "3.0.0")
    return None


def get_current_trace_id() -> str:
    """Returns the current active span's Trace ID formatted as a 32-character hex string."""
    if OTEL_AVAILABLE and trace is not None:
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, "032x")
    import uuid
    return uuid.uuid4().hex


def sanitize_trace_data(data: Any) -> Any:
    """
    Sanitizes trace payloads to prevent leaking secret keys (e.g. gsk_*, passwords).
    """
    if isinstance(data, str):
        # Redact Groq or secret API keys
        cleaned = re.sub(r"gsk_[a-zA-Z0-9_-]{20,}", "[REDACTED_API_KEY]", data)
        cleaned = re.sub(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", "Bearer [REDACTED_JWT]", cleaned, flags=re.IGNORECASE)
        return cleaned
    elif isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if any(secret_term in k.lower() for secret_term in ["secret", "password", "api_key", "token"]):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_trace_data(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_trace_data(item) for item in data]
    return data


@contextmanager
def trace_agent_node(
    node_name: str,
    inputs: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Optional[Any], None, None]:
    """
    Context manager that wraps LangGraph node executions into an OpenTelemetry span.
    Safe against remote collector failures.
    """
    tracer = get_tracer()
    if not tracer:
        yield None
        return

    span_name = f"langgraph.node.{node_name}"
    span = tracer.start_span(span_name)

    try:
        # Attach initial attributes and inputs
        span.set_attribute("langgraph.node", node_name)
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(str(k), str(v) if not isinstance(v, (int, float, bool)) else v)

        if inputs:
            sanitized_inputs = sanitize_trace_data(inputs)
            span.set_attribute("node.inputs", json.dumps(sanitized_inputs, default=str)[:2000])

        yield span

        span.set_status(Status(StatusCode.OK))

    except Exception as exc:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise exc
    finally:
        span.end()
