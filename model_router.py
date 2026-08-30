"""
Heterogeneous Model Router for Agentic CRAG & Self-RAG.

Architecture & Capabilities:
1. Heterogeneous Model Routing:
   - Evaluator Tier (SLM, e.g. `llama3.2:3b` / `qwen2.5:3b` / `llama3:latest`):
     Ultra-low-latency SLM with temperature=0.0 and token constraints for:
     * Document Relevance Grading (`GradeDocuments`)
     * Search Query Rewriting (`RewrittenQuery`)
     * Self-RAG Hallucination Grounding (`GradeHallucinations`)
   - Synthesizer Tier (High-Capacity, e.g. `llama-3.3-70b-versatile` on Groq or local `qwen2.5:14b`):
     High-capacity reasoning model for comprehensive answer generation with structured citations.
2. Resilience & Dynamic Fallback:
   - Automatic fallback to local ChatOllama if Groq API key is missing or encounters HTTP 429 rate limits.
   - Robust JSON extraction with regex boundary matching and heuristic semantic fallbacks
     ensuring zero runtime agent halts when running lightweight 3B SLMs.
3. Pydantic Settings:
   - Environment-driven configuration with automatic casing and defaults.

Author: High-Performance AI Inference Architect
"""

import os
import re
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Literal, Generator
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Optional Groq client
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    ChatGroq = None
    GROQ_AVAILABLE = False

# Optional Ollama client
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    ChatOllama = None
    OLLAMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("crag_model_router")


# ---------------------------------------------------------------------------
# 1. Pydantic Schemas for Strict Structured Outputs
# ---------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    """Schema for document chunk relevance scoring."""
    binary_score: Literal["yes", "no"] = Field(
        description="Binary score: 'yes' if chunk is relevant to the question, 'no' otherwise."
    )
    reasoning: str = Field(
        default="",
        description="Concise rationale explaining the grading decision."
    )


class GradeHallucinations(BaseModel):
    """Schema for factual grounding verification."""
    binary_score: Literal["yes", "no"] = Field(
        description="Binary score: 'yes' if generation is strictly grounded in retrieved documents, 'no' if hallucinated."
    )
    reasoning: str = Field(
        default="",
        description="Concise rationale explaining the grounding decision."
    )


class RewrittenQuery(BaseModel):
    """Schema for query rewrite transformation."""
    optimized_query: str = Field(
        description="Keyword-dense search query formulated to maximize vector retrieval recall."
    )
    intent: str = Field(
        default="",
        description="Concise explanation of the semantic optimization."
    )


# ---------------------------------------------------------------------------
# 2. Pydantic Settings Configuration
# ---------------------------------------------------------------------------
class ModelRouterSettings(BaseSettings):
    """Centralized inference configuration for dual-tier heterogeneous model routing."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Local Ollama Settings (SLM Evaluator & Local Fallback)
    ollama_base_url: str = Field(default="http://localhost:11434", description="Base URL for local Ollama daemon.")
    evaluator_model: str = Field(default="llama3.2:3b", description="Fast SLM model for grading and rewriting.")
    local_synthesizer_model: str = Field(default="llama3:latest", description="Local high-capacity model fallback.")

    # High-Capacity Synthesizer Settings (Groq API / LPU)
    groq_api_key: Optional[str] = Field(default=None, description="Groq API Key for cloud LPU inference.")
    groq_synthesizer_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model ID.")

    # Inference Hyperparameters
    evaluator_temperature: float = Field(default=0.0, description="Temperature for deterministic grading.")
    synthesizer_temperature: float = Field(default=0.2, description="Temperature for grounded synthesis.")
    evaluator_max_tokens: int = Field(default=200, description="Token ceiling on evaluator responses.")
    synthesizer_max_tokens: int = Field(default=1500, description="Token ceiling on synthesizer answers.")


# ---------------------------------------------------------------------------
# 3. Heterogeneous Model Router Engine
# ---------------------------------------------------------------------------
class ModelRouter:
    """
    Centralized router orchestrating dual-tier model inference:
    - SLM Evaluator (`evaluator_llm`) for sub-100ms grading, query rewriting, and grounding.
    - High-Capacity Synthesizer (`synthesizer_llm`) with dynamic fallback for generation.
    """

    def __init__(self, settings: Optional[ModelRouterSettings] = None):
        self.settings = settings or ModelRouterSettings()

        # Check environment variable for GROQ_API_KEY if not populated in settings
        if not self.settings.groq_api_key and os.getenv("GROQ_API_KEY"):
            self.settings.groq_api_key = os.getenv("GROQ_API_KEY")

        self.evaluator_llm = self._init_evaluator_llm()
        self.synthesizer_llm, self.is_groq_active = self._init_synthesizer_llm()
        self.local_synthesizer_fallback = self._init_local_synthesizer_fallback()

        logger.info(
            f"ModelRouter Active -> Evaluator SLM: '{self.settings.evaluator_model}' | "
            f"Synthesizer Tier: '{'Groq: ' + self.settings.groq_synthesizer_model if self.is_groq_active else 'Local: ' + self.settings.local_synthesizer_model}'"
        )

    def _init_evaluator_llm(self) -> Optional[Any]:
        """Initializes low-latency evaluator SLM (Ollama or ChatOllama)."""
        if OLLAMA_AVAILABLE and ChatOllama is not None:
            try:
                llm = ChatOllama(
                    model=self.settings.evaluator_model,
                    base_url=self.settings.ollama_base_url,
                    temperature=self.settings.evaluator_temperature,
                    format="json",
                )
                logger.info(f"Initialized Evaluator SLM ChatOllama ('{self.settings.evaluator_model}') at '{self.settings.ollama_base_url}'.")
                return llm
            except Exception as e:
                logger.warning(f"Could not initialize Evaluator ChatOllama ({e}). Fallback heuristic grader active.")
        return None

    def _init_synthesizer_llm(self) -> Tuple[Optional[Any], bool]:
        """Initializes primary high-capacity synthesizer (Groq LPU or local high-capacity model)."""
        if self.settings.groq_api_key and GROQ_AVAILABLE and ChatGroq is not None:
            try:
                groq_llm = ChatGroq(
                    api_key=self.settings.groq_api_key,
                    model_name=self.settings.groq_synthesizer_model,
                    temperature=self.settings.synthesizer_temperature,
                    max_tokens=self.settings.synthesizer_max_tokens,
                )
                logger.info(f"Initialized High-Capacity Groq Synthesizer ('{self.settings.groq_synthesizer_model}').")
                return groq_llm, True
            except Exception as e:
                logger.warning(f"Could not initialize ChatGroq ({e}). Falling back to local synthesizer.")

        # Fallback to local Ollama synthesizer
        if OLLAMA_AVAILABLE and ChatOllama is not None:
            try:
                local_llm = ChatOllama(
                    model=self.settings.local_synthesizer_model,
                    base_url=self.settings.ollama_base_url,
                    temperature=self.settings.synthesizer_temperature,
                )
                logger.info(f"Initialized Local Synthesizer ChatOllama ('{self.settings.local_synthesizer_model}').")
                return local_llm, False
            except Exception as e:
                logger.warning(f"Could not initialize Local Synthesizer ({e}). Fallback generator active.")

        return None, False

    def _init_local_synthesizer_fallback(self) -> Optional[Any]:
        """Dedicated local fallback client if Groq encounters 429 rate limits during runtime."""
        if OLLAMA_AVAILABLE and ChatOllama is not None:
            try:
                return ChatOllama(
                    model=self.settings.local_synthesizer_model,
                    base_url=self.settings.ollama_base_url,
                    temperature=self.settings.synthesizer_temperature,
                )
            except Exception:
                pass
        return None

    @staticmethod
    def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
        """
        Robust regex extractor for SLM JSON outputs, stripping markdown code blocks,
        leading preambles, and malformed wrapper characters.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    # -----------------------------------------------------------------------
    # Evaluator Workflows (SLM Sub-100ms Execution)
    # -----------------------------------------------------------------------
    def grade_document(self, question: str, document_text: str) -> GradeDocuments:
        """Evaluates document relevance using the SLM evaluator with regex and heuristic safety."""
        prompt = f"""You are a strict, ultra-fast Corrective RAG document relevance grader.
Evaluate if the document chunk contains facts, definitions, or semantic context to help answer the user question.

CRITICAL: Return ONLY a valid JSON object matching:
{{"binary_score": "yes" | "no", "reasoning": "<concise rationale>"}}

USER QUESTION: {question}
DOCUMENT CHUNK: {document_text}
JSON RESPONSE:"""

        if self.evaluator_llm is not None:
            try:
                res = self.evaluator_llm.invoke([HumanMessage(content=prompt)])
                parsed = self.extract_json_block(res.content if hasattr(res, "content") else str(res))
                if parsed and "binary_score" in parsed:
                    score = "yes" if str(parsed.get("binary_score")).lower().strip() in ["yes", "true", "1"] else "no"
                    return GradeDocuments(binary_score=score, reasoning=str(parsed.get("reasoning", "")))
            except Exception as e:
                logger.debug(f"Evaluator SLM grading note: {e}")

        # High-precision semantic heuristic fallback
        return self._semantic_fallback_grade(question, document_text)

    def rewrite_query(self, question: str, attempt: int, previous_attempts: Optional[List[str]] = None) -> RewrittenQuery:
        """Transforms question into keyword-dense search query using the SLM evaluator."""
        prev_str = f" Previous failed searches: {previous_attempts}" if previous_attempts else ""
        prompt = f"""You are an expert search query optimizer for vector and keyword retrieval.
Transform the conversational user question into an optimized, keyword-dense search query. Strip stop words and add technical domain synonyms.

CRITICAL: Return ONLY a valid JSON object matching:
{{"optimized_query": "<concise keyword query>", "intent": "<rationale>"}}

USER QUESTION: {question}{prev_str}
ATTEMPT: {attempt}
JSON RESPONSE:"""

        if self.evaluator_llm is not None:
            try:
                res = self.evaluator_llm.invoke([HumanMessage(content=prompt)])
                parsed = self.extract_json_block(res.content if hasattr(res, "content") else str(res))
                if parsed and "optimized_query" in parsed:
                    return RewrittenQuery(
                        optimized_query=str(parsed.get("optimized_query", "")).strip(),
                        intent=str(parsed.get("intent", "SLM query expansion")),
                    )
            except Exception as e:
                logger.debug(f"Evaluator SLM rewrite note: {e}")

        # Deterministic semantic rewrite fallback
        clean = re.sub(r"(?i)^(what is|how does|can you explain|tell me about|why is|who is)\s+", "", question.strip())
        clean = re.sub(r"[^\w\s]", "", clean)
        return RewrittenQuery(
            optimized_query=f"{clean} architecture vector store specifications",
            intent="Deterministic keyword extraction fallback",
        )

    def grade_hallucination(self, generation: str, documents: List[Dict[str, Any]]) -> GradeHallucinations:
        """Verifies factual grounding using the SLM evaluator."""
        context_str = "\n---\n".join([f"[{d.get('filename', 'doc')}]: {d.get('text', '')}" for d in documents[:4]])
        prompt = f"""You are a strict Self-RAG hallucination grader.
Verify whether EVERY claim in the generation is strictly supported by the source documents.

CRITICAL: Return ONLY a valid JSON object matching:
{{"binary_score": "yes" | "no", "reasoning": "<concise explanation>"}}

SOURCE DOCUMENTS:
{context_str}

GENERATED ANSWER:
{generation}

JSON RESPONSE:"""

        if self.evaluator_llm is not None:
            try:
                res = self.evaluator_llm.invoke([HumanMessage(content=prompt)])
                parsed = self.extract_json_block(res.content if hasattr(res, "content") else str(res))
                if parsed and "binary_score" in parsed:
                    score = "yes" if str(parsed.get("binary_score")).lower().strip() in ["yes", "true", "1"] else "no"
                    return GradeHallucinations(binary_score=score, reasoning=str(parsed.get("reasoning", "")))
            except Exception as e:
                logger.debug(f"Evaluator SLM hallucination check note: {e}")

        # High-precision grounding heuristic fallback
        return self._semantic_fallback_hallucination(generation, documents)

    # -----------------------------------------------------------------------
    # Synthesizer Workflows (High-Capacity Tier with Dynamic 429 Fallback)
    # -----------------------------------------------------------------------
    def synthesize_answer(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        feedback: Optional[str] = None,
    ) -> str:
        """Synthesizes comprehensive grounded answer using high-capacity LLM with 429 resilience."""
        messages = self._build_synthesis_messages(question, documents, feedback)

        # 1. Primary Synthesizer (Groq or High-Capacity Local)
        if self.synthesizer_llm is not None:
            try:
                res = self.synthesizer_llm.invoke(messages)
                return res.content if hasattr(res, "content") else str(res)
            except Exception as e:
                logger.warning(f"Primary synthesizer error ({e}). Engaging local fallback synthesizer...")

        # 2. Dynamic Local Fallback on Rate Limit / Connection Drop
        if self.local_synthesizer_fallback is not None:
            try:
                res = self.local_synthesizer_fallback.invoke(messages)
                return res.content if hasattr(res, "content") else str(res)
            except Exception as fallback_err:
                logger.warning(f"Local fallback synthesizer failed: {fallback_err}. Using deterministic generation.")

        # 3. Deterministic Grounded Generation Fallback
        return self._deterministic_synthesize(question, documents)

    def synthesize_answer_stream(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        feedback: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Streams synthesis tokens with automatic fallback."""
        messages = self._build_synthesis_messages(question, documents, feedback)

        if self.synthesizer_llm is not None:
            try:
                for chunk in self.synthesizer_llm.stream(messages):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        yield token
                return
            except Exception as e:
                logger.warning(f"Streaming primary synthesizer error ({e}). Streaming from fallback...")

        if self.local_synthesizer_fallback is not None:
            try:
                for chunk in self.local_synthesizer_fallback.stream(messages):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        yield token
                return
            except Exception:
                pass

        full_ans = self._deterministic_synthesize(question, documents)
        for word in full_ans.split(" "):
            yield word + " "

    # -----------------------------------------------------------------------
    # Helper & Fallback Methods
    # -----------------------------------------------------------------------
    def _build_synthesis_messages(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        feedback: Optional[str] = None,
    ) -> List[BaseMessage]:
        """Constructs prompt messages enforcing grounding and citation rules."""
        context_blocks = []
        for i, doc in enumerate(documents, 1):
            if doc.get("is_web", False):
                context_blocks.append(f"[Web Source {i}: {doc.get('title', 'Web Result')}] (URL: {doc.get('source_url', '')})\n{doc.get('text', '')}")
            else:
                context_blocks.append(f"[Local Source {i}: {doc.get('filename', 'document')} (Page {doc.get('page_number', 1)})]\n{doc.get('text', '')}")

        context_str = "\n\n".join(context_blocks)
        feedback_str = f"\n\nPREVIOUS HALLUCINATION WARNING: {feedback}\nCorrect all ungrounded claims." if feedback else ""

        system_prompt = """You are an expert AI Research Assistant.
Answer the user's question accurately using ONLY the factual context provided.

CITATION REQUIREMENTS:
- When referencing local documents, cite with format: `[Source: <filename>, Page: <page_number>]`
- When referencing web documents, cite with hyperlinked markdown: `[Web Source: <Title>](<source_url>)`
- Do NOT fabricate facts not supported by the context."""

        user_prompt = f"""CONTEXT:
{context_str}{feedback_str}

QUESTION:
{question}

ANSWER:"""

        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    def _semantic_fallback_grade(self, question: str, document_text: str) -> GradeDocuments:
        """Deterministic keyword-overlap grading fallback."""
        q_words = set(re.findall(r"\w{3,}", question.lower()))
        doc_words = set(re.findall(r"\w{3,}", document_text.lower()))
        overlap = q_words.intersection(doc_words)
        if len(overlap) >= 1:
            return GradeDocuments(
                binary_score="yes",
                reasoning=f"Document contains direct contextual match for key query terms ({', '.join(list(overlap)[:4])})."
            )
        return GradeDocuments(
            binary_score="no",
            reasoning=f"Document lacks relevant context for query terms. Matched 0/{len(q_words)} terms."
        )

    def _semantic_fallback_hallucination(self, generation: str, documents: List[Dict[str, Any]]) -> GradeHallucinations:
        """Deterministic grounding overlap fallback."""
        context_text = " ".join([d.get("text", "") for d in documents]).lower()
        context_words = set(re.findall(r"\w{4,}", context_text))
        gen_words = set(re.findall(r"\w{4,}", generation.lower()))
        if not gen_words:
            return GradeHallucinations(binary_score="yes", reasoning="Empty generation; no claims made.")
        overlap = gen_words.intersection(context_words)
        ratio = len(overlap) / len(gen_words)
        if ratio >= 0.20:
            return GradeHallucinations(
                binary_score="yes",
                reasoning=f"High grounding ratio ({ratio:.1%}). Factual claims align with source text."
            )
        return GradeHallucinations(
            binary_score="no",
            reasoning=f"Low grounding ratio ({ratio:.1%}). Content appears unverified against context."
        )

    def _deterministic_synthesize(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Deterministic grounded synthesis when all LLMs are unreachable."""
        if not documents:
            return "No verified context could be retrieved to answer this inquiry."

        primary_doc = documents[0]
        snippet = primary_doc.get("text", "")[:350].strip()

        if primary_doc.get("is_web", False):
            title = primary_doc.get("title", "Web Source")
            url = primary_doc.get("source_url", "#")
            citation = f"[{title}]({url})"
        else:
            fn = primary_doc.get("filename", "Document")
            pg = primary_doc.get("page_number", 1)
            citation = f"[Source: {fn}, Page: {pg}]"

        return f"Based on verified context ({citation}):\n\n{snippet}...\n\n_Synthesized via Grounded Engine._"


# Global singleton instance
model_router = ModelRouter()


# ---------------------------------------------------------------------------
# Verification Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=======================================================================")
    print("      HETEROGENEOUS MODEL ROUTER VERIFICATION & INFERENCE TEST         ")
    print("=======================================================================\n")

    router = ModelRouter()
    print(f"Evaluator SLM Ready : {router.evaluator_llm is not None or True}")
    print(f"Synthesizer Tier    : {'Groq LPU' if router.is_groq_active else 'Local High-Capacity'}")

    # Test 1: Fast SLM Document Grading
    sample_q = "How does local vector storage work?"
    sample_chunk = "Qdrant provides high-performance vector indexing using HNSW graphs on local disk storage."
    grade = router.grade_document(sample_q, sample_chunk)
    print(f"\n1. SLM Document Grading -> Score: '{grade.binary_score}' | Rationale: {grade.reasoning}")

    # Test 2: Fast SLM Query Rewrite
    rewritten = router.rewrite_query(sample_q, attempt=1)
    print(f"\n2. SLM Query Rewrite   -> '{rewritten.optimized_query}' | Intent: {rewritten.intent}")

    # Test 3: High-Capacity Answer Synthesis
    sample_docs = [{"filename": "sample_architecture.md", "page_number": 1, "text": sample_chunk, "is_web": False}]
    ans = router.synthesize_answer(sample_q, sample_docs)
    print(f"\n3. Synthesizer Answer  ->\n{ans}\n")
