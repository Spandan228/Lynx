"""
Modular Web Search Tool & Query Optimizer for Agentic Corrective RAG (CRAG).

Features:
- DuckDuckGo (`ddgs` / `duckduckgo-search`) live search integration.
- HTML sanitization, entity unescaping, and snippet length normalization.
- Conversational query optimizer converting long questions into search engine keywords.
- Standardized chunk representation compatible with LangGraph state and local retrieval schemas.
- Safe rate-limit, timeout, and offline exception handling.
"""

import re
import html
import uuid
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Try importing DDGS from either ddgs or duckduckgo_search
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

logger = logging.getLogger("crag_web_search")

# Common conversational stop phrases to remove when optimizing queries for web search
CONVERSATIONAL_STOP_PATTERNS = [
    r"\b(can you please tell me|could you tell me|what is|what are|how do i|how does|please explain|tell me about|i want to know|where can i find|give me information on|is there any)\b",
    r"\b(in the context of|according to|based on|summarize|detailed explanation of)\b",
]

COMMON_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
}


class WebSearchResult(BaseModel):
    """Structured representation of a sanitized web search result."""
    title: str = Field(..., description="The title of the web article.")
    source_url: str = Field(..., description="The direct URL of the search result.")
    snippet: str = Field(..., description="Cleaned, sanitized textual snippet.")
    score: float = Field(default=0.85, description="Search relevance confidence.")


class WebSearchEngine:
    """
    Robust live web search client with query optimization and HTML snippet sanitization.
    """

    def __init__(self, max_results: int = 3, timeout_seconds: int = 8):
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def sanitize_snippet(raw_text: str, max_chars: int = 600) -> str:
        """
        Strips HTML tags, decodes XML/HTML entities, removes non-printable chars,
        and truncates snippet length to avoid feeding noisy HTML into LLM context.
        """
        if not raw_text:
            return ""

        # 1. Unescape HTML entities (&amp;, &#39;, &quot;, etc.)
        cleaned = html.unescape(raw_text)

        # 2. Strip HTML tags (<b>, <span>, <div>, etc.)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        # 3. Normalize whitespace and newlines
        cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        # 4. Remove unsupported control characters while preserving valid unicode
        cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch == " ")

        # 5. Truncate to maximum characters cleanly at word boundary
        if len(cleaned) > max_chars:
            cutoff = cleaned[:max_chars].rfind(" ")
            cleaned = cleaned[:cutoff] + "..." if cutoff > 0 else cleaned[:max_chars] + "..."

        return cleaned

    @staticmethod
    def optimize_query(conversational_query: str) -> str:
        """
        Converts conversational, verbose user prompts into tight search engine keywords.
        Example:
          'Can you please tell me what the warp velocity limit of the USS Enterprise is?'
          -> 'warp velocity limit USS Enterprise'
        """
        if not conversational_query:
            return ""

        query_text = conversational_query.strip()

        # 1. Strip common conversational prefix/suffix phrases
        for pattern in CONVERSATIONAL_STOP_PATTERNS:
            query_text = re.sub(pattern, " ", query_text, flags=re.IGNORECASE)

        # 2. Remove punctuation except hyphens/quotes
        clean_punct = re.sub(r"[^\w\s\-\"\']", " ", query_text)

        # 3. Extract meaningful keywords
        tokens = [tok for tok in clean_punct.split() if tok.lower() not in COMMON_STOPWORDS]

        # If keyword extraction filtered everything, fallback to original cleaned query
        if not tokens:
            return re.sub(r"\s+", " ", clean_punct).strip()

        return " ".join(tokens)

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes web search via DuckDuckGo and formats output into standard RAG chunks.
        Returns empty list with error logs if offline or rate-limited.
        """
        limit = max_results or self.max_results
        optimized_query = self.optimize_query(query)
        logger.info(f"Executing web search for: '{optimized_query}' (Original: '{query}')")

        if DDGS is None:
            logger.warning("duckduckgo_search / ddgs library not found. Returning empty web results.")
            return []

        retrieved_chunks: List[Dict[str, Any]] = []

        try:
            ddgs_client = DDGS(timeout=self.timeout_seconds)
            raw_results = list(ddgs_client.text(optimized_query, max_results=limit))

            if not raw_results:
                # Retry once with raw query if optimized query returned 0 hits
                logger.info(f"0 results for optimized query. Retrying with original: '{query}'")
                raw_results = list(ddgs_client.text(query, max_results=limit))

            for idx, res in enumerate(raw_results, start=1):
                raw_title = res.get("title", f"Web Result {idx}")
                raw_snippet = res.get("body") or res.get("snippet", "")
                raw_url = res.get("href") or res.get("link", "https://duckduckgo.com")

                clean_title = self.sanitize_snippet(raw_title, max_chars=120)
                clean_snippet = self.sanitize_snippet(raw_snippet, max_chars=600)

                if not clean_snippet:
                    continue

                chunk_text = f"**Title:** {clean_title}\n**URL:** {raw_url}\n**Snippet:** {clean_snippet}"
                chunk_id = f"web_{uuid.uuid4().hex[:8]}"

                chunk_dict = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "filename": f"[Web: {clean_title}]({raw_url})",
                    "source_url": raw_url,
                    "title": clean_title,
                    "page_number": 1,
                    "is_web": True,
                    "score": 0.88,
                    "metadata": {
                        "source": "web_search",
                        "url": raw_url,
                        "title": clean_title,
                    },
                }
                retrieved_chunks.append(chunk_dict)

            logger.info(f"Web search successfully retrieved {len(retrieved_chunks)} result chunk(s).")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Web search request failed: {e}", exc_info=False)
            return []


# Global singleton instance
web_search_client = WebSearchEngine()
