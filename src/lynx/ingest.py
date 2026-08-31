"""
Enterprise-Grade Document Intelligence & Ingestion Pipeline using Docling & FastEmbed.

Architecture & Capabilities:
1. Document Intelligence Loader (`DoclingDocumentLoader`):
   - Primary: IBM Research `Docling` (`DocumentConverter`) parsing multi-column layouts,
     complex PDF/DOCX structures, and embedded tables into semantic Markdown.
   - Fallback: Graceful multi-tier fallback to `PyPDF` and encoding-resilient text readers.
2. Table-Aware Semantic Chunker (`TableAwareSemanticChunker`):
   - Atomic Table Slicing: Tabular structures are treated atomically (never cut mid-row).
   - Header Retention: Large tables split across chunks repeat the column headers to maintain
     semantic context during dense vectorization.
   - Target Size: 512 tokens with 50-token overlap.
3. Idempotent Vector Store Layer (`QdrantVectorStore`):
   - Local disk-backed Qdrant with `Distance.COSINE` metric.
   - Pre-computation SHA-256 binary document hashing to skip duplicate documents.
   - Deterministic UUIDv5 point IDs ensuring zero vector collisions upon re-indexing.
4. Local Embeddings (`LocalEmbeddingEngine`):
   - FastEmbed ONNX Runtime execution (`BAAI/bge-small-en-v1.5`, 384 dimensions).

Author: Staff Data Ingestion and Document Intelligence Engineer
"""

import os
import re
import sys
import json
import uuid
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Generator

# Suppress minor third-party deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import tiktoken
from pydantic import BaseModel, Field

# FastEmbed for high-performance ONNX embeddings
from fastembed import TextEmbedding

# Qdrant Client models
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

# Text splitting primitives
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional PyPDF fallback
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Optional Docling Document Converter
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DocumentConverter = None
    DOCLING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_ingest")


# ---------------------------------------------------------------------------
# Data Models & Schemas
# ---------------------------------------------------------------------------
class IngestionConfig(BaseModel):
    """Configuration parameters for document parsing, chunking, and storage."""
    data_dir: str = Field(default="./data", description="Path to directory with raw documents.")
    qdrant_path: str = Field(default="./qdrant_storage", description="Path to local Qdrant on-disk DB.")
    qdrant_url: Optional[str] = Field(default=None, description="Optional remote Qdrant server URL.")
    qdrant_api_key: Optional[str] = Field(default=None, description="API Key for remote Qdrant.")
    collection_name: str = Field(default="agentic_rag_knowledge", description="Qdrant collection name.")
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5", description="FastEmbed model ID.")
    chunk_size_tokens: int = Field(default=512, description="Target chunk size in tokens.")
    chunk_overlap_tokens: int = Field(default=50, description="Chunk overlap in tokens.")
    batch_size: int = Field(default=64, description="Embedding and upsert batch size.")
    default_tenant_id: str = Field(default="tenant_default", description="Default tenant ID for ingested documents.")
    default_owner_id: str = Field(default="system", description="Default owner user ID.")
    default_allowed_roles: List[str] = Field(default=["user", "admin"], description="Default allowed RBAC roles.")
    supported_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".pptx", ".md", ".txt", ".markdown"],
        description="Allowed file extensions for ingestion.",
    )


@dataclass
class DocumentMetadata:
    """Document-level metadata extracted during intelligence parsing."""
    filename: str
    file_path: str
    doc_hash: str
    file_size_bytes: int
    total_pages: int
    table_count: int
    headings: List[str] = field(default_factory=list)
    parser_used: str = "docling"


@dataclass
class DocumentChunk:
    """Standardized representation of a semantically indexed text/table chunk with RBAC."""
    chunk_id: str
    doc_hash: str
    filename: str
    file_path: str
    page_number: Optional[int]
    chunk_index: int
    text: str
    snippet: str
    token_count: int
    char_count: int
    tenant_id: str = "tenant_default"
    owner_id: str = "system"
    allowed_roles: List[str] = field(default_factory=lambda: ["user", "admin"])
    is_table: bool = False
    table_index: Optional[int] = None
    headings_hierarchy: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Serializes chunk attributes to Qdrant storage payload."""
        return {
            "chunk_id": self.chunk_id,
            "doc_hash": self.doc_hash,
            "filename": self.filename,
            "file_path": self.file_path,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "snippet": self.snippet,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "tenant_id": self.tenant_id,
            "owner_id": self.owner_id,
            "allowed_roles": self.allowed_roles,
            "is_table": self.is_table,
            "table_index": self.table_index,
            "headings": self.headings_hierarchy,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# 1. Document Intelligence Loader (Docling with Resilient Fallbacks)
# ---------------------------------------------------------------------------
class DoclingDocumentLoader:
    """
    Enterprise document parser using Docling for multi-column layouts and tables,
    with automatic fallbacks for legacy/scanned files.
    """

    def __init__(self, supported_extensions: List[str]):
        self.supported_extensions = [ext.lower() for ext in supported_extensions]
        self._docling_converter: Optional[Any] = None
        self._docling_checked: bool = False

    @property
    def converter(self) -> Optional[Any]:
        """Lazy-loaded Docling DocumentConverter instance to conserve memory at startup."""
        if not self._docling_checked:
            self._docling_checked = True
            if DOCLING_AVAILABLE and DocumentConverter is not None:
                try:
                    self._docling_converter = DocumentConverter()
                    logger.info("Docling DocumentConverter initialized on-demand.")
                except Exception as e:
                    logger.warning(f"Could not initialize Docling DocumentConverter: {e}. Falling back to standard loaders.")
                    self._docling_converter = None
        return self._docling_converter

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes deterministic SHA-256 hash of binary file content."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def parse_document(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        """
        Parses a document into clean Markdown representation and extracts document metadata.
        """
        file_size = file_path.stat().st_size
        doc_hash = self.compute_sha256(file_path)

        # 1. Primary: Try Docling Document Converter
        conv = self.converter
        if conv is not None and file_path.suffix.lower() in [".pdf", ".docx", ".pptx", ".md", ".txt"]:
            try:
                logger.info(f"Parsing '{file_path.name}' via Docling layout engine...")
                conv_res = conv.convert(str(file_path))
                doc = conv_res.document

                # Export to clean, table-preserved Markdown
                markdown_text = doc.export_to_markdown()

                # Extract table count and headings
                table_count = len(getattr(doc, "tables", []))
                headings = [
                    h.text.strip()
                    for h in getattr(doc, "texts", [])
                    if hasattr(h, "label") and "heading" in str(h.label).lower()
                ]

                # Estimate page count
                page_count = len(getattr(doc, "pages", [])) or 1

                meta = DocumentMetadata(
                    filename=file_path.name,
                    file_path=str(file_path),
                    doc_hash=doc_hash,
                    file_size_bytes=file_size,
                    total_pages=page_count,
                    table_count=table_count,
                    headings=headings[:10],
                    parser_used="docling",
                )
                logger.info(f"Docling successfully parsed '{file_path.name}' ({page_count} pages, {table_count} tables).")
                return markdown_text, meta

            except Exception as docling_err:
                logger.warning(f"Docling parsing encountered an issue on '{file_path.name}': {docling_err}. Engaging fallback parser.")

        # 2. Fallback: PyPDF / Text Loader
        return self._fallback_parse(file_path, doc_hash, file_size)

    def _fallback_parse(self, file_path: Path, doc_hash: str, file_size: int) -> Tuple[str, DocumentMetadata]:
        """Graceful fallback for PDFs and raw text files."""
        ext = file_path.suffix.lower()
        extracted_text = ""
        total_pages = 1

        if ext == ".pdf" and PdfReader is not None:
            try:
                reader = PdfReader(str(file_path))
                total_pages = len(reader.pages)
                page_texts = []
                for i, page in enumerate(reader.pages):
                    p_text = page.extract_text() or ""
                    page_texts.append(f"\n\n<!-- Page {i+1} -->\n\n{p_text}")
                extracted_text = "".join(page_texts).strip()
            except Exception as e:
                logger.error(f"PyPDF fallback failed for '{file_path.name}': {e}")
                extracted_text = ""
        else:
            # Multi-tier text decoding
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
            for enc in encodings:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        extracted_text = f.read()
                    break
                except UnicodeDecodeError:
                    continue

        # Extract markdown tables via regex heuristic
        table_matches = re.findall(r"(\|.*\|\n\|[-:| ]+\|\n(?:\|.*\|\n?)+)", extracted_text)
        table_count = len(table_matches)

        # Extract markdown headings
        headings = re.findall(r"^#{1,3}\s+(.+)$", extracted_text, re.MULTILINE)

        meta = DocumentMetadata(
            filename=file_path.name,
            file_path=str(file_path),
            doc_hash=doc_hash,
            file_size_bytes=file_size,
            total_pages=total_pages,
            table_count=table_count,
            headings=headings[:10],
            parser_used="fallback_reader",
        )
        return extracted_text, meta


# ---------------------------------------------------------------------------
# 2. Table-Aware Semantic Chunker
# ---------------------------------------------------------------------------
class TableAwareSemanticChunker:
    """
    Intelligent chunker that treats tables atomically and repeats headers
    when large tables require multi-chunk segmentation.
    """

    # Regex matching Markdown tables with header, divider, and rows
    TABLE_PATTERN = re.compile(
        r"((?:\|[^\n]+\|\r?\n)"           # Table Header Row
        r"(?:\|[\s\-:|]+\|\r?\n)"        # Header Separator Row
        r"(?:\|[^\n]+\|\r?\n?)+)",       # One or more Table Data Rows
        re.MULTILINE
    )

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        )

    def count_tokens(self, text: str) -> int:
        """Accurately calculates token count via tiktoken."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text.split()))

    def chunk_document(
        self,
        markdown_text: str,
        meta: DocumentMetadata,
        tenant_id: str = "tenant_default",
        owner_id: str = "system",
        allowed_roles: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """
        Segments document markdown into table-aware, semantic chunks with RBAC metadata.
        """
        if not markdown_text.strip():
            return []

        roles = allowed_roles or ["user", "admin"]
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        # Split document into alternating text and table segments
        segments = self._split_text_and_tables(markdown_text)
        current_headings: List[str] = list(meta.headings[:3])

        for segment_type, segment_content, table_idx in segments:
            if segment_type == "table":
                # Process Table Block Atomically
                table_chunks = self._chunk_table_atomically(
                    table_markdown=segment_content,
                    table_idx=table_idx,
                    meta=meta,
                    start_chunk_idx=global_chunk_idx,
                    headings=current_headings,
                    tenant_id=tenant_id,
                    owner_id=owner_id,
                    allowed_roles=roles,
                )
                chunks.extend(table_chunks)
                global_chunk_idx += len(table_chunks)
            else:
                # Update current active heading context if present
                found_headings = re.findall(r"^#{1,3}\s+(.+)$", segment_content, re.MULTILINE)
                if found_headings:
                    current_headings = found_headings[-2:]

                # Process Standard Text Block Semantically
                text_splits = self.text_splitter.split_text(segment_content)
                for split_text in text_splits:
                    clean_text = split_text.strip()
                    if not clean_text or len(clean_text) < 15:
                        continue

                    tok_cnt = self.count_tokens(clean_text)
                    chunk_id = f"{Path(meta.filename).stem}_c{global_chunk_idx}"
                    snippet = clean_text[:140] + ("..." if len(clean_text) > 140 else "")

                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        doc_hash=meta.doc_hash,
                        filename=meta.filename,
                        file_path=meta.file_path,
                        page_number=1,
                        chunk_index=global_chunk_idx,
                        text=clean_text,
                        snippet=snippet,
                        token_count=tok_cnt,
                        char_count=len(clean_text),
                        tenant_id=tenant_id,
                        owner_id=owner_id,
                        allowed_roles=roles,
                        is_table=False,
                        table_index=None,
                        headings_hierarchy=current_headings,
                    )
                    chunks.append(chunk)
                    global_chunk_idx += 1

        logger.info(f"Generated {len(chunks)} table-aware chunk(s) for '{meta.filename}' (Tenant: {tenant_id}, Roles: {roles})")
        return chunks

    def _split_text_and_tables(self, markdown_text: str) -> List[Tuple[str, str, Optional[int]]]:
        """
        Extracts table blocks as distinct structural units while preserving surrounding text.
        Returns list of (type, content, table_index).
        """
        segments: List[Tuple[str, str, Optional[int]]] = []
        last_end = 0
        table_idx = 1

        for match in self.TABLE_PATTERN.finditer(markdown_text):
            start, end = match.span()

            # Preceding text block
            preceding_text = markdown_text[last_end:start].strip()
            if preceding_text:
                segments.append(("text", preceding_text, None))

            # Table block
            table_content = match.group(0).strip()
            if table_content:
                segments.append(("table", table_content, table_idx))
                table_idx += 1

            last_end = end

        # Remaining trailing text block
        trailing_text = markdown_text[last_end:].strip()
        if trailing_text:
            segments.append(("text", trailing_text, None))

        return segments

    def _chunk_table_atomically(
        self,
        table_markdown: str,
        table_idx: Optional[int],
        meta: DocumentMetadata,
        start_chunk_idx: int,
        headings: List[str],
        tenant_id: str = "tenant_default",
        owner_id: str = "system",
        allowed_roles: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunks table atomically with RBAC metadata. If table exceeds chunk size limit, splits row-by-row
        while repeating the header and divider on each sub-chunk.
        """
        table_tokens = self.count_tokens(table_markdown)
        roles = allowed_roles or ["user", "admin"]

        # 1. Fits within target chunk size -> Single Atomic Chunk
        if table_tokens <= self.chunk_size:
            chunk_id = f"{Path(meta.filename).stem}_tbl{table_idx or 1}_c{start_chunk_idx}"
            snippet = f"[Table {table_idx or 1}] " + table_markdown[:120].replace("\n", " ") + "..."
            return [
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_hash=meta.doc_hash,
                    filename=meta.filename,
                    file_path=meta.file_path,
                    page_number=1,
                    chunk_index=start_chunk_idx,
                    text=table_markdown,
                    snippet=snippet,
                    token_count=table_tokens,
                    char_count=len(table_markdown),
                    tenant_id=tenant_id,
                    owner_id=owner_id,
                    allowed_roles=roles,
                    is_table=True,
                    table_index=table_idx,
                    headings_hierarchy=headings,
                )
            ]

        # 2. Large Table: Slices row-by-row with header replication
        lines = [line.strip() for line in table_markdown.splitlines() if line.strip()]
        if len(lines) < 3:
            # Degenerate table format; return as single chunk
            chunk_id = f"{Path(meta.filename).stem}_tbl{table_idx or 1}_c{start_chunk_idx}"
            return [
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_hash=meta.doc_hash,
                    filename=meta.filename,
                    file_path=meta.file_path,
                    page_number=1,
                    chunk_index=start_chunk_idx,
                    text=table_markdown,
                    snippet=table_markdown[:140] + "...",
                    token_count=table_tokens,
                    char_count=len(table_markdown),
                    tenant_id=tenant_id,
                    owner_id=owner_id,
                    allowed_roles=roles,
                    is_table=True,
                    table_index=table_idx,
                    headings_hierarchy=headings,
                )
            ]

        header_line = lines[0]
        divider_line = lines[1]
        data_rows = lines[2:]

        sub_chunks: List[DocumentChunk] = []
        current_rows: List[str] = []
        current_idx = start_chunk_idx

        for row in data_rows:
            test_table = "\n".join([header_line, divider_line] + current_rows + [row])
            if self.count_tokens(test_table) > self.chunk_size and current_rows:
                # Flush current table sub-chunk
                final_table_text = "\n".join([header_line, divider_line] + current_rows)
                c_id = f"{Path(meta.filename).stem}_tbl{table_idx or 1}_p{len(sub_chunks)+1}_c{current_idx}"
                tok_cnt = self.count_tokens(final_table_text)

                sub_chunks.append(
                    DocumentChunk(
                        chunk_id=c_id,
                        doc_hash=meta.doc_hash,
                        filename=meta.filename,
                        file_path=meta.file_path,
                        page_number=1,
                        chunk_index=current_idx,
                        text=final_table_text,
                        snippet=f"[Table {table_idx or 1} Part {len(sub_chunks)+1}] {header_line[:60]}...",
                        token_count=tok_cnt,
                        char_count=len(final_table_text),
                        tenant_id=tenant_id,
                        owner_id=owner_id,
                        allowed_roles=roles,
                        is_table=True,
                        table_index=table_idx,
                        headings_hierarchy=headings,
                    )
                )
                current_idx += 1
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            final_table_text = "\n".join([header_line, divider_line] + current_rows)
            c_id = f"{Path(meta.filename).stem}_tbl{table_idx or 1}_p{len(sub_chunks)+1}_c{current_idx}"
            tok_cnt = self.count_tokens(final_table_text)
            sub_chunks.append(
                DocumentChunk(
                    chunk_id=c_id,
                    doc_hash=meta.doc_hash,
                    filename=meta.filename,
                    file_path=meta.file_path,
                    page_number=1,
                    chunk_index=current_idx,
                    text=final_table_text,
                    snippet=f"[Table {table_idx or 1} Part {len(sub_chunks)+1}] {header_line[:60]}...",
                    token_count=tok_cnt,
                    char_count=len(final_table_text),
                    tenant_id=tenant_id,
                    owner_id=owner_id,
                    allowed_roles=roles,
                    is_table=True,
                    table_index=table_idx,
                    headings_hierarchy=headings,
                )
            )

        return sub_chunks


# ---------------------------------------------------------------------------
# 3. Local Embedding Service (FastEmbed ONNX)
# ---------------------------------------------------------------------------
class LocalEmbeddingEngine:
    """
    Lightweight, local embedding generation powered by FastEmbed & ONNX Runtime.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        logger.info(f"Initializing local FastEmbed model: '{model_name}'...")
        self.model = TextEmbedding(model_name=model_name)

        sample_vec = list(self.model.embed(["probe vector initialization"]))[0]
        self.vector_dim = len(sample_vec)
        logger.info(f"Embedding model ready. Vector dimension: {self.vector_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Computes dense vector representations for text strings."""
        if not texts:
            return []

        embeddings_gen = self.model.embed(texts, batch_size=batch_size)
        return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in embeddings_gen]


# ---------------------------------------------------------------------------
# 4. Qdrant Vector Storage Layer (Multi-Tenant & RBAC Indexed)
# ---------------------------------------------------------------------------
class QdrantVectorStore:
    """
    Manages Qdrant vector database operations: collection initialization,
    tenant-aware idempotency lookups, RBAC payload indexing, and batch upserting.
    """

    def __init__(
        self,
        collection_name: str,
        vector_dim: int,
        storage_path: Optional[str] = "./qdrant_storage",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.is_remote = bool(url)

        if client is not None:
            self.client = client
        elif self.is_remote:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            path = storage_path or "./qdrant_storage"
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Safely initializes collection and creates RBAC payload indexes."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' "
                f"(dim={self.vector_dim}, distance=Cosine)..."
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            )

        # Create payload keyword indexes for fast multi-tenant and role filtering
        for field_name in ["tenant_id", "allowed_roles", "owner_id", "doc_hash"]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug(f"Payload index note for {field_name}: {e}")

        logger.info(f"Collection '{self.collection_name}' verified with RBAC payload indexes.")

    def is_document_indexed(self, doc_hash: str, tenant_id: Optional[str] = None) -> bool:
        """Checks if points matching document SHA-256 hash and tenant_id are already stored."""
        try:
            must_conds = [FieldCondition(key="doc_hash", match=MatchValue(value=doc_hash))]
            if tenant_id:
                must_conds.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)))

            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(must=must_conds),
                exact=True,
            )
            return count_result.count > 0
        except Exception as e:
            logger.debug(f"Idempotency filter check note: {e}")
            return False

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 64,
    ) -> int:
        """
        Upserts chunk vectors and payload metadata into Qdrant using deterministic UUIDv5 IDs.
        """
        if not chunks or not embeddings:
            return 0

        assert len(chunks) == len(embeddings), "Mismatched chunks and embedding vectors!"
        total_points = len(chunks)
        points: List[PointStruct] = []

        for chunk, vector in zip(chunks, embeddings):
            # Deterministic point ID from doc hash and chunk ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.doc_hash}:{chunk.chunk_id}"))

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk.to_qdrant_payload(),
            )
            points.append(point)

        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        logger.info(f"Successfully upserted {total_points} point(s) into '{self.collection_name}'.")
        return total_points


# ---------------------------------------------------------------------------
# 5. Master Ingestion Pipeline Orchestrator
# ---------------------------------------------------------------------------
class IngestionPipeline:
    """
    Production-grade pipeline orchestrator executing Docling layout parsing,
    hash-based idempotency checks, table-aware chunking, embedding, and vector upsert.
    """

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        client: Optional[QdrantClient] = None,
        vector_store: Optional[QdrantVectorStore] = None,
    ):
        self.config = config or IngestionConfig()

        self.loader = DoclingDocumentLoader(self.config.supported_extensions)
        self.chunker = TableAwareSemanticChunker(
            chunk_size=self.config.chunk_size_tokens,
            chunk_overlap=self.config.chunk_overlap_tokens,
        )
        self.embedding_engine = LocalEmbeddingEngine(self.config.embedding_model_name)
        self.vector_store = vector_store or QdrantVectorStore(
            collection_name=self.config.collection_name,
            vector_dim=self.embedding_engine.vector_dim,
            storage_path=self.config.qdrant_path,
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            client=client,
        )

    def scan_directory(self) -> List[Path]:
        """Scans configured data directory for supported documents."""
        data_path = Path(self.config.data_dir)
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            return []

        files = []
        for ext in self.config.supported_extensions:
            files.extend(data_path.rglob(f"*{ext}"))
        return sorted(files)

    def process_file(
        self,
        file_path: Path,
        tenant_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        allowed_roles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Processes a single file through the intelligence parsing & ingestion pipeline with RBAC."""
        start_time = time.time()
        file_hash = DoclingDocumentLoader.compute_sha256(file_path)

        t_id = tenant_id or self.config.default_tenant_id
        o_id = owner_id or self.config.default_owner_id
        roles = allowed_roles or self.config.default_allowed_roles

        # 1. Idempotency check: Skip if already indexed within this tenant
        if self.vector_store.is_document_indexed(file_hash, tenant_id=t_id):
            logger.info(f"Skipping '{file_path.name}' (SHA256: {file_hash[:8]}...) - Already indexed for tenant '{t_id}'.")
            return {
                "filename": file_path.name,
                "status": "skipped",
                "reason": "already_indexed",
                "chunks_indexed": 0,
                "tenant_id": t_id,
                "allowed_roles": roles,
                "doc_hash": file_hash,
                "duration_seconds": round(time.time() - start_time, 3),
            }

        # 2. Parse Layout & Extract Tables via Docling
        markdown_text, meta = self.loader.parse_document(file_path)
        if not markdown_text.strip():
            logger.warning(f"No text extracted from '{file_path.name}'. Skipping.")
            return {
                "filename": file_path.name,
                "status": "failed",
                "reason": "empty_extraction",
                "chunks_indexed": 0,
                "tenant_id": t_id,
                "duration_seconds": round(time.time() - start_time, 3),
            }

        # 3. Table-Aware Semantic Chunking with RBAC
        chunks = self.chunker.chunk_document(
            markdown_text,
            meta,
            tenant_id=t_id,
            owner_id=o_id,
            allowed_roles=roles,
        )
        if not chunks:
            return {
                "filename": file_path.name,
                "status": "failed",
                "reason": "zero_chunks_produced",
                "chunks_indexed": 0,
                "tenant_id": t_id,
                "duration_seconds": round(time.time() - start_time, 3),
            }

        # 4. Dense Vector Embedding Computation
        texts_to_embed = [chunk.text for chunk in chunks]
        embeddings = self.embedding_engine.embed_texts(
            texts=texts_to_embed,
            batch_size=self.config.batch_size,
        )

        # 5. Batch Upsert to Qdrant
        points_upserted = self.vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            batch_size=self.config.batch_size,
        )

        elapsed = round(time.time() - start_time, 3)
        logger.info(f"Indexed '{file_path.name}' for tenant '{t_id}': {points_upserted} chunks in {elapsed}s.")

        return {
            "filename": file_path.name,
            "status": "indexed",
            "chunks_indexed": points_upserted,
            "tenant_id": t_id,
            "owner_id": o_id,
            "allowed_roles": roles,
            "tables_found": meta.table_count,
            "parser_used": meta.parser_used,
            "doc_hash": file_hash,
            "duration_seconds": elapsed,
        }

    def run(
        self,
        tenant_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        allowed_roles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Executes full scan and ingestion over the data directory with RBAC."""
        files = self.scan_directory()
        logger.info(f"Discovered {len(files)} document(s) in '{self.config.data_dir}'.")

        stats = {
            "total_files": len(files),
            "indexed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks_created": 0,
            "tenant_id": tenant_id or self.config.default_tenant_id,
            "details": [],
        }

        for file_path in files:
            res = self.process_file(
                file_path=file_path,
                tenant_id=tenant_id,
                owner_id=owner_id,
                allowed_roles=allowed_roles,
            )
            stats["details"].append(res)

            if res["status"] == "indexed":
                stats["indexed_files"] += 1
                stats["total_chunks_created"] += res["chunks_indexed"]
            elif res["status"] == "skipped":
                stats["skipped_files"] += 1
            else:
                stats["failed_files"] += 1

        logger.info(
            f"Ingestion complete for tenant '{stats['tenant_id']}': {stats['indexed_files']} indexed, "
            f"{stats['skipped_files']} skipped, {stats['failed_files']} failed. "
            f"Total chunks in batch: {stats['total_chunks_created']}."
        )
        return stats


# ---------------------------------------------------------------------------
# 6. Verification & Executable Demo Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=======================================================================")
    print("   DOCLING-POWERED TABLE-AWARE INGESTION PIPELINE VERIFICATION DEMO    ")
    print("=======================================================================\n")

    # 1. Create a sample enterprise technical & financial report with complex embedded tables
    test_data_dir = Path("./data")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    sample_doc_path = test_data_dir / "financial_q3_report.md"

    sample_doc_content = """# Q3 Infrastructure Financial & Performance Report

## Executive Summary
This document summarizes compute hardware allocation, vector retrieval latency benchmarks, and cost efficiency across distributed clusters.

## Vector Store Performance & Cost Matrix

| Cluster ID | Engine Type | HNSW M | Cosine Recall@10 | P99 Latency (ms) | Monthly Cost ($) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| US-East-1 | Qdrant Local | 16 | 0.982 | 4.2 ms | $120.00 |
| US-West-2 | Qdrant Local | 32 | 0.994 | 5.8 ms | $180.00 |
| EU-Central-1 | Qdrant Server | 16 | 0.980 | 12.4 ms | $340.00 |
| AP-South-1 | Qdrant Server | 32 | 0.991 | 14.1 ms | $390.00 |

## Ingestion Architecture & Security
Idempotent ingestion guarantees that documents are never indexed twice, saving compute resources and preventing vector database bloating.
"""

    with open(sample_doc_path, "w", encoding="utf-8") as f:
        f.write(sample_doc_content)
    print(f"Created sample enterprise report with embedded table at: {sample_doc_path}")

    # 2. Initialize and execute Ingestion Pipeline
    config = IngestionConfig(data_dir=str(test_data_dir), collection_name="agentic_rag_knowledge")
    pipeline = IngestionPipeline(config=config)

    # 3. Test Table-Aware Semantic Chunker on the sample document
    loader = DoclingDocumentLoader(config.supported_extensions)
    chunker = TableAwareSemanticChunker(chunk_size=config.chunk_size_tokens, chunk_overlap=config.chunk_overlap_tokens)

    markdown_text, meta = loader.parse_document(sample_doc_path)
    chunks = chunker.chunk_document(markdown_text, meta)

    print(f"\n--- Parsed Document Metadata ---")
    print(f"Filename     : {meta.filename}")
    print(f"Total Tables : {meta.table_count}")
    print(f"Parser Used  : {meta.parser_used}")
    print(f"Doc SHA-256  : {meta.doc_hash}")

    print(f"\n--- Generated Chunks Breakdown ({len(chunks)} Chunks) ---")
    for i, c in enumerate(chunks, 1):
        table_badge = " [TABULAR BLOCK]" if c.is_table else ""
        print(f"\n[Chunk {i}/{len(chunks)}]{table_badge} (Tokens: {c.token_count}, ID: {c.chunk_id}):")
        print("-" * 50)
        print(c.text)
        print("-" * 50)

    # 4. Run Full Pipeline Directory Scan
    print("\n--- Running Master Ingestion Pipeline ---")
    stats = pipeline.run()
    print("\nPipeline Summary Stats:")
    print(json.dumps(stats, indent=2))
