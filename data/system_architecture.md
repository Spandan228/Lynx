# Vector Database Storage Strategy
Qdrant serves as the vector store offering HNSW indexing and payload filtering.
Payload indexing enables fast scalar filtering on metadata such as document hashes and categories.
Idempotent ingestion guarantees that documents are never indexed twice, saving compute resources.
