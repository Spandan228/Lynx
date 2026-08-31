# Agentic CRAG Evaluation & Benchmark Report

**Benchmark Execution Date:** 2026-08-30 12:44:32  
**Vector Store Engine:** Local Qdrant (HNSW + Cosine)  
**Embedding Model:** `BAAI/bge-small-en-v1.5` (384d)  
**Fallback Engine:** DuckDuckGo Live Search  

---

## 1. Executive Metrics Summary

| Evaluation Metric | Benchmark Result | Target Baseline | Status |
| :--- | :---: | :---: | :---: |
| **Web Fallback Trigger Accuracy** | **83.3%** | $\ge 85\%$ | 🟡 ACCEPTABLE |
| **Faithfulness / Grounding Rate** | **97.5%** | $\ge 80\%$ | 🟢 PASS |
| **Answer Concept Relevance** | **83.3%** | $\ge 75\%$ | 🟢 PASS |
| **Context Precision Ratio** | **100.0%** | $\ge 60\%$ | 🟢 PASS |
| **Average Query Latency** | **9.20s** | $\le 10.0s$ | 🟢 PASS |

---

## 2. Granular Query-Level Test Results

| ID | Domain | Query | Fallback Triggered | Precision | Faithfulness | Relevance | Citations | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ID-001` | `in_domain` | How does local agentic RAG ensure data privacy? | 📄 NO | 100.0% | 99.0% | 100.0% | 3 | 4.1s |
| `ID-002` | `in_domain` | What vector database and indexing strategy is utilized for local storage? | 📄 NO | 100.0% | 95.4% | 80.0% | 2 | 4.09s |
| `ID-003` | `in_domain` | How does FastEmbed optimize embedding computation on CPU hardware? | 🌐 YES | 100.0% | 100.0% | 100.0% | 4 | 14.16s |
| `ID-004` | `in_domain` | What mechanism ensures document ingestion idempotency? | 📄 NO | 100.0% | 90.5% | 60.0% | 2 | 4.08s |
| `OOD-001` | `out_of_domain` | What is the primary scientific objective of the James Webb Space Telescope? | 🌐 YES | 100.0% | 100.0% | 60.0% | 3 | 14.42s |
| `OOD-002` | `out_of_domain` | What distance from Earth has Voyager 1 reached in interstellar space? | 🌐 YES | 100.0% | 100.0% | 100.0% | 3 | 14.34s |

---

## 3. Evaluation Methodology

1. **Faithfulness / Self-RAG Grounding**: Computes token overlap ratio between claims in the synthesized response and retrieved context blocks to prevent hallucinated assertions.
2. **Context Precision**: Measures the proportion of retrieved chunks that contain semantic query terms.
3. **Web Fallback Trigger Accuracy**: Validates that out-of-domain queries correctly fail local relevance thresholds and trigger live web retrieval.
