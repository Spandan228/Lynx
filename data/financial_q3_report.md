# Q3 Infrastructure Financial & Performance Report

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
