"""
Comprehensive Multi-Tenant & Role-Based Access Control (RBAC) Security Test Suite.

Verifies:
1. Multi-Tenant Isolation: Complete cryptographic/payload separation between Tenant A and Tenant B.
2. Role-Based Access Control: Regular user role cannot retrieve 'admin_only' documents.
3. BM25 Sparse Boundary Enforcement: Sparse keyword index does not leak unauthorized context snippets.
4. JWT Bearer Token Lifecycle: Signing, expiration, decoding, and FastAPI Dependency injection.
5. End-to-End API /query RBAC enforcement via FastAPI TestClient.

Author: Cloud Security and IAM Architect
"""

import os
import sys
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

# Security & RAG Module Imports
from lynx.auth import (
    UserSecurityContext,
    create_access_token,
    decode_access_token,
    JWT_SECRET_KEY,
)
from lynx.ingest import (
    IngestionPipeline,
    IngestionConfig,
    DocumentChunk,
    QdrantVectorStore,
    LocalEmbeddingEngine,
)
from lynx.retriever import HybridRetriever, RetrievedChunk
from lynx.graph import create_crag_graph
from lynx.app import app, service_state


def run_multi_tenant_security_tests():
    print("=" * 75)
    print("  MULTI-TENANT & ROLE-BASED ACCESS CONTROL (RBAC) SECURITY TEST SUITE  ")
    print("=" * 75)

    temp_dir = tempfile.mkdtemp(prefix="test_security_rag_")
    test_qdrant_path = os.path.join(temp_dir, "qdrant_db")
    test_data_dir = os.path.join(temp_dir, "data")
    os.makedirs(test_data_dir, exist_ok=True)

    passed_tests = 0
    total_tests = 5

    try:
        # Create single shared QdrantClient to avoid local file locking collisions
        shared_client = QdrantClient(path=test_qdrant_path)
        collection_name = "test_security_knowledge"

        # -------------------------------------------------------------------
        # TEST 1: JWT Access Token Creation & Validation
        # -------------------------------------------------------------------
        print("\n[TEST 1/5] Testing JWT Access Token Signing & RBAC Claim Verification...")
        admin_ctx = UserSecurityContext(
            tenant_id="tenant_alpha",
            user_id="usr_admin_001",
            roles=["admin", "finance_reader"],
            email="admin@alpha.corp",
        )
        token = create_access_token(admin_ctx)
        assert isinstance(token, str) and len(token) > 20, "JWT token generation failed."

        decoded_ctx = decode_access_token(token)
        assert decoded_ctx.tenant_id == "tenant_alpha", f"Expected tenant_alpha, got {decoded_ctx.tenant_id}"
        assert decoded_ctx.user_id == "usr_admin_001", f"Expected usr_admin_001, got {decoded_ctx.user_id}"
        assert "admin" in decoded_ctx.roles, "Role 'admin' missing in decoded claims."
        assert decoded_ctx.has_role("admin") is True, "has_role('admin') check failed."
        assert decoded_ctx.overlaps_roles(["finance_reader"]) is True, "overlaps_roles check failed."

        print(f"  [PASS] JWT Token successfully signed and decoded with tenant '{decoded_ctx.tenant_id}' & roles {decoded_ctx.roles}.")
        passed_tests += 1

        # -------------------------------------------------------------------
        # Ingestion Setup: Multi-Tenant & Multi-Role Documents
        # -------------------------------------------------------------------
        print("\n[*] Initializing Ingestion Layer with RBAC Payload Indexing...")
        config = IngestionConfig(
            data_dir=test_data_dir,
            qdrant_path=test_qdrant_path,
            collection_name=collection_name,
        )
        pipeline = IngestionPipeline(config=config, client=shared_client)

        # Document 1: Tenant Alpha - General Knowledge (Roles: user, admin)
        doc1_path = Path(test_data_dir) / "alpha_general.txt"
        doc1_path.write_text(
            "Project Aurora is Tenant Alpha's public initiative for distributed energy grid optimization.",
            encoding="utf-8",
        )

        # Document 2: Tenant Alpha - Confidential Executive Strategy (Roles: admin)
        doc2_path = Path(test_data_dir) / "alpha_executive_secret.txt"
        doc2_path.write_text(
            "Project Starlight Executive Compensation and Confidential Alpha Strategy: Executive bonuses are capped at 400% with Q4 stock awards.",
            encoding="utf-8",
        )

        # Document 3: Tenant Beta - Confidential Beta Specs (Roles: admin, user)
        doc3_path = Path(test_data_dir) / "beta_propulsion.txt"
        doc3_path.write_text(
            "Project Nebula Ion Propulsion Engine is proprietary technology developed exclusively by Tenant Beta.",
            encoding="utf-8",
        )

        res1 = pipeline.process_file(
            doc1_path,
            tenant_id="tenant_alpha",
            owner_id="usr_alice",
            allowed_roles=["user", "admin"],
        )
        res2 = pipeline.process_file(
            doc2_path,
            tenant_id="tenant_alpha",
            owner_id="usr_admin_001",
            allowed_roles=["admin"],
        )
        res3 = pipeline.process_file(
            doc3_path,
            tenant_id="tenant_beta",
            owner_id="usr_bob",
            allowed_roles=["user", "admin"],
        )

        assert res1["status"] == "indexed", "Failed indexing doc1"
        assert res2["status"] == "indexed", "Failed indexing doc2"
        assert res3["status"] == "indexed", "Failed indexing doc3"
        print(f"  [+] Ingested 3 documents across 'tenant_alpha' (general & admin_only) and 'tenant_beta'.")

        # Initialize Retriever with the shared Qdrant client
        retriever = HybridRetriever(
            qdrant_path=test_qdrant_path,
            collection_name=collection_name,
            client=shared_client,
        )

        # -------------------------------------------------------------------
        # TEST 2: Strict Cross-Tenant Isolation
        # -------------------------------------------------------------------
        print("\n[TEST 2/5] Testing Cross-Tenant Data Isolation...")
        # Tenant Alpha user searches for Tenant Beta's proprietary tech ("Project Nebula Ion Propulsion")
        tenant_alpha_user = UserSecurityContext(
            tenant_id="tenant_alpha",
            user_id="usr_charlie",
            roles=["user", "admin"],
        )
        leak_results = retriever.search(
            query="Project Nebula Ion Propulsion Engine technology",
            top_k=5,
            security_context=tenant_alpha_user,
        )

        # Ensure NO chunks from Tenant Beta are returned
        for chunk in leak_results:
            chunk_tenant = chunk.tenant_id or chunk.metadata.get("tenant_id")
            assert chunk_tenant != "tenant_beta", f"CRITICAL LEAKAGE: Tenant Alpha retrieved Tenant Beta document '{chunk.filename}'!"

        # Ensure Tenant Beta CAN retrieve their own document
        tenant_beta_user = UserSecurityContext(
            tenant_id="tenant_beta",
            user_id="usr_bob",
            roles=["user"],
        )
        beta_results = retriever.search(
            query="Project Nebula Ion Propulsion Engine technology",
            top_k=5,
            security_context=tenant_beta_user,
        )
        assert len(beta_results) > 0, "Tenant Beta failed to retrieve its own document."
        assert beta_results[0].tenant_id == "tenant_beta" or beta_results[0].metadata.get("tenant_id") == "tenant_beta"

        print(f"  [PASS] Zero cross-tenant data leakage verified: Tenant Alpha queries for Beta content returned {len(leak_results)} matching chunks, while Tenant Beta retrieved {len(beta_results)} chunk(s).")
        passed_tests += 1

        # -------------------------------------------------------------------
        # TEST 3: Role-Based Access Control (RBAC) Filtering
        # -------------------------------------------------------------------
        print("\n[TEST 3/5] Testing Role-Based Access Control (RBAC: 'user' vs 'admin')...")
        regular_user = UserSecurityContext(
            tenant_id="tenant_alpha",
            user_id="usr_standard",
            roles=["user"],  # Only regular 'user' role
        )

        # Regular user tries to retrieve 'admin_only' executive compensation
        unauthorized_results = retriever.search(
            query="Project Starlight Executive Compensation and Confidential bonuses",
            top_k=5,
            security_context=regular_user,
        )

        for chunk in unauthorized_results:
            assert chunk.filename != "alpha_executive_secret.txt", f"RBAC BREACH: Regular user retrieved admin-only file '{chunk.filename}'!"

        # Admin user queries same prompt
        admin_user = UserSecurityContext(
            tenant_id="tenant_alpha",
            user_id="usr_admin_001",
            roles=["admin"],
        )
        authorized_results = retriever.search(
            query="Project Starlight Executive Compensation and Confidential bonuses",
            top_k=5,
            security_context=admin_user,
        )

        assert len(authorized_results) > 0, "Admin user should retrieve the executive secret document."
        admin_filenames = [c.filename for c in authorized_results]
        assert "alpha_executive_secret.txt" in admin_filenames, "Admin could not retrieve executive secret doc."

        print(f"  [PASS] RBAC Enforcement verified: User role retrieved {len(unauthorized_results)} unauthorized chunk(s) (blocked). Admin role retrieved {len(authorized_results)} authorized chunk(s).")
        passed_tests += 1

        # -------------------------------------------------------------------
        # TEST 4: BM25 Sparse Index Security Filtering
        # -------------------------------------------------------------------
        print("\n[TEST 4/5] Testing BM25 Sparse Keyword Index RBAC Filtering...")
        # Direct test on BM25 sparse_search to guarantee in-memory index security
        sparse_unauth = retriever.sparse_search(
            query="Starlight Executive Compensation",
            limit=5,
            security_context=regular_user,
        )
        for chunk, score in sparse_unauth:
            assert chunk.filename != "alpha_executive_secret.txt", f"SPARSE INDEX LEAK: BM25 returned admin chunk to standard user!"

        sparse_auth = retriever.sparse_search(
            query="Starlight Executive Compensation",
            limit=5,
            security_context=admin_user,
        )
        assert len(sparse_auth) > 0, "BM25 should return matching chunks to authorized admin."

        print(f"  [PASS] In-memory BM25 sparse index strictly respects tenant and role security filters.")
        passed_tests += 1

        # -------------------------------------------------------------------
        # TEST 5: End-to-End FastAPI REST API Authentication & RBAC
        # -------------------------------------------------------------------
        print("\n[TEST 5/5] Testing FastAPI /query & /auth/token End-to-End Integration...")
        # Inject test dependencies into service state
        service_state.hybrid_retriever = retriever
        service_state.ingestion_pipeline = pipeline
        service_state.qdrant_client = shared_client
        service_state.graph_runner = create_crag_graph(
            retriever=retriever,
            max_retrieval_retries=1,
            max_generation_retries=1,
        )

        client = TestClient(app)

        # 1. Test /auth/token endpoint
        token_payload = {
            "tenant_id": "tenant_alpha",
            "user_id": "usr_token_test",
            "roles": ["user"],
        }
        token_resp = client.post("/auth/token", json=token_payload)
        assert token_resp.status_code == 200, f"Token minting failed: {token_resp.text}"
        user_jwt = token_resp.json()["access_token"]

        # 2. Test /query with regular user JWT token -> attempts to access executive secrets
        headers = {"Authorization": f"Bearer {user_jwt}"}
        query_payload = {
            "query": "What are the executive stock bonuses in Project Starlight?",
            "top_k": 3,
        }
        resp = client.post("/query", json=query_payload, headers=headers)
        assert resp.status_code == 200, f"Query endpoint failed: {resp.text}"
        resp_data = resp.json()

        # Verify no retrieved sources contain alpha_executive_secret.txt
        retrieved_files = [s["filename"] for s in resp_data.get("retrieved_sources", [])]
        assert "alpha_executive_secret.txt" not in retrieved_files, f"Unauthorized file leaked in API response: {retrieved_files}"

        # 3. Test /query with Admin JWT token
        admin_token_resp = client.post("/auth/token", json={"tenant_id": "tenant_alpha", "user_id": "usr_admin", "roles": ["admin"]})
        admin_jwt = admin_token_resp.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_jwt}"}

        admin_resp = client.post("/query", json=query_payload, headers=admin_headers)
        assert admin_resp.status_code == 200
        admin_resp_data = admin_resp.json()
        admin_retrieved_files = [s["filename"] for s in admin_resp_data.get("retrieved_sources", [])]
        assert "alpha_executive_secret.txt" in admin_retrieved_files, "Admin user should retrieve executive secret file via API."

        print(f"  [PASS] FastAPI Bearer JWT Authentication and RBAC Verified: Standard token blocked unauthorized files; Admin token retrieved authorized context.")
        passed_tests += 1

        print("\n" + "=" * 75)
        print(f"  SECURITY AUDIT COMPLETED: {passed_tests}/{total_tests} TESTS PASSED (100% SUCCESS)  ")
        print("  ZERO CROSS-TENANT OR ROLE-BASED LEAKS DETECTED  ")
        print("=" * 75)

    finally:
        # Clean up temporary test files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_multi_tenant_security_tests()

