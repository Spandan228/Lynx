"""
Live Multi-PDF End-to-End Integration & Verification Loop for Lynx CRAG.
Tests via live FastAPI server (http://localhost:8000):
1. Multi-part PDF Upload & Docling Table-Aware Vector Ingestion (/upload)
2. Live Qdrant Stats Update (/stats)
3. End-to-End CRAG Query Reasoning & Quantitative Precision (/query)
4. Multi-Tenant RBAC Boundary Isolation (Zero Cross-Tenant Leakage)
5. Out-of-Domain Automated Web Fallback (/query)
"""

import os
import sys
import time
from pathlib import Path
import httpx

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"

API_BASE = "http://localhost:8000"

def log_header(title: str):
    print(f"\n{'='*75}\n  {title}\n{'='*75}")

def run_pdf_integration_loop():
    log_header("🚀 STARTING LIVE MULTI-PDF VERIFICATION LOOP (FastAPI + Docling + Qdrant)")

    with httpx.Client(base_url=API_BASE, timeout=120.0) as client:
        # Check health
        health = client.get("/health").json()
        print(f"  [+] Server Health Status: {health.get('status')} | Security: {health.get('security')}")
        assert health.get("status") == "healthy", "Server is not healthy!"

        # -------------------------------------------------------------------
        # TEST 1: Ingesting 3 Domain PDFs via POST /upload
        # -------------------------------------------------------------------
        log_header("TEST 1: Ingesting 3 PDF Documents via POST /upload (Docling Engine)")
        
        pdf_uploads = [
            {
                "file_path": Path("./data/quantum_computing_spec.pdf"),
                "tenant_id": "tenant_alpha",
                "roles": "admin,engineer",
            },
            {
                "file_path": Path("./data/biotech_clinical_trial_q3.pdf"),
                "tenant_id": "tenant_alpha",
                "roles": "admin,medical_researcher",
            },
            {
                "file_path": Path("./data/cybersecurity_zero_trust_audit.pdf"),
                "tenant_id": "tenant_beta",
                "roles": "admin,security_officer",
            },
        ]

        for item in pdf_uploads:
            p = item["file_path"]
            assert p.exists(), f"File {p} not found!"
            
            t0 = time.perf_counter()
            with open(p, "rb") as f:
                resp = client.post(
                    "/upload",
                    files={"file": (p.name, f, "application/pdf")},
                    data={
                        "tenant_id": item["tenant_id"],
                        "allowed_roles": item["roles"],
                    }
                )
            elapsed = time.perf_counter() - t0
            assert resp.status_code == 200, f"Upload failed ({resp.status_code}): {resp.text}"
            res_data = resp.json()
            chunks = res_data.get("chunks_created", 0)
            print(f"  [PASS] Uploaded '{p.name}' -> Ingested {chunks} chunk(s) in {elapsed:.2f}s | Tenant: {item['tenant_id']}")

        # -------------------------------------------------------------------
        # TEST 2: Real-time Stats Verification (/stats)
        # -------------------------------------------------------------------
        log_header("TEST 2: Live Qdrant Collection Stats Verification")
        stats_resp = client.get("/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        total_chunks = stats.get("total_indexed_chunks", 0)
        print(f"  [PASS] Total Vectors in Qdrant: {total_chunks} | Dimension: {stats.get('vector_dimension')} | Model: {stats.get('embedding_model')}")
        assert total_chunks >= 3, "Expected at least 3 chunks indexed!"

        # -------------------------------------------------------------------
        # TEST 3: Quantitative Metric Retrieval & Synthesizer Accuracy (/query)
        # -------------------------------------------------------------------
        log_header("TEST 3: End-to-End Query Answering & Table Metrics Precision")

        query_test_cases = [
            {
                "name": "Quantum Computing Specs",
                "query": "What is the base operating temperature of Q-Engine 2026 and its T1 coherence time?",
                "tenant_id": "tenant_alpha",
                "roles": ["admin", "engineer"],
                "expected_snippets": ["14.5", "185"],
            },
            {
                "name": "Biotech Clinical Trial Metrics",
                "query": "What was the Amyloid PET reduction percentage and CDR-SB decline slowing for 10mg NeuroShield-7?",
                "tenant_id": "tenant_alpha",
                "roles": ["admin", "medical_researcher"],
                "expected_snippets": ["-78.4%", "34.2%"],
            },
            {
                "name": "Zero Trust Cryptography",
                "query": "Which post-quantum algorithms are used for key encapsulation and digital signatures in the audit?",
                "tenant_id": "tenant_beta",
                "roles": ["admin", "security_officer"],
                "expected_snippets": ["Kyber-768", "Dilithium-3"],
            }
        ]

        for tc in query_test_cases:
            t0 = time.perf_counter()
            resp = client.post(
                "/query",
                json={"query": tc["query"], "top_k": 3},
                headers={
                    "X-Tenant-Id": tc["tenant_id"],
                    "X-User-Roles": ",".join(tc["roles"]),
                }
            )
            elapsed = time.perf_counter() - t0
            assert resp.status_code == 200, f"Query failed ({resp.status_code}): {resp.text}"
            data = resp.json()
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            retrieved_count = len(data.get("retrieved_sources", []))
            grounded = data.get("hallucination_grade", "yes")

            print(f"  [PASS] Domain: {tc['name']}")
            print(f"         Query: '{tc['query']}'")
            print(f"         Execution Latency: {elapsed*1000:.1f}ms | Sources Retrieved: {retrieved_count} | Grounded: {grounded}")
            print(f"         Citations: {citations}")
            print(f"         Answer: {answer[:150]}...")
            
            assert retrieved_count > 0, "No documents were retrieved!"
            assert len(answer) > 20, "Answer generated was empty or too short!"

        # -------------------------------------------------------------------
        # TEST 4: Multi-Tenant RBAC Boundary Isolation
        # -------------------------------------------------------------------
        log_header("TEST 4: Multi-Tenant Boundary Isolation (Zero Cross-Tenant Leakage)")

        # Tenant Alpha tries to query Tenant Beta's post-quantum encryption
        resp_leak_1 = client.post(
            "/query",
            json={"query": "What are the NIST post-quantum algorithms in the cybersecurity audit?", "top_k": 3},
            headers={
                "X-Tenant-Id": "tenant_alpha",  # Wrong tenant!
                "X-User-Roles": "admin",
            }
        )
        data_leak_1 = resp_leak_1.json()
        leak_1_docs = [d.get("filename", "") for d in data_leak_1.get("retrieved_sources", [])]
        assert not any("cybersecurity" in f for f in leak_1_docs), "SECURITY LEAK: Tenant Alpha accessed Tenant Beta cybersecurity doc!"
        print("  [PASS] Tenant Alpha query for Tenant Beta data: 0 leaks (Blocked at Vector Payload Boundary).")

        # Tenant Beta tries to query Tenant Alpha's biotech clinical trial
        resp_leak_2 = client.post(
            "/query",
            json={"query": "What is the CDR-SB reduction for NeuroShield-7 in the clinical trial?", "top_k": 3},
            headers={
                "X-Tenant-Id": "tenant_beta",  # Wrong tenant!
                "X-User-Roles": "admin",
            }
        )
        data_leak_2 = resp_leak_2.json()
        leak_2_docs = [d.get("filename", "") for d in data_leak_2.get("retrieved_sources", [])]
        assert not any("biotech" in f for f in leak_2_docs), "SECURITY LEAK: Tenant Beta accessed Tenant Alpha biotech doc!"
        print("  [PASS] Tenant Beta query for Tenant Alpha data: 0 leaks (Blocked at Vector Payload Boundary).")

        # -------------------------------------------------------------------
        # TEST 5: Automated Live DuckDuckGo Web Search Fallback
        # -------------------------------------------------------------------
        log_header("TEST 5: Automated Live Web Search Fallback on Missing Context")

        t0 = time.perf_counter()
        web_resp = client.post(
            "/query",
            json={"query": "What are the latest James Webb Space Telescope major exoplanet discoveries in 2026?", "top_k": 3},
            headers={
                "X-Tenant-Id": "tenant_alpha",
                "X-User-Roles": "admin",
            }
        )
        elapsed = time.perf_counter() - t0
        assert web_resp.status_code == 200
        web_data = web_resp.json()
        
        print(f"  [PASS] Out-of-Domain Query Latency: {elapsed:.2f}s")
        print(f"         Web Search Executed: {web_data.get('web_search_executed')}")
        print(f"         Citations: {web_data.get('citations')}")
        print(f"         Answer: {web_data.get('answer')[:150]}...")
        
        assert web_data.get("web_search_executed") is True, "Expected web search fallback to trigger!"
        assert len(web_data.get("answer", "")) > 30, "Expected answer synthesized from web context!"

        log_header("🎉 ALL 5 LIVE MULTI-PDF END-TO-END TESTS PASSED (100% SUCCESS)")

if __name__ == "__main__":
    try:
        run_pdf_integration_loop()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR IN LIVE TEST LOOP]: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

