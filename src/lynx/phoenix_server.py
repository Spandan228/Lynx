"""
Phoenix Observability Server Daemon
Runs Arize Phoenix on port 6006 for OpenTelemetry LLM tracing & UI with UTF-8 encoding.
"""
import sys
import time
import os

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    import phoenix as px
    print("Launching Arize Phoenix on port 6006 (0.0.0.0)...", flush=True)
    session = px.launch_app(port=6006, host="0.0.0.0")
    print(f"Arize Phoenix UI successfully active at: {session.url}", flush=True)
    while True:
        time.sleep(3600)
except Exception as e:
    print(f"Error starting Phoenix server: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
