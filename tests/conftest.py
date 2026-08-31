"""
Pytest configuration and shared fixtures for Lynx CRAG test suite.
Ensures the `src/` layout is on sys.path so `lynx.*` imports resolve.
"""
import sys
from pathlib import Path

# Add src/ to Python path so tests can import from lynx.*
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
