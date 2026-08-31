# Contributing to Lynx CRAG

Thank you for your interest in contributing! This document outlines how to set up your development environment and submit high-quality contributions.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Coding Standards](#coding-standards)

---

## Code of Conduct

Be respectful, collaborative, and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## Getting Started

### Prerequisites

- Python 3.10+
- Ollama with `llama3.2:3b` and `llama3` pulled
- Git

### Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/Lynx.git
cd Lynx

# 2. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install in editable mode (src layout)
pip install -e ".[dev]"
# Or plain:
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env
# Edit .env with your values
```

---

## Project Structure

```
src/lynx/      ← All source code (importable as `lynx.*`)
tests/         ← All test files (pytest)
scripts/       ← Utility scripts (load testing, benchmarks, PDF generation)
static/        ← Frontend assets (HTML, CSS, JS)
data/          ← Sample knowledge documents
reports/       ← Benchmark and load test reports
```

---

## Development Workflow

```bash
# Start the API server
uvicorn lynx.app:app --host 0.0.0.0 --port 8000 --reload

# Run the isolated CI test suite (no live services needed)
pytest tests/test_ci.py -v

# Run all tests (requires Ollama + Qdrant running)
pytest tests/ -v
```

---

## Running Tests

| Command | Description |
|---|---|
| `pytest tests/test_ci.py -v` | Isolated unit tests (CI safe, no live infra) |
| `pytest tests/test_pipeline.py -v` | End-to-end CRAG integration tests |
| `pytest tests/test_multi_tenant_security.py -v` | Multi-tenant RBAC boundary tests |
| `pytest tests/ -v` | Full test suite |

---

## Submitting a Pull Request

1. **Fork** the repository and create your branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** with clear, focused commits:
   ```bash
   git commit -m "feat: Add X capability to Y module"
   ```
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` — new feature
   - `fix:` — bug fix
   - `docs:` — documentation only
   - `refactor:` — code restructure, no behavior change
   - `test:` — adding or updating tests
   - `chore:` — tooling, CI, dependencies

3. **Ensure tests pass:**
   ```bash
   pytest tests/test_ci.py -v
   ```

4. **Open a Pull Request** against `main` with a clear description of what and why.

---

## Coding Standards

- **Formatter**: `ruff format` (line length: 100)
- **Linter**: `ruff check`
- **Type hints**: Required on all public functions
- **Docstrings**: Google-style docstrings on all classes and public methods
- **Imports**: Always use `from lynx.X import Y` — never add root to sys.path in source files

```bash
# Check code style
ruff check src/ tests/
ruff format src/ tests/
```

---

## Questions?

Open a [GitHub Discussion](https://github.com/Spandan228/Lynx/discussions) or file an [Issue](https://github.com/Spandan228/Lynx/issues).
