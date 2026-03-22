# Movie Knowledge Graph Project

This repository follows the roadmap in `PROJECT_PLAN.md`.

Current status:
- Phase 0 started: project structure and configuration files created.
- Phase 1 started: graph schema, data loading, and first visualization modules added.
- Later phases are left for next iterations on purpose.

## 1. Setup

1. Prepare the environment:

```powershell
.\setup_env.ps1
```

This project currently uses the local `.venv` environment.

2. Install dependencies manually if you want:

```bash
.\.venv\Scripts\python -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill your Neo4j credentials.

The versions in `requirements.txt` were updated to match what installs on Python `3.14`.

## 2. Run (Phase 1)

```bash
.\run_project.ps1
```

This will:
- Print the conceptual graph schema
- Try to connect to Neo4j
- Load a small sample movie graph
- Add simple `IN_GENRE` relationships
- Print schema statistics
- Save a schema figure to `outputs/figures/schema_diagram.png`

## 3. Notes

- The code has simple English comments for learning purposes.
- Work is intentionally incremental and testable, not all phases at once.
- Current sample schema uses `Movie`, `Actor`, `Director`, `User`, and `Genre`.
- Current sample relationships are `ACTED_IN`, `DIRECTED`, `RATED`, and `IN_GENRE`.
- `matplotlib` is required now because the schema figure is part of the project output.
- `node2vec` is not in the requirements for now because its current package constraints do not fit this Python version.
- If you want to use VS Code terminal, run the `.ps1` scripts from the project root.
