# Movie Knowledge Graph Project

This repository follows the roadmap in `PROJECT_PLAN.md`.

Current direction:
- The project uses the Neo4j `recommendations` dump dataset.
- Neo4j Desktop loads the dump.
- Python validates the graph, runs exploration, computes topology metrics, and extracts manual features.

## 1. Setup

1. Prepare the environment:

```powershell
.\setup_env.ps1
```

2. Copy `.env.example` to `.env` and fill your Neo4j connection values.

Important fields:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

Desktop load steps:
- See `LOAD_RECOMMENDATIONS_DATASET.md`

## 2. Dataset

Use the Neo4j recommendations example dataset:
- Repository: `neo4j-graph-examples/recommendations`
- Dump file: `data/recommendations-40.dump`

The Python code expects the graph model from that dataset:
- Nodes: `Movie`, `Actor`, `Director`, `User`, `Genre`
- Relationships: `ACTED_IN`, `DIRECTED`, `RATED`, `IN_GENRE`

## 3. Run

```powershell
.\run_project.ps1
```

This will:
- Print the conceptual graph schema
- Validate the loaded dump dataset
- Run exploration queries
- Compute graph metrics
- Extract manual actor features
- Save CSV tables in `outputs/results/`
- Save a schema figure to `outputs/figures/schema_diagram.png`

## 4. Notes

- The code has simple English comments for learning purposes.
- The project now assumes the graph is already loaded in Neo4j Desktop.
- `matplotlib` is required because the schema figure is part of the project output.
- If you want to use VS Code terminal, run the `.ps1` scripts from the project root.
