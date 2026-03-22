"""Main entrypoint for incremental project execution."""
from __future__ import annotations

from neo4j.exceptions import DriverError, Neo4jError, ServiceUnavailable

from src.data_loader import run_phase1_load
from src.graph_model import print_schema
from src.visualization import visualize_schema


def main() -> None:
    """Run only the first milestone for now (Phase 0 + Phase 1 base)."""
    print("=" * 60)
    print("MOVIE KNOWLEDGE GRAPH - MILESTONE 1")
    print("=" * 60)

    print("\n1) Printing conceptual schema...")
    print_schema()

    print("\n2) Saving schema diagram...")
    figure_path = visualize_schema()
    print(f"Schema diagram saved: {figure_path}")

    print("\n3) Loading and validating sample data in Neo4j...")
    try:
        run_phase1_load()
    except (Neo4jError, DriverError, ServiceUnavailable) as exc:
        print("\nNeo4j is not reachable or query failed.")
        print("Please check `.env` and Neo4j server status, then run again.")
        print(f"Details: {exc}")

    print("\nMilestone 1 finished.")


if __name__ == "__main__":
    main()
