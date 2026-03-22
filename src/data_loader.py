"""Validation helpers for the Neo4j recommendations dump dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

import config

EXPECTED_LABELS = {"Movie", "Actor", "Director", "User", "Genre"}
EXPECTED_RELATIONSHIPS = {"ACTED_IN", "DIRECTED", "RATED", "IN_GENRE"}
MIN_EXPECTED_COUNTS = {
    "Movie": 1000,
    "User": 1000,
    "Actor": 1000,
}


@dataclass
class SchemaStats:
    """Simple container to print and inspect schema counts."""

    node_counts: List[Tuple[str, int]]
    relationship_counts: List[Tuple[str, int]]


class MovieGraphLoader:
    """
    Validate an already loaded Neo4j dataset.

    In the recommendations dump workflow, Neo4j Desktop loads the dump file.
    Python starts after that and analyzes the existing graph.
    """

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()

    def test_connection(self) -> None:
        """Run a tiny query to check Neo4j connectivity."""
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            record = session.run("RETURN 1 AS ok").single()
        if not record or record["ok"] != 1:
            raise RuntimeError("Neo4j connection test failed.")

    def verify_schema(self) -> SchemaStats:
        """Return node and relationship counts by type."""
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            node_result = session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(*) AS count
                ORDER BY label
                """
            )
            relationship_result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY type
                """
            )

            node_counts = [(record["label"], record["count"]) for record in node_result]
            relationship_counts = [(record["type"], record["count"]) for record in relationship_result]

        return SchemaStats(node_counts=node_counts, relationship_counts=relationship_counts)

    def validate_expected_model(self, stats: SchemaStats) -> None:
        """Check whether the loaded dump looks like the recommendations dataset."""
        found_labels = {label for label, _ in stats.node_counts}
        found_relationships = {rel_type for rel_type, _ in stats.relationship_counts}
        count_map = {label: count for label, count in stats.node_counts}

        missing_labels = sorted(EXPECTED_LABELS - found_labels)
        missing_relationships = sorted(EXPECTED_RELATIONSHIPS - found_relationships)
        too_small = [
            f"{label}<{minimum}"
            for label, minimum in MIN_EXPECTED_COUNTS.items()
            if count_map.get(label, 0) < minimum
        ]

        if missing_labels or missing_relationships or too_small:
            lines = ["Loaded database does not match the expected recommendations graph."]
            if missing_labels:
                lines.append(f"Missing labels: {', '.join(missing_labels)}")
            if missing_relationships:
                lines.append(f"Missing relationship types: {', '.join(missing_relationships)}")
            if too_small:
                lines.append(f"Dataset looks too small: {', '.join(too_small)}")
            raise RuntimeError(" | ".join(lines))

    def print_schema_stats(self, stats: SchemaStats) -> None:
        """Print schema stats in a friendly way."""
        print("\n=== Node Counts ===")
        for label, count in stats.node_counts:
            print(f"- {label}: {count}")

        print("\n=== Relationship Counts ===")
        for rel_type, count in stats.relationship_counts:
            print(f"- {rel_type}: {count}")


def run_phase1_load() -> None:
    """Validate the already loaded dump dataset."""
    loader = MovieGraphLoader()
    try:
        print("Checking Neo4j connection...")
        loader.test_connection()
        print("Connection OK.")

        print(f"Using database: {config.NEO4J_DATABASE}")
        print(f"Dataset mode: {config.DATASET_NAME}")

        stats = loader.verify_schema()
        loader.validate_expected_model(stats)
        loader.print_schema_stats(stats)
        print("\nExisting recommendations dataset is ready.")
    except Neo4jError as exc:
        print("Neo4j error during validation:")
        print(f"- {exc}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    run_phase1_load()
