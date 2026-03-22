"""Neo4j data loading helpers for Phase 1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

import config


# We keep small sample data here so Phase 1 can run quickly.
SAMPLE_MOVIES: List[Tuple[str, int, str]] = [
    ("The Matrix", 1999, "Welcome to the Real World"),
    ("John Wick", 2014, "Don't set him off"),
    ("Speed", 1994, "Get ready for rush hour"),
]

SAMPLE_PEOPLE: List[Tuple[str, int]] = [
    ("Keanu Reeves", 1964),
    ("Laurence Fishburne", 1961),
    ("Carrie-Anne Moss", 1967),
    ("Lana Wachowski", 1965),
    ("Lilly Wachowski", 1967),
    ("Chad Stahelski", 1968),
    ("Jan de Bont", 1943),
]

SAMPLE_RELATIONSHIPS: List[Tuple[str, str, str, Dict[str, List[str]]]] = [
    ("Keanu Reeves", "ACTED_IN", "The Matrix", {"roles": ["Neo"]}),
    ("Laurence Fishburne", "ACTED_IN", "The Matrix", {"roles": ["Morpheus"]}),
    ("Carrie-Anne Moss", "ACTED_IN", "The Matrix", {"roles": ["Trinity"]}),
    ("Lana Wachowski", "DIRECTED", "The Matrix", {}),
    ("Lilly Wachowski", "DIRECTED", "The Matrix", {}),
    ("Keanu Reeves", "ACTED_IN", "John Wick", {"roles": ["John Wick"]}),
    ("Chad Stahelski", "DIRECTED", "John Wick", {}),
    ("Keanu Reeves", "ACTED_IN", "Speed", {"roles": ["Jack Traven"]}),
    ("Jan de Bont", "DIRECTED", "Speed", {}),
]


@dataclass
class SchemaStats:
    """Simple container to print and inspect schema counts."""

    node_counts: List[Tuple[str, int]]
    relationship_counts: List[Tuple[str, int]]


class MovieGraphLoader:
    """Load and enrich the movie knowledge graph."""

    def __init__(self) -> None:
        # Driver is created once and reused by methods.
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()

    def test_connection(self) -> None:
        """Run a tiny query to check Neo4j connectivity."""
        with self.driver.session() as session:
            record = session.run("RETURN 1 AS ok").single()
        if not record or record["ok"] != 1:
            raise RuntimeError("Neo4j connection test failed.")

    def clear_database(self) -> None:
        """Remove old graph so each run starts clean."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def load_sample_movies_dataset(self) -> None:
        """
        Load a small, clean sample graph.

        Why sample graph:
        - It is fast.
        - It is easy to verify.
        - It avoids long setup in first iteration.
        """
        with self.driver.session() as session:
            for title, released, tagline in SAMPLE_MOVIES:
                session.run(
                    """
                    MERGE (m:Movie {title: $title})
                    SET m.released = $released,
                        m.tagline = $tagline
                    """,
                    title=title,
                    released=released,
                    tagline=tagline,
                )

            for name, born in SAMPLE_PEOPLE:
                session.run(
                    """
                    MERGE (p:Person {name: $name})
                    SET p.born = $born
                    """,
                    name=name,
                    born=born,
                )

            for person_name, rel_type, movie_title, rel_props in SAMPLE_RELATIONSHIPS:
                # Relationship type cannot be parameterized in Cypher.
                query = f"""
                    MATCH (p:Person {{name: $person_name}})
                    MATCH (m:Movie {{title: $movie_title}})
                    MERGE (p)-[r:{rel_type}]->(m)
                    SET r += $rel_props
                """
                session.run(
                    query,
                    person_name=person_name,
                    movie_title=movie_title,
                    rel_props=rel_props,
                )

    def add_genre_nodes(self) -> None:
        """
        Create Genre nodes and BELONGS_TO links.

        We map genres manually in this first version.
        """
        genre_mapping = {
            "The Matrix": ["Sci-Fi", "Action"],
            "John Wick": ["Action", "Thriller"],
            "Speed": ["Action", "Thriller"],
        }
        with self.driver.session() as session:
            for movie_title, genres in genre_mapping.items():
                for genre in genres:
                    session.run(
                        """
                        MATCH (m:Movie {title: $title})
                        MERGE (g:Genre {name: $genre})
                        MERGE (m)-[:BELONGS_TO]->(g)
                        """,
                        title=movie_title,
                        genre=genre,
                    )

    def verify_schema(self) -> SchemaStats:
        """Return node and relationship counts by type."""
        with self.driver.session() as session:
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

        return SchemaStats(
            node_counts=node_counts,
            relationship_counts=relationship_counts,
        )

    def print_schema_stats(self, stats: SchemaStats) -> None:
        """Print schema stats in a friendly way."""
        print("\n=== Node Counts ===")
        for label, count in stats.node_counts:
            print(f"- {label}: {count}")

        print("\n=== Relationship Counts ===")
        for rel_type, count in stats.relationship_counts:
            print(f"- {rel_type}: {count}")


def run_phase1_load() -> None:
    """Helper function to run the full Phase 1 loading flow."""
    loader = MovieGraphLoader()
    try:
        print("Checking Neo4j connection...")
        loader.test_connection()
        print("Connection OK.")

        print("Clearing old data...")
        loader.clear_database()

        print("Loading sample movies dataset...")
        loader.load_sample_movies_dataset()

        print("Adding genre nodes...")
        loader.add_genre_nodes()

        stats = loader.verify_schema()
        loader.print_schema_stats(stats)
        print("\nData loading finished.")
    except Neo4jError as exc:
        print("Neo4j error during Phase 1 loading:")
        print(f"- {exc}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    run_phase1_load()
