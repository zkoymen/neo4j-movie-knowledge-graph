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

SAMPLE_ACTORS: List[Tuple[str, int]] = [
    ("Keanu Reeves", 1964),
    ("Laurence Fishburne", 1961),
    ("Carrie-Anne Moss", 1967),
]

SAMPLE_DIRECTORS: List[Tuple[str, int]] = [
    ("Lana Wachowski", 1965),
    ("Lilly Wachowski", 1967),
    ("Chad Stahelski", 1968),
    ("Jan de Bont", 1943),
]

SAMPLE_USERS: List[str] = [
    "Alice",
    "Bob",
]

SAMPLE_ACTED_IN: List[Tuple[str, str, Dict[str, List[str]]]] = [
    ("Keanu Reeves", "The Matrix", {"roles": ["Neo"]}),
    ("Laurence Fishburne", "The Matrix", {"roles": ["Morpheus"]}),
    ("Carrie-Anne Moss", "The Matrix", {"roles": ["Trinity"]}),
    ("Keanu Reeves", "John Wick", {"roles": ["John Wick"]}),
    ("Keanu Reeves", "Speed", {"roles": ["Jack Traven"]}),
]

SAMPLE_DIRECTED: List[Tuple[str, str]] = [
    ("Lana Wachowski", "The Matrix"),
    ("Lilly Wachowski", "The Matrix"),
    ("Chad Stahelski", "John Wick"),
    ("Jan de Bont", "Speed"),
]

SAMPLE_RATINGS: List[Tuple[str, str, int]] = [
    ("Alice", "The Matrix", 9),
    ("Alice", "John Wick", 8),
    ("Bob", "The Matrix", 10),
    ("Bob", "Speed", 7),
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
            for title, year, tagline in SAMPLE_MOVIES:
                session.run(
                    """
                    MERGE (m:Movie {title: $title})
                    SET m.year = $year,
                        m.released = $year,
                        m.tagline = $tagline
                    """,
                    title=title,
                    year=year,
                    tagline=tagline,
                )

            for name, born in SAMPLE_ACTORS:
                session.run(
                    """
                    MERGE (a:Actor {name: $name})
                    SET a.born = $born
                    """,
                    name=name,
                    born=born,
                )

            for name, born in SAMPLE_DIRECTORS:
                session.run(
                    """
                    MERGE (d:Director {name: $name})
                    SET d.born = $born
                    """,
                    name=name,
                    born=born,
                )

            for name in SAMPLE_USERS:
                session.run(
                    """
                    MERGE (u:User {name: $name})
                    """,
                    name=name,
                )

            for actor_name, movie_title, rel_props in SAMPLE_ACTED_IN:
                session.run(
                    """
                    MATCH (a:Actor {name: $actor_name})
                    MATCH (m:Movie {title: $movie_title})
                    MERGE (a)-[r:ACTED_IN]->(m)
                    SET r += $rel_props
                    """,
                    actor_name=actor_name,
                    movie_title=movie_title,
                    rel_props=rel_props,
                )

            for director_name, movie_title in SAMPLE_DIRECTED:
                session.run(
                    """
                    MATCH (d:Director {name: $director_name})
                    MATCH (m:Movie {title: $movie_title})
                    MERGE (d)-[:DIRECTED]->(m)
                    """,
                    director_name=director_name,
                    movie_title=movie_title,
                )

            for user_name, movie_title, rating in SAMPLE_RATINGS:
                session.run(
                    """
                    MATCH (u:User {name: $user_name})
                    MATCH (m:Movie {title: $movie_title})
                    MERGE (u)-[r:RATED]->(m)
                    SET r.rating = $rating
                    """,
                    user_name=user_name,
                    movie_title=movie_title,
                    rating=rating,
                )

    def add_genre_nodes(self) -> None:
        """
        Create Genre nodes and IN_GENRE links.

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
                        MERGE (m)-[:IN_GENRE]->(g)
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
