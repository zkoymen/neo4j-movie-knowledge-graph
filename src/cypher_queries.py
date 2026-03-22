"""Basic Cypher exploration queries for Phase 2."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

import config

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class GraphExplorer:
    """Run simple Cypher queries and save their results."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def _query_to_df(self, query: str, **params: object) -> pd.DataFrame:
        """Run a query and return the rows as a DataFrame."""
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(query, **params)
            return pd.DataFrame([dict(record) for record in result])

    def _save_df(self, df: pd.DataFrame, filename: str) -> Path:
        """Save a DataFrame in outputs/results."""
        output_path = RESULTS_DIR / filename
        df.to_csv(output_path, index=False)
        return output_path

    def get_node_counts(self) -> pd.DataFrame:
        """Count nodes by label."""
        return self._query_to_df(
            """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC, label
            """
        )

    def get_relationship_counts(self) -> pd.DataFrame:
        """Count relationships by type."""
        return self._query_to_df(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC, type
            """
        )

    def get_top_rated_movies(self) -> pd.DataFrame:
        """Return movies sorted by average rating."""
        return self._query_to_df(
            """
            MATCH (m:Movie)
            OPTIONAL MATCH (:User)-[r:RATED]->(m)
            WITH m, round(avg(r.rating) * 10) / 10.0 AS source_avg_rating, count(r) AS rating_count
            RETURN m.title AS title,
                   m.year AS year,
                   coalesce(m.imdbRating, m.rating, source_avg_rating) AS avg_rating,
                   rating_count
            ORDER BY avg_rating DESC, rating_count DESC, title
            """
        )

    def get_movies_by_genre(self) -> pd.DataFrame:
        """Return movies grouped by genre."""
        return self._query_to_df(
            """
            MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
            RETURN g.name AS genre,
                   collect(m.title) AS movies,
                   count(m) AS movie_count
            ORDER BY movie_count DESC, genre
            """
        )

    def get_actor_movie_counts(self) -> pd.DataFrame:
        """Return how many movies each actor appears in."""
        return self._query_to_df(
            """
            MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
            RETURN a.name AS actor,
                   count(m) AS movie_count,
                   collect(m.title) AS movies
            ORDER BY movie_count DESC, actor
            """
        )

    def get_actor_collaborations(self) -> pd.DataFrame:
        """Return actors who acted in the same movie."""
        return self._query_to_df(
            """
            MATCH (a1:Actor)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Actor)
            WHERE a1.name < a2.name
            RETURN a1.name AS actor_1,
                   a2.name AS actor_2,
                   count(m) AS shared_movies,
                   collect(m.title) AS movies
            ORDER BY shared_movies DESC, actor_1, actor_2
            """
        )

    def get_director_movie_counts(self) -> pd.DataFrame:
        """Return how many movies each director directed."""
        return self._query_to_df(
            """
            MATCH (d:Director)-[:DIRECTED]->(m:Movie)
            RETURN d.name AS director,
                   count(m) AS movie_count,
                   collect(m.title) AS movies
            ORDER BY movie_count DESC, director
            """
        )

    def get_actor_director_pairs(self) -> pd.DataFrame:
        """Return actor and director collaborations."""
        return self._query_to_df(
            """
            MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director)
            RETURN a.name AS actor,
                   d.name AS director,
                   count(m) AS collaborations,
                   collect(m.title) AS movies
            ORDER BY collaborations DESC, actor, director
            """
        )

    def run_basic_exploration(self) -> dict[str, pd.DataFrame]:
        """Run the first exploration set and save all CSV files."""
        results = {
            "node_counts": self.get_node_counts(),
            "relationship_counts": self.get_relationship_counts(),
            "top_rated_movies": self.get_top_rated_movies(),
            "movies_by_genre": self.get_movies_by_genre(),
            "actor_movie_counts": self.get_actor_movie_counts(),
            "actor_collaborations": self.get_actor_collaborations(),
            "director_movie_counts": self.get_director_movie_counts(),
            "actor_director_pairs": self.get_actor_director_pairs(),
        }

        for name, df in results.items():
            self._save_df(df, f"{name}.csv")

        return results


if __name__ == "__main__":
    explorer = GraphExplorer()
    try:
        query_results = explorer.run_basic_exploration()
        for name, df in query_results.items():
            print(f"\n=== {name} ===")
            print(df)
    finally:
        explorer.close()
