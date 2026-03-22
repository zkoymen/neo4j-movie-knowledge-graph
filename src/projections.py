"""Graph projection utilities for Phase 3."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from neo4j import GraphDatabase

import config

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class GraphProjections:
    """Create projected graphs from the recommendations dataset."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def create_actor_cooccurrence_graph(
        self,
        min_movie_count: int | None = None,
        max_actors: int | None = None,
    ) -> nx.Graph:
        """Create actor-actor graph based on shared movies."""
        if min_movie_count is None:
            min_movie_count = config.CORE_ACTOR_MIN_MOVIES
        if max_actors is None:
            max_actors = config.CORE_ACTOR_MAX_ACTORS

        graph = nx.Graph()

        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (a:Actor)-[:ACTED_IN]->(:Movie)
                WITH a, count(*) AS movie_count
                WHERE movie_count >= $min_movie_count
                ORDER BY movie_count DESC, a.name
                LIMIT $max_actors
                WITH collect(a) AS actor_subset

                UNWIND actor_subset AS a1
                MATCH (a1)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Actor)
                WHERE a2 IN actor_subset AND a1.name < a2.name
                RETURN a1.name AS actor_1,
                       a2.name AS actor_2,
                       count(m) AS shared_movies,
                       collect(m.title)[0..10] AS movies
                """,
                min_movie_count=min_movie_count,
                max_actors=max_actors,
            )

            rows = []
            for record in result:
                graph.add_edge(
                    record["actor_1"],
                    record["actor_2"],
                    weight=record["shared_movies"],
                    movies=record["movies"],
                )
                rows.append(dict(record))

        pd.DataFrame(rows).to_csv(RESULTS_DIR / "actor_cooccurrence_projection.csv", index=False)
        return graph

    def create_movie_similarity_graph(
        self,
        min_score: int | None = None,
        max_movies: int | None = None,
    ) -> nx.Graph:
        """
        Create movie-movie similarity graph.

        Score:
        - shared genres: weight 3
        - shared actors: weight 2
        - shared directors: weight 4
        """
        if min_score is None:
            min_score = config.MOVIE_SIMILARITY_MIN_SCORE
        if max_movies is None:
            max_movies = config.MOVIE_SIMILARITY_MAX_MOVIES

        graph = nx.Graph()

        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(
                """
                // We only use a strong movie subset first.
                // This keeps the pair generation smaller and faster.
                MATCH (:User)-[r:RATED]->(m:Movie)
                WITH m, count(r) AS rating_count
                ORDER BY rating_count DESC, m.title
                LIMIT $max_movies
                WITH collect(m) AS movie_subset

                UNWIND movie_subset AS m1
                MATCH (m1)-[:IN_GENRE|ACTED_IN|DIRECTED]-(trait)-[:IN_GENRE|ACTED_IN|DIRECTED]-(m2:Movie)
                WHERE m2 IN movie_subset AND elementId(m1) < elementId(m2)
                WITH m1, m2, collect(DISTINCT trait) AS shared_traits
                WITH m1, m2,
                     size([t IN shared_traits WHERE 'Genre' IN labels(t)]) AS shared_genres,
                     size([t IN shared_traits WHERE 'Actor' IN labels(t)]) AS shared_actors,
                     size([t IN shared_traits WHERE 'Director' IN labels(t)]) AS shared_directors
                WITH m1, m2, shared_genres, shared_actors, shared_directors,
                     (3 * shared_genres) + (2 * shared_actors) + (4 * shared_directors) AS score
                WHERE score >= $min_score

                RETURN m1.title AS movie_1,
                       m2.title AS movie_2,
                       shared_genres,
                       shared_actors,
                       shared_directors,
                       score
                ORDER BY score DESC, movie_1, movie_2
                """,
                min_score=min_score,
                max_movies=max_movies,
            )

            rows = []
            for record in result:
                graph.add_edge(
                    record["movie_1"],
                    record["movie_2"],
                    weight=record["score"],
                    shared_genres=record["shared_genres"],
                    shared_actors=record["shared_actors"],
                    shared_directors=record["shared_directors"],
                )
                rows.append(dict(record))

        pd.DataFrame(rows).to_csv(RESULTS_DIR / "movie_similarity_projection.csv", index=False)
        return graph


if __name__ == "__main__":
    projections = GraphProjections()
    try:
        actor_graph = projections.create_actor_cooccurrence_graph()
        movie_graph = projections.create_movie_similarity_graph()
        print(
            {
                "actor_nodes": actor_graph.number_of_nodes(),
                "actor_edges": actor_graph.number_of_edges(),
                "movie_nodes": movie_graph.number_of_nodes(),
                "movie_edges": movie_graph.number_of_edges(),
            }
        )
    finally:
        projections.close()
