"""Manual graph feature extraction for Phase 2."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

import config
from src.graph_analysis import GraphAnalyzer

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class FeatureExtractor:
    """Create actor-level manual features and save them."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def _query_to_df(self, query: str, **params: object) -> pd.DataFrame:
        """Run query and convert rows to DataFrame."""
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(query, **params)
            return pd.DataFrame([dict(record) for record in result])

    def extract_actor_features(
        self,
        min_movie_count: int | None = None,
        max_actors: int | None = None,
        output_name: str = "actor_features.csv",
        save_analysis_outputs: bool = True,
        analysis_output_prefix: str = "",
    ) -> pd.DataFrame:
        """
        Build a feature table for actors.

        We keep this method configurable because the project now needs
        two different actor scopes:
        - a report-friendly core graph
        - a broader set for node classification
        """
        if min_movie_count is None:
            min_movie_count = config.CORE_ACTOR_MIN_MOVIES
        if max_actors is None:
            max_actors = config.CORE_ACTOR_MAX_ACTORS

        analyzer = GraphAnalyzer(
            save_outputs=save_analysis_outputs,
            output_prefix=analysis_output_prefix,
        )
        try:
            # First we build the actor graph for the requested scope.
            analyzer.build_actor_cooccurrence_graph(
                min_movie_count=min_movie_count,
                max_actors=max_actors,
            )
            degree_df = analyzer.compute_degree_distribution()
            centrality_df = analyzer.compute_centralities()
            community_df = analyzer.detect_communities()
            analyzer.get_graph_summary()
            core_actor_nodes = analyzer.core_actor_nodes
        finally:
            analyzer.close()

        # Then we collect simple movie-side features for the same actor set.
        movie_feature_df = self._query_to_df(
            """
            MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
            WHERE a.name IN $actor_names
            OPTIONAL MATCH (a)-[:ACTED_IN]->(:Movie)<-[:DIRECTED]-(d:Director)
            OPTIONAL MATCH (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g:Genre)
            RETURN a.name AS node,
                   count(DISTINCT m) AS movie_count,
                   round(avg(coalesce(m.imdbRating, m.rating)) * 10) / 10.0 AS avg_movie_rating,
                   count(DISTINCT d) AS director_count,
                   count(DISTINCT g) AS genre_diversity
            ORDER BY node
            """,
            actor_names=core_actor_nodes,
        )

        # Last we merge all manual features into one ML-ready table.
        features_df = (
            degree_df.merge(centrality_df, on="node", how="left")
            .merge(community_df, on="node", how="left")
            .merge(movie_feature_df, on="node", how="left")
            .sort_values("node")
            .reset_index(drop=True)
        )

        features_df.to_csv(RESULTS_DIR / output_name, index=False)
        return features_df

    def save_actor_features_to_neo4j(self, features_df: pd.DataFrame) -> None:
        """Write extracted features back to Actor nodes."""
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            for row in features_df.to_dict(orient="records"):
                session.run(
                    """
                    MATCH (a:Actor {name: $name})
                    SET a.degree = $degree,
                        a.degree_centrality = $degree_centrality,
                        a.betweenness_centrality = $betweenness_centrality,
                        a.closeness_centrality = $closeness_centrality,
                        a.pagerank = $pagerank,
                        a.community = $community,
                        a.movie_count = $movie_count,
                        a.avg_movie_rating = $avg_movie_rating,
                        a.director_count = $director_count,
                        a.genre_diversity = $genre_diversity
                    """,
                    name=row["node"],
                    degree=int(row["degree"]),
                    degree_centrality=float(row["degree_centrality"]),
                    betweenness_centrality=float(row["betweenness_centrality"]),
                    closeness_centrality=float(row["closeness_centrality"]),
                    pagerank=float(row["pagerank"]),
                    community=int(row["community"]),
                    movie_count=int(row["movie_count"]),
                    avg_movie_rating=float(row["avg_movie_rating"]),
                    director_count=int(row["director_count"]),
                    genre_diversity=int(row["genre_diversity"]),
                )


if __name__ == "__main__":
    extractor = FeatureExtractor()
    try:
        actor_features = extractor.extract_actor_features()
        extractor.save_actor_features_to_neo4j(actor_features)
        print(actor_features)
    finally:
        extractor.close()
