"""Graph analysis utilities for Phase 2."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from community import community_louvain
from neo4j import GraphDatabase

import config

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class GraphAnalyzer:
    """Build simple graph projections and calculate topology metrics."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self.graph: nx.Graph | None = None
        self.core_actor_nodes: list[str] = []
        self.core_actor_meta: pd.DataFrame | None = None

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def build_actor_cooccurrence_graph(
        self,
        min_movie_count: int | None = None,
        max_actors: int | None = None,
    ) -> nx.Graph:
        """
        Create an undirected graph where actors connect by shared movies.

        We use a core actor subgraph to keep the analysis meaningful and tractable.
        Low-activity actors create a very long sparse tail and slow down topology metrics.
        """
        if min_movie_count is None:
            min_movie_count = config.CORE_ACTOR_MIN_MOVIES
        if max_actors is None:
            max_actors = config.CORE_ACTOR_MAX_ACTORS

        graph = nx.Graph()

        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            actor_result = session.run(
                """
                MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
                WITH a.name AS actor, count(DISTINCT m) AS movie_count
                WHERE movie_count >= $min_movie_count
                RETURN actor, movie_count
                ORDER BY movie_count DESC, actor
                LIMIT $max_actors
                """,
                min_movie_count=min_movie_count,
                max_actors=max_actors,
            )
            actor_rows = [dict(record) for record in actor_result]
            self.core_actor_nodes = [row["actor"] for row in actor_rows]
            self.core_actor_meta = pd.DataFrame(actor_rows)

            result = session.run(
                """
                MATCH (a1:Actor)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Actor)
                WHERE a1.name < a2.name
                  AND a1.name IN $actor_names
                  AND a2.name IN $actor_names
                RETURN a1.name AS source,
                       a2.name AS target,
                       count(m) AS weight,
                       collect(m.title) AS movies
                """,
                actor_names=self.core_actor_nodes,
            )

            for record in result:
                graph.add_edge(
                    record["source"],
                    record["target"],
                    weight=record["weight"],
                    movies=record["movies"],
                )

        self.graph = graph
        return graph

    def compute_degree_distribution(self) -> pd.DataFrame:
        """Calculate degree for each actor in the projection graph."""
        if self.graph is None:
            self.build_actor_cooccurrence_graph()

        degrees = dict(self.graph.degree())
        degree_df = pd.DataFrame(
            {
                "node": list(degrees.keys()),
                "degree": list(degrees.values()),
            }
        ).sort_values(["degree", "node"], ascending=[False, True])

        degree_df.to_csv(RESULTS_DIR / "degree_distribution.csv", index=False)
        return degree_df

    def compute_centralities(self) -> pd.DataFrame:
        """Calculate centrality metrics on the actor graph."""
        if self.graph is None:
            self.build_actor_cooccurrence_graph()

        if self.graph.number_of_nodes() == 0:
            empty_df = pd.DataFrame(
                columns=[
                    "node",
                    "degree_centrality",
                    "betweenness_centrality",
                    "closeness_centrality",
                    "pagerank",
                ]
            )
            empty_df.to_csv(RESULTS_DIR / "centralities.csv", index=False)
            return empty_df

        # Exact betweenness is too slow on this dataset.
        # We use approximation with k sampled source nodes.
        sample_k = min(config.APPROX_BETWEENNESS_K, self.graph.number_of_nodes())

        centrality_df = pd.DataFrame(
            {
                "node": list(self.graph.nodes()),
                "degree_centrality": pd.Series(nx.degree_centrality(self.graph)),
                "betweenness_centrality": pd.Series(
                    nx.betweenness_centrality(
                        self.graph,
                        k=sample_k,
                        seed=config.RANDOM_STATE,
                    )
                ),
                "closeness_centrality": pd.Series(nx.closeness_centrality(self.graph)),
                "pagerank": pd.Series(nx.pagerank(self.graph)),
            }
        ).sort_values(["degree_centrality", "node"], ascending=[False, True])

        centrality_df.to_csv(RESULTS_DIR / "centralities.csv", index=False)
        return centrality_df

    def detect_communities(self) -> pd.DataFrame:
        """Detect communities with Louvain."""
        if self.graph is None:
            self.build_actor_cooccurrence_graph()

        if self.graph.number_of_edges() == 0:
            community_df = pd.DataFrame(
                {
                    "node": list(self.graph.nodes()),
                    "community": [0 for _ in self.graph.nodes()],
                }
            )
        else:
            partition = community_louvain.best_partition(self.graph)
            community_df = pd.DataFrame(
                {
                    "node": list(partition.keys()),
                    "community": list(partition.values()),
                }
            ).sort_values(["community", "node"])

        community_df.to_csv(RESULTS_DIR / "communities.csv", index=False)
        return community_df

    def get_graph_summary(self) -> pd.DataFrame:
        """Return basic graph statistics as one-row DataFrame."""
        if self.graph is None:
            self.build_actor_cooccurrence_graph()

        if self.graph.number_of_nodes() == 0:
            summary = {
                "nodes": 0,
                "edges": 0,
                "density": 0.0,
                "connected_components": 0,
            }
        else:
            summary = {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": round(nx.density(self.graph), 4),
                "connected_components": nx.number_connected_components(self.graph),
                "min_movie_count_filter": config.CORE_ACTOR_MIN_MOVIES,
                "max_actor_filter": config.CORE_ACTOR_MAX_ACTORS,
                "betweenness_k": config.APPROX_BETWEENNESS_K,
            }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(RESULTS_DIR / "graph_summary.csv", index=False)
        return summary_df


if __name__ == "__main__":
    analyzer = GraphAnalyzer()
    try:
        analyzer.build_actor_cooccurrence_graph()
        print(analyzer.compute_degree_distribution())
        print(analyzer.compute_centralities())
        print(analyzer.detect_communities())
        print(analyzer.get_graph_summary())
    finally:
        analyzer.close()
