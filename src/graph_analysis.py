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

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def build_actor_cooccurrence_graph(self) -> nx.Graph:
        """Create an undirected graph where actors connect by shared movies."""
        graph = nx.Graph()

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a1:Actor)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Actor)
                WHERE a1.name < a2.name
                RETURN a1.name AS source,
                       a2.name AS target,
                       count(m) AS weight,
                       collect(m.title) AS movies
                """
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

        centrality_df = pd.DataFrame(
            {
                "node": list(self.graph.nodes()),
                "degree_centrality": pd.Series(nx.degree_centrality(self.graph)),
                "betweenness_centrality": pd.Series(nx.betweenness_centrality(self.graph)),
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
