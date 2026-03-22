"""Main entrypoint for incremental project execution."""
from __future__ import annotations

from neo4j.exceptions import DriverError, Neo4jError, ServiceUnavailable

from src.cypher_queries import GraphExplorer
from src.data_loader import run_phase1_load
from src.feature_extraction import FeatureExtractor
from src.graph_analysis import GraphAnalyzer
from src.link_prediction import LinkPredictor
from src.graph_model import print_schema
from src.projections import GraphProjections
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

    print("\n3) Validating recommendations dump data in Neo4j...")
    try:
        run_phase1_load()

        print("\n4) Running first exploration queries...")
        explorer = GraphExplorer()
        try:
            results = explorer.run_basic_exploration()
        finally:
            explorer.close()

        print("\n--- Node Counts ---")
        print(results["node_counts"])

        print("\n--- Relationship Counts ---")
        print(results["relationship_counts"])

        print("\n--- Top Rated Movies ---")
        print(results["top_rated_movies"])

        print("\n5) Running graph analysis on the core actor subgraph...")
        analyzer = GraphAnalyzer()
        try:
            analyzer.build_actor_cooccurrence_graph()
            degree_df = analyzer.compute_degree_distribution()
            centrality_df = analyzer.compute_centralities()
            community_df = analyzer.detect_communities()
            summary_df = analyzer.get_graph_summary()
        finally:
            analyzer.close()

        print("\n--- Degree Distribution ---")
        print(degree_df.head())

        print("\n--- Centralities ---")
        print(centrality_df.head())

        print("\n--- Communities ---")
        print(community_df.head())

        print("\n--- Graph Summary ---")
        print(summary_df)

        print("\n6) Extracting manual actor features from the core actor subgraph...")
        extractor = FeatureExtractor()
        try:
            actor_features_df = extractor.extract_actor_features()
            extractor.save_actor_features_to_neo4j(actor_features_df)
        finally:
            extractor.close()

        print("\n--- Actor Features ---")
        print(actor_features_df.head())

        print("\n7) Building graph projections...")
        projections = GraphProjections()
        try:
            actor_graph = projections.create_actor_cooccurrence_graph()
            movie_graph = projections.create_movie_similarity_graph()
        finally:
            projections.close()

        print(
            {
                "actor_projection_nodes": actor_graph.number_of_nodes(),
                "actor_projection_edges": actor_graph.number_of_edges(),
                "movie_similarity_nodes": movie_graph.number_of_nodes(),
                "movie_similarity_edges": movie_graph.number_of_edges(),
            }
        )

        print("\n8) Running link prediction on the actor projection...")
        predictor = LinkPredictor(actor_graph)
        predictor.prepare_dataset()
        link_comparison_df = predictor.run_experiments()
        link_rfe_df = predictor.recursive_feature_elimination()
        predicted_links_df = predictor.predict_new_links()

        print("\n--- Link Prediction Comparison ---")
        print(link_comparison_df)

        print("\n--- Link Prediction RFE ---")
        print(link_rfe_df.head(10))

        print("\n--- Predicted Actor Links ---")
        print(predicted_links_df.head(10))
    except (Neo4jError, DriverError, ServiceUnavailable) as exc:
        print("\nNeo4j is not reachable or query failed.")
        print("Please check `.env` and Neo4j server status, then run again.")
        print(f"Details: {exc}")

    print("\nMilestone 1 finished.")


if __name__ == "__main__":
    main()
