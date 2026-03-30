"""Simple ReFeX-style feature expansion and evaluation."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

import config
from src.feature_extraction import FeatureExtractor
from src.graph_analysis import GraphAnalyzer
from src.node_classification import NodeClassifier

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class RefexFeatureEngineer:
    """Build recursive neighborhood features for the broader actor graph."""

    FEATURE_FILE = "refex_actor_features.csv"

    def __init__(self) -> None:
        self.graph: nx.Graph | None = None
        self.base_features: pd.DataFrame | None = None
        self.refex_features: pd.DataFrame | None = None

    def _ensure_base_features(self) -> pd.DataFrame:
        """Load or build the broader actor feature table first."""
        path = RESULTS_DIR / "actor_features_classification.csv"
        if not path.exists():
            extractor = FeatureExtractor()
            try:
                features_df = extractor.extract_actor_features(
                    min_movie_count=config.CLASSIFICATION_ACTOR_MIN_MOVIES,
                    max_actors=config.CLASSIFICATION_ACTOR_MAX_ACTORS,
                    output_name="actor_features_classification.csv",
                    save_analysis_outputs=True,
                    analysis_output_prefix="classification_",
                )
                extractor.save_actor_features_to_neo4j(features_df)
            finally:
                extractor.close()

        feature_df = pd.read_csv(path)

        # Clustering adds one more local structural signal for ReFeX.
        analyzer = GraphAnalyzer(save_outputs=False)
        try:
            graph = analyzer.build_actor_cooccurrence_graph(
                min_movie_count=config.CLASSIFICATION_ACTOR_MIN_MOVIES,
                max_actors=config.CLASSIFICATION_ACTOR_MAX_ACTORS,
            )
        finally:
            analyzer.close()

        clustering_df = pd.DataFrame(
            {
                "node": list(graph.nodes()),
                "clustering_coefficient": list(nx.clustering(graph).values()),
            }
        )

        self.graph = graph
        self.base_features = (
            feature_df.merge(clustering_df, on="node", how="left")
            .sort_values("node")
            .reset_index(drop=True)
        )
        return self.base_features

    def build_refex_features(self, iterations: int = 2) -> pd.DataFrame:
        """Create recursive neighborhood aggregation features."""
        if self.base_features is None or self.graph is None:
            self._ensure_base_features()

        feature_df = self.base_features.copy()

        # We start from the stronger local features already extracted.
        working_columns = [
            "degree",
            "betweenness_centrality",
            "closeness_centrality",
            "pagerank",
            "movie_count",
            "avg_movie_rating",
            "director_count",
            "genre_diversity",
            "clustering_coefficient",
        ]

        feature_metadata = []

        for iteration in range(1, iterations + 1):
            # We only expand the feature set that exists at the start of this round.
            current_columns = working_columns.copy()
            new_feature_rows = []

            for node in feature_df["node"]:
                neighbors = list(self.graph.neighbors(node)) if self.graph.has_node(node) else []
                row = {"node": node}

                for column in current_columns:
                    if not neighbors:
                        row[f"refex_{iteration}_mean_{column}"] = 0.0
                        row[f"refex_{iteration}_sum_{column}"] = 0.0
                        continue

                    neighbor_values = (
                        feature_df.loc[feature_df["node"].isin(neighbors), column]
                        .fillna(0.0)
                        .astype(float)
                    )
                    row[f"refex_{iteration}_mean_{column}"] = float(neighbor_values.mean())
                    row[f"refex_{iteration}_sum_{column}"] = float(neighbor_values.sum())

                new_feature_rows.append(row)

            new_feature_df = pd.DataFrame(new_feature_rows)
            feature_df = feature_df.merge(new_feature_df, on="node", how="left")

            for column in current_columns:
                feature_metadata.append(
                    {
                        "iteration": iteration,
                        "source_feature": column,
                        "new_feature": f"refex_{iteration}_mean_{column}",
                        "aggregation": "mean",
                    }
                )
                feature_metadata.append(
                    {
                        "iteration": iteration,
                        "source_feature": column,
                        "new_feature": f"refex_{iteration}_sum_{column}",
                        "aggregation": "sum",
                    }
                )

            new_columns = [column for column in new_feature_df.columns if column != "node"]
            working_columns.extend(new_columns)

        # We drop flat columns and near-duplicate columns to keep the set usable.
        numeric_df = feature_df.drop(columns=["node"]).copy()
        non_constant_columns = [
            column for column in numeric_df.columns if numeric_df[column].nunique(dropna=False) > 1
        ]
        numeric_df = numeric_df[non_constant_columns]

        correlation_matrix = numeric_df.corr().abs()
        columns_to_drop: set[str] = set()
        for i, column in enumerate(correlation_matrix.columns):
            for other_column in correlation_matrix.columns[:i]:
                if correlation_matrix.loc[column, other_column] > 0.98:
                    columns_to_drop.add(column)
                    break

        numeric_df = numeric_df.drop(columns=sorted(columns_to_drop), errors="ignore")
        refex_df = pd.concat([feature_df[["node"]], numeric_df], axis=1)

        refex_df.to_csv(RESULTS_DIR / self.FEATURE_FILE, index=False)
        pd.DataFrame(feature_metadata).to_csv(RESULTS_DIR / "refex_feature_metadata.csv", index=False)
        pd.DataFrame(
            [
                {
                    "nodes": len(refex_df),
                    "features": len(refex_df.columns) - 1,
                    "iterations": iterations,
                }
            ]
        ).to_csv(RESULTS_DIR / "refex_feature_summary.csv", index=False)

        self.refex_features = refex_df
        return refex_df


class RefexNodeClassificationExperiment:
    """Run node classification on the ReFeX feature table."""

    def run(self) -> dict[str, pd.DataFrame]:
        """Build features and evaluate three models."""
        engineer = RefexFeatureEngineer()
        engineer.build_refex_features()

        classifier = NodeClassifier(
            feature_file=RefexFeatureEngineer.FEATURE_FILE,
            output_prefix="refex_",
            auto_build_features=False,
        )
        try:
            dataset_df = classifier.build_dataset()
            train_df, test_df = classifier.prepare_train_test_split()
            comparison_df = classifier.run_experiments()
            ranking_df = classifier.recursive_feature_elimination()
            prediction_df = classifier.predict_test_labels()
        finally:
            classifier.close()

        return {
            "dataset": dataset_df,
            "train": train_df,
            "test": test_df,
            "comparison": comparison_df,
            "ranking": ranking_df,
            "predictions": prediction_df,
        }


if __name__ == "__main__":
    experiment = RefexNodeClassificationExperiment()
    outputs = experiment.run()
    print(outputs["comparison"])
    print(outputs["ranking"].head(10))
