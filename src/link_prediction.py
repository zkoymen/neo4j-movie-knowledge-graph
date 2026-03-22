"""Link prediction pipeline for actor collaborations."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

import config

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class LinkPredictor:
    """Predict missing actor-actor collaboration links."""

    def __init__(self, graph: nx.Graph, random_state: int | None = None) -> None:
        self.graph = graph.copy()
        self.random_state = config.RANDOM_STATE if random_state is None else random_state
        self.train_graph: nx.Graph | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.best_model = None
        self.best_feature_names: list[str] = []
        self.scaler: StandardScaler | None = None

    def _sample_non_edges(self, graph: nx.Graph, count: int) -> list[tuple[str, str]]:
        """Sample random non-edges without materializing all possible pairs."""
        rng = random.Random(self.random_state)
        nodes = list(graph.nodes())
        sampled: set[tuple[str, str]] = set()
        max_attempts = max(count * 50, 1000)
        attempts = 0

        while len(sampled) < count and attempts < max_attempts:
            source, target = rng.sample(nodes, 2)
            edge = tuple(sorted((source, target)))
            attempts += 1
            if source == target or graph.has_edge(*edge) or edge in sampled:
                continue
            sampled.add(edge)

        if len(sampled) < count:
            raise RuntimeError("Could not sample enough non-edges for link prediction.")

        return list(sampled)

    def _choose_test_edges(self, test_fraction: float) -> list[tuple[str, str]]:
        """
        Choose removable edges for the test set.

        We avoid bridge edges first so the training graph keeps more structure.
        """
        edges = sorted(tuple(sorted(edge)) for edge in self.graph.edges())
        bridges = {tuple(sorted(edge)) for edge in nx.bridges(self.graph)}
        removable = [edge for edge in edges if edge not in bridges]

        if len(removable) < 10:
            removable = edges

        rng = random.Random(self.random_state)
        test_size = max(1, int(len(removable) * test_fraction))
        return rng.sample(removable, test_size)

    def _graph_node_features(self, graph: nx.Graph) -> tuple[dict, dict]:
        """Compute node-level helper maps on a given graph."""
        partition = (
            community_louvain.best_partition(graph)
            if graph.number_of_edges() > 0
            else {node: 0 for node in graph.nodes()}
        )

        if graph.number_of_nodes() == 0:
            return {}, partition

        betweenness_k = min(config.APPROX_BETWEENNESS_K, graph.number_of_nodes())

        features = {
            "degree": dict(graph.degree()),
            "degree_centrality": nx.degree_centrality(graph),
            "betweenness_centrality": nx.betweenness_centrality(
                graph,
                k=betweenness_k,
                seed=self.random_state,
            ),
            "closeness_centrality": nx.closeness_centrality(graph),
            "pagerank": nx.pagerank(graph) if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()},
        }
        return features, partition

    def _pair_features(
        self,
        graph: nx.Graph,
        edges: Iterable[tuple[str, str]],
        label: int,
        node_features: dict | None = None,
        partition: dict | None = None,
    ) -> pd.DataFrame:
        """Create manual topological features for candidate links."""
        if node_features is None or partition is None:
            node_features, partition = self._graph_node_features(graph)
        rows = []

        for source, target in edges:
            common_neighbors = len(list(nx.common_neighbors(graph, source, target)))
            jaccard = next(nx.jaccard_coefficient(graph, [(source, target)]))[2]
            adamic_adar = next(nx.adamic_adar_index(graph, [(source, target)]))[2]
            pref_attach = next(nx.preferential_attachment(graph, [(source, target)]))[2]
            resource_alloc = next(nx.resource_allocation_index(graph, [(source, target)]))[2]

            src_degree = node_features["degree"].get(source, 0)
            tgt_degree = node_features["degree"].get(target, 0)

            rows.append(
                {
                    "source": source,
                    "target": target,
                    "source_degree": src_degree,
                    "target_degree": tgt_degree,
                    "degree_sum": src_degree + tgt_degree,
                    "degree_diff": abs(src_degree - tgt_degree),
                    "source_degree_centrality": node_features["degree_centrality"].get(source, 0.0),
                    "target_degree_centrality": node_features["degree_centrality"].get(target, 0.0),
                    "source_betweenness": node_features["betweenness_centrality"].get(source, 0.0),
                    "target_betweenness": node_features["betweenness_centrality"].get(target, 0.0),
                    "source_closeness": node_features["closeness_centrality"].get(source, 0.0),
                    "target_closeness": node_features["closeness_centrality"].get(target, 0.0),
                    "source_pagerank": node_features["pagerank"].get(source, 0.0),
                    "target_pagerank": node_features["pagerank"].get(target, 0.0),
                    "same_community": int(partition.get(source, -1) == partition.get(target, -2)),
                    "common_neighbors": common_neighbors,
                    "jaccard": jaccard,
                    "adamic_adar": adamic_adar,
                    "preferential_attachment": pref_attach,
                    "resource_allocation": resource_alloc,
                    "label": label,
                }
            )

        return pd.DataFrame(rows)

    def prepare_dataset(self, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build train/test datasets from positive and negative edge samples."""
        test_positive_edges = self._choose_test_edges(test_fraction)
        self.train_graph = self.graph.copy()
        self.train_graph.remove_edges_from(test_positive_edges)

        train_positive_edges = [tuple(sorted(edge)) for edge in self.train_graph.edges()]
        train_negative_edges = self._sample_non_edges(self.train_graph, len(train_positive_edges))
        test_negative_edges = self._sample_non_edges(self.train_graph, len(test_positive_edges))

        node_features, partition = self._graph_node_features(self.train_graph)

        train_positive_df = self._pair_features(
            self.train_graph,
            train_positive_edges,
            label=1,
            node_features=node_features,
            partition=partition,
        )
        train_negative_df = self._pair_features(
            self.train_graph,
            train_negative_edges,
            label=0,
            node_features=node_features,
            partition=partition,
        )
        test_positive_df = self._pair_features(
            self.train_graph,
            test_positive_edges,
            label=1,
            node_features=node_features,
            partition=partition,
        )
        test_negative_df = self._pair_features(
            self.train_graph,
            test_negative_edges,
            label=0,
            node_features=node_features,
            partition=partition,
        )

        self.train_df = pd.concat([train_positive_df, train_negative_df], ignore_index=True)
        self.test_df = pd.concat([test_positive_df, test_negative_df], ignore_index=True)

        self.train_df.to_csv(RESULTS_DIR / "link_prediction_train.csv", index=False)
        self.test_df.to_csv(RESULTS_DIR / "link_prediction_test.csv", index=False)
        return self.train_df, self.test_df

    def _split_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Prepare numeric arrays for scikit-learn."""
        if self.train_df is None or self.test_df is None:
            self.prepare_dataset()

        feature_columns = [
            column
            for column in self.train_df.columns
            if column not in {"source", "target", "label"}
        ]

        x_train = self.train_df[feature_columns].values
        y_train = self.train_df["label"].values
        x_test = self.test_df[feature_columns].values
        y_test = self.test_df["label"].values
        return x_train, y_train, x_test, y_test, feature_columns

    def run_experiments(self) -> pd.DataFrame:
        """Train three ML models with GridSearchCV."""
        x_train, y_train, x_test, y_test, feature_columns = self._split_features()

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "params": {"C": [0.1, 1.0, 10.0]},
                "use_scaled": True,
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=self.random_state),
                "params": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
                "use_scaled": False,
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=self.random_state),
                "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                "use_scaled": False,
            },
        }

        rows = []
        best_auc = -1.0

        for model_name, cfg in models.items():
            x_fit = x_train_scaled if cfg["use_scaled"] else x_train
            x_eval = x_test_scaled if cfg["use_scaled"] else x_test

            grid = GridSearchCV(
                estimator=cfg["model"],
                param_grid=cfg["params"],
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
            )
            grid.fit(x_fit, y_train)
            best_estimator = grid.best_estimator_

            probabilities = best_estimator.predict_proba(x_eval)[:, 1]
            predictions = best_estimator.predict(x_eval)
            auc = roc_auc_score(y_test, probabilities)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            rows.append(
                {
                    "Model": model_name,
                    "AUC-ROC": auc,
                    "Accuracy": accuracy,
                    "F1": f1,
                    "Best Params": str(grid.best_params_),
                }
            )

            if auc > best_auc:
                best_auc = auc
                self.best_model = best_estimator
                self.best_feature_names = feature_columns

            report_df = pd.DataFrame(
                classification_report(y_test, predictions, output_dict=True)
            ).transpose()
            report_df.to_csv(
                RESULTS_DIR / f"link_prediction_report_{model_name.lower().replace(' ', '_')}.csv",
                index=True,
            )

        comparison_df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
        comparison_df.to_csv(RESULTS_DIR / "link_prediction_comparison.csv", index=False)
        return comparison_df

    def recursive_feature_elimination(self, n_features_to_select: int = 6) -> pd.DataFrame:
        """Rank link prediction features using RFE."""
        x_train, y_train, _, _, feature_columns = self._split_features()
        estimator = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        selector.fit(x_train, y_train)

        ranking_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Ranking": selector.ranking_,
                "Selected": selector.support_,
            }
        ).sort_values(["Ranking", "Feature"])

        ranking_df.to_csv(RESULTS_DIR / "link_prediction_rfe.csv", index=False)
        return ranking_df

    def predict_new_links(
        self,
        top_n: int = 20,
        candidate_actor_count: int | None = None,
    ) -> pd.DataFrame:
        """Score candidate missing links among the most connected actors."""
        if self.best_model is None:
            self.run_experiments()

        if candidate_actor_count is None:
            candidate_actor_count = config.LINK_PREDICTION_MAX_ACTORS

        degrees = sorted(self.graph.degree(), key=lambda item: (-item[1], item[0]))
        candidate_nodes = [node for node, _ in degrees[:candidate_actor_count]]

        candidate_edges = []
        for index, source in enumerate(candidate_nodes):
            for target in candidate_nodes[index + 1 :]:
                if self.graph.has_edge(source, target):
                    continue
                candidate_edges.append((source, target))

        if not candidate_edges:
            empty_df = pd.DataFrame(columns=["source", "target", "score"])
            empty_df.to_csv(RESULTS_DIR / "predicted_actor_links.csv", index=False)
            return empty_df

        node_features, partition = self._graph_node_features(self.graph)
        feature_df = self._pair_features(
            self.graph,
            candidate_edges,
            label=0,
            node_features=node_features,
            partition=partition,
        )
        x_candidates = feature_df[self.best_feature_names].values

        if isinstance(self.best_model, LogisticRegression) and self.scaler is not None:
            x_candidates = self.scaler.transform(x_candidates)

        scores = self.best_model.predict_proba(x_candidates)[:, 1]
        prediction_df = feature_df[["source", "target"]].copy()
        prediction_df["score"] = scores
        prediction_df = prediction_df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        prediction_df.to_csv(RESULTS_DIR / "predicted_actor_links.csv", index=False)
        return prediction_df


if __name__ == "__main__":
    from src.projections import GraphProjections

    projections = GraphProjections()
    try:
        actor_graph = projections.create_actor_cooccurrence_graph()
    finally:
        projections.close()

    predictor = LinkPredictor(actor_graph)
    predictor.prepare_dataset()
    print(predictor.run_experiments())
    print(predictor.recursive_feature_elimination())
    print(predictor.predict_new_links())
