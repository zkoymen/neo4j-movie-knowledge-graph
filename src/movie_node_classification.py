"""Movie node classification with graph-derived features."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from community import community_louvain
from neo4j import GraphDatabase
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MovieNodeClassifier:
    """Classify single-genre movies from graph-derived movie features."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self.graph: nx.Graph | None = None
        self.dataset: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.best_model = None
        self.best_feature_names: list[str] = []
        self.label_encoder = LabelEncoder()
        self.scaler: StandardScaler | None = None

    def close(self) -> None:
        """Close Neo4j driver."""
        self.driver.close()

    def _query_to_df(self, query: str, **params: object) -> pd.DataFrame:
        """Run a Neo4j query and convert it into a DataFrame."""
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(query, **params)
            return pd.DataFrame([dict(record) for record in result])

    def _query_single_genre_labels(self) -> pd.DataFrame:
        """Keep only movies with one clear genre label."""
        label_df = self._query_to_df(
            """
            MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
            WITH m, collect(g.name) AS genres
            WHERE size(genres) = 1
            RETURN
                coalesce(
                    toString(m.movieId),
                    toString(m.tmdbId),
                    toString(m.imdbId),
                    m.title + ' (' + toString(coalesce(m.year, 0)) + ')'
                ) AS node,
                m.title AS title,
                genres[0] AS dominant_genre
            ORDER BY node
            """
        )

        if label_df.empty:
            raise ValueError("Movie labels are empty. Could not build movie classification task.")

        return label_df

    def build_movie_graph(self, movie_keys: list[str]) -> nx.Graph:
        """Build a movie graph without using genre edges."""
        graph = nx.Graph()

        # We connect movies only through shared actors/directors.
        # This avoids leaking the target genre label into the features.
        edge_df = self._query_to_df(
            """
            MATCH (m1:Movie)<-[:ACTED_IN|DIRECTED]-(p)-[:ACTED_IN|DIRECTED]->(m2:Movie)
            WITH
                m1,
                m2,
                p,
                coalesce(
                    toString(m1.movieId),
                    toString(m1.tmdbId),
                    toString(m1.imdbId),
                    m1.title + ' (' + toString(coalesce(m1.year, 0)) + ')'
                ) AS key_1,
                coalesce(
                    toString(m2.movieId),
                    toString(m2.tmdbId),
                    toString(m2.imdbId),
                    m2.title + ' (' + toString(coalesce(m2.year, 0)) + ')'
                ) AS key_2
            WHERE key_1 < key_2
              AND key_1 IN $movie_keys
              AND key_2 IN $movie_keys
            WITH m1, m2, collect(DISTINCT p) AS shared_people
            RETURN
                   coalesce(
                       toString(m1.movieId),
                       toString(m1.tmdbId),
                       toString(m1.imdbId),
                       m1.title + ' (' + toString(coalesce(m1.year, 0)) + ')'
                   ) AS source,
                   coalesce(
                       toString(m2.movieId),
                       toString(m2.tmdbId),
                       toString(m2.imdbId),
                       m2.title + ' (' + toString(coalesce(m2.year, 0)) + ')'
                   ) AS target,
                   size(shared_people) AS weight
            """,
            movie_keys=movie_keys,
        )

        for row in edge_df.to_dict(orient="records"):
            graph.add_edge(
                row["source"],
                row["target"],
                weight=row["weight"],
            )

        # We also keep isolated movies in the graph.
        for movie_key in movie_keys:
            if movie_key not in graph:
                graph.add_node(movie_key)

        self.graph = graph
        return graph

    def extract_movie_features(self) -> pd.DataFrame:
        """Create graph features for single-genre movies."""
        label_df = self._query_single_genre_labels()
        genre_counts = label_df["dominant_genre"].value_counts()

        # We keep genres with enough support for a multi-class comparison.
        stable_genres = genre_counts.loc[genre_counts >= 50].index.tolist()
        label_df = label_df.loc[label_df["dominant_genre"].isin(stable_genres)].copy()
        label_df = label_df.reset_index(drop=True)

        movie_keys = label_df["node"].tolist()
        self.build_movie_graph(movie_keys)

        degree_series = pd.Series(dict(self.graph.degree()), name="degree")
        degree_centrality = pd.Series(nx.degree_centrality(self.graph), name="degree_centrality")
        closeness = pd.Series(nx.closeness_centrality(self.graph), name="closeness_centrality")
        pagerank = pd.Series(nx.pagerank(self.graph), name="pagerank")
        clustering = pd.Series(nx.clustering(self.graph), name="clustering_coefficient")

        sample_k = min(config.APPROX_BETWEENNESS_K, self.graph.number_of_nodes())
        betweenness = pd.Series(
            nx.betweenness_centrality(
                self.graph,
                k=sample_k,
                seed=config.RANDOM_STATE,
            ),
            name="betweenness_centrality",
        )

        if self.graph.number_of_edges() == 0:
            community_df = pd.DataFrame(
                {
                    "node": list(self.graph.nodes()),
                    "community": 0,
                }
            )
        else:
            community_df = pd.DataFrame(
                {
                    "node": list(community_louvain.best_partition(self.graph).keys()),
                    "community": list(community_louvain.best_partition(self.graph).values()),
                }
            )

        graph_feature_df = pd.concat(
            [
                degree_series,
                degree_centrality,
                betweenness,
                closeness,
                pagerank,
                clustering,
            ],
            axis=1,
        ).reset_index(names="node")

        movie_meta_df = self._query_to_df(
            """
            MATCH (m:Movie)
            WITH m,
                 coalesce(
                     toString(m.movieId),
                     toString(m.tmdbId),
                     toString(m.imdbId),
                     m.title + ' (' + toString(coalesce(m.year, 0)) + ')'
                 ) AS movie_key
            WHERE movie_key IN $movie_keys
            OPTIONAL MATCH (a:Actor)-[:ACTED_IN]->(m)
            OPTIONAL MATCH (d:Director)-[:DIRECTED]->(m)
            OPTIONAL MATCH (:User)-[r:RATED]->(m)
            RETURN movie_key AS node,
                   count(DISTINCT a) AS actor_count,
                   count(DISTINCT d) AS director_count,
                   count(DISTINCT r) AS rating_count,
                   round(avg(r.rating) * 10) / 10.0 AS avg_rating,
                   coalesce(toFloat(m.imdbRating), 0.0) AS imdb_rating,
                   coalesce(toFloat(m.imdbVotes), 0.0) AS imdb_votes
            ORDER BY node
            """,
            movie_keys=movie_keys,
        )

        feature_df = (
            label_df.merge(graph_feature_df, on="node", how="left")
            .merge(community_df, on="node", how="left")
            .merge(movie_meta_df, on="node", how="left")
            .dropna()
            .reset_index(drop=True)
        )

        feature_df.to_csv(RESULTS_DIR / "movie_classification_dataset.csv", index=False)
        (
            feature_df["dominant_genre"]
            .value_counts()
            .rename_axis("genre")
            .reset_index(name="movie_count")
            .to_csv(RESULTS_DIR / "movie_classification_label_distribution.csv", index=False)
        )
        pd.DataFrame(
            [
                {
                    "nodes": self.graph.number_of_nodes(),
                    "edges": self.graph.number_of_edges(),
                    "density": round(nx.density(self.graph), 4),
                    "connected_components": nx.number_connected_components(self.graph),
                }
            ]
        ).to_csv(RESULTS_DIR / "movie_classification_graph_summary.csv", index=False)

        self.dataset = feature_df
        return feature_df

    def prepare_train_test_split(self, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test sets."""
        if self.dataset is None:
            self.extract_movie_features()

        train_df, test_df = train_test_split(
            self.dataset,
            test_size=test_size,
            random_state=config.RANDOM_STATE,
            stratify=self.dataset["dominant_genre"],
        )

        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        self.train_df.to_csv(RESULTS_DIR / "movie_classification_train.csv", index=False)
        self.test_df.to_csv(RESULTS_DIR / "movie_classification_test.csv", index=False)
        return self.train_df, self.test_df

    def _split_features(self):
        """Prepare arrays for scikit-learn."""
        if self.train_df is None or self.test_df is None:
            self.prepare_train_test_split()

        feature_columns = [
            column
            for column in self.train_df.columns
            if column not in {"node", "title", "dominant_genre"}
        ]

        x_train = self.train_df[feature_columns].values
        x_test = self.test_df[feature_columns].values
        y_train = self.label_encoder.fit_transform(self.train_df["dominant_genre"])
        y_test = self.label_encoder.transform(self.test_df["dominant_genre"])
        return x_train, y_train, x_test, y_test, feature_columns

    def run_experiments(self) -> pd.DataFrame:
        """Train three classifiers and compare their scores."""
        x_train, y_train, x_test, y_test, feature_columns = self._split_features()

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=3000, random_state=config.RANDOM_STATE),
                "params": {"C": [0.1, 1.0, 10.0]},
                "use_scaled": True,
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=config.RANDOM_STATE),
                "params": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
                "use_scaled": False,
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
                "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                "use_scaled": False,
            },
        }

        rows = []
        best_score = -1.0

        for model_name, cfg in models.items():
            x_fit = x_train_scaled if cfg["use_scaled"] else x_train
            x_eval = x_test_scaled if cfg["use_scaled"] else x_test

            grid = GridSearchCV(
                estimator=cfg["model"],
                param_grid=cfg["params"],
                scoring="f1_macro",
                cv=3,
                n_jobs=-1,
            )
            grid.fit(x_fit, y_train)
            best_estimator = grid.best_estimator_

            predictions = best_estimator.predict(x_eval)
            accuracy = accuracy_score(y_test, predictions)
            f1_macro = f1_score(y_test, predictions, average="macro")

            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "F1-Macro": f1_macro,
                    "Best Params": str(grid.best_params_),
                }
            )

            if f1_macro > best_score:
                best_score = f1_macro
                self.best_model = best_estimator
                self.best_feature_names = feature_columns

            report_df = pd.DataFrame(
                classification_report(
                    y_test,
                    predictions,
                    labels=list(range(len(self.label_encoder.classes_))),
                    target_names=self.label_encoder.classes_,
                    output_dict=True,
                    zero_division=0,
                )
            ).transpose()
            report_df.to_csv(
                RESULTS_DIR / f"movie_classification_report_{model_name.lower().replace(' ', '_')}.csv",
                index=True,
            )

        comparison_df = pd.DataFrame(rows).sort_values("F1-Macro", ascending=False).reset_index(drop=True)
        comparison_df.to_csv(RESULTS_DIR / "movie_classification_comparison.csv", index=False)
        return comparison_df

    def recursive_feature_elimination(self, n_features_to_select: int = 6) -> pd.DataFrame:
        """Rank movie features with RFE."""
        x_train, y_train, _, _, feature_columns = self._split_features()
        estimator = RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE)
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        selector.fit(x_train, y_train)

        ranking_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Ranking": selector.ranking_,
                "Selected": selector.support_,
            }
        ).sort_values(["Ranking", "Feature"])

        ranking_df.to_csv(RESULTS_DIR / "movie_classification_rfe.csv", index=False)
        return ranking_df

    def predict_test_labels(self) -> pd.DataFrame:
        """Save predictions on the test set."""
        if self.best_model is None:
            self.run_experiments()

        _, _, x_test, _, _ = self._split_features()
        x_eval = x_test
        if isinstance(self.best_model, LogisticRegression) and self.scaler is not None:
            x_eval = self.scaler.transform(x_test)

        predicted_codes = self.best_model.predict(x_eval)
        prediction_df = self.test_df[["node", "title", "dominant_genre"]].copy()
        prediction_df["predicted_genre"] = self.label_encoder.inverse_transform(predicted_codes)
        prediction_df["is_correct"] = prediction_df["dominant_genre"] == prediction_df["predicted_genre"]
        prediction_df.to_csv(RESULTS_DIR / "movie_classification_predictions.csv", index=False)
        return prediction_df


if __name__ == "__main__":
    classifier = MovieNodeClassifier()
    try:
        classifier.extract_movie_features()
        classifier.prepare_train_test_split()
        print(classifier.run_experiments())
        print(classifier.recursive_feature_elimination())
        print(classifier.predict_test_labels().head(10))
    finally:
        classifier.close()
