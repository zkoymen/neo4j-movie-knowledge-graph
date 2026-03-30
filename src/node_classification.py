"""Node classification pipeline for dominant actor genre prediction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
from src.feature_extraction import FeatureExtractor

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class NodeClassifier:
    """Classify actors by their dominant movie genre."""

    DEFAULT_FEATURE_FILE = "actor_features_classification.csv"

    def __init__(
        self,
        feature_file: str | None = None,
        output_prefix: str = "",
        auto_build_features: bool = True,
    ) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self.feature_file = feature_file or self.DEFAULT_FEATURE_FILE
        self.output_prefix = output_prefix
        self.auto_build_features = auto_build_features
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

    def _result_path(self, filename: str) -> Path:
        """Build a result file path for this classifier run."""
        if self.output_prefix:
            return RESULTS_DIR / f"{self.output_prefix}{filename}"
        return RESULTS_DIR / filename

    def _load_actor_features(self) -> pd.DataFrame:
        """Load or build a broader actor feature table for classification."""
        feature_path = RESULTS_DIR / self.feature_file
        if not feature_path.exists():
            if not self.auto_build_features:
                raise FileNotFoundError(
                    f"{self.feature_file} is missing and auto-build is disabled."
                )

            extractor = FeatureExtractor()
            try:
                # Node classification needs broader coverage than the old 1495-actor core.
                features_df = extractor.extract_actor_features(
                    min_movie_count=config.CLASSIFICATION_ACTOR_MIN_MOVIES,
                    max_actors=config.CLASSIFICATION_ACTOR_MAX_ACTORS,
                    output_name=self.feature_file,
                    save_analysis_outputs=True,
                    analysis_output_prefix="classification_",
                )
                extractor.save_actor_features_to_neo4j(features_df)
            finally:
                extractor.close()

        return pd.read_csv(feature_path)

    def _query_labels(self) -> pd.DataFrame:
        """Create supervised labels from the graph."""
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (a:Actor)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g:Genre)
                WITH a, g.name AS genre, count(*) AS genre_freq
                ORDER BY a.name, genre_freq DESC, genre
                WITH a, collect({genre: genre, freq: genre_freq}) AS genre_stats
                WITH a,
                     genre_stats[0].genre AS dominant_genre,
                     genre_stats[0].freq AS top_freq,
                     reduce(total = 0, item IN genre_stats | total + item.freq) AS total_freq
                WHERE total_freq >= $min_actor_genre_links
                  AND (top_freq * 1.0 / total_freq) >= $min_purity
                RETURN a.name AS node,
                       dominant_genre,
                       top_freq,
                       total_freq,
                       (top_freq * 1.0 / total_freq) AS purity
                ORDER BY node
                """,
                min_actor_genre_links=config.NODE_CLASSIFICATION_MIN_ACTOR_GENRE_LINKS,
                min_purity=config.NODE_CLASSIFICATION_MIN_PURITY,
            )
            label_df = pd.DataFrame([dict(record) for record in result])

        if label_df.empty:
            raise ValueError("Node classification labels could not be created from the graph.")

        return label_df

    def build_dataset(self) -> pd.DataFrame:
        """Merge manual features with dominant genre labels."""
        feature_df = self._load_actor_features()
        label_df = self._query_labels()
        merged_df = feature_df.merge(label_df, on="node", how="inner").copy()

        # We do not want one or two-sample classes in 3-fold CV.
        # Instead of keeping every tiny label, we keep stable classes
        # and move the sparse tail into a single "Other" class.
        genre_counts = merged_df["dominant_genre"].value_counts()
        stable_genres = genre_counts.loc[genre_counts >= 15].index.tolist()
        top_genres = stable_genres[: config.NODE_CLASSIFICATION_TOP_GENRES]

        dataset = merged_df.copy()
        dataset["dominant_genre"] = dataset["dominant_genre"].where(
            dataset["dominant_genre"].isin(top_genres),
            "Other",
        )

        if dataset.empty:
            raise ValueError("Node classification dataset is empty after filtering top genres.")

        # We keep all numeric columns except the helper label columns.
        # This makes the classifier reusable for both manual features and ReFeX features.
        feature_columns = [
            column
            for column in dataset.columns
            if column not in {"node", "dominant_genre", "top_freq", "total_freq"}
            and pd.api.types.is_numeric_dtype(dataset[column])
        ]
        dataset = dataset[["node", "dominant_genre", *feature_columns]].dropna().reset_index(drop=True)

        final_counts = dataset["dominant_genre"].value_counts()
        dataset = dataset.loc[dataset["dominant_genre"].isin(final_counts.loc[final_counts >= 10].index)].copy()
        dataset = dataset.reset_index(drop=True)

        dataset.to_csv(self._result_path("node_classification_dataset.csv"), index=False)
        genre_counts = (
            dataset["dominant_genre"]
            .value_counts()
            .rename_axis("genre")
            .reset_index(name="actor_count")
        )
        genre_counts.to_csv(
            self._result_path("node_classification_label_distribution.csv"),
            index=False,
        )

        self.dataset = dataset
        return dataset

    def prepare_train_test_split(
        self,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create train and test splits."""
        if self.dataset is None:
            self.build_dataset()

        train_df, test_df = train_test_split(
            self.dataset,
            test_size=test_size,
            random_state=config.RANDOM_STATE,
            stratify=self.dataset["dominant_genre"],
        )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df.to_csv(self._result_path("node_classification_train.csv"), index=False)
        test_df.to_csv(self._result_path("node_classification_test.csv"), index=False)

        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df

    def _split_features(self):
        """Prepare arrays for scikit-learn."""
        if self.train_df is None or self.test_df is None:
            self.prepare_train_test_split()

        feature_columns = [
            column
            for column in self.train_df.columns
            if column not in {"node", "dominant_genre"}
        ]

        x_train = self.train_df[feature_columns].values
        x_test = self.test_df[feature_columns].values

        y_train = self.label_encoder.fit_transform(self.train_df["dominant_genre"])
        y_test = self.label_encoder.transform(self.test_df["dominant_genre"])

        return x_train, y_train, x_test, y_test, feature_columns

    def run_experiments(self) -> pd.DataFrame:
        """Train three classifiers and compare them."""
        x_train, y_train, x_test, y_test, feature_columns = self._split_features()

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        models = {
            "Logistic Regression": {
                "model": LogisticRegression(
                    max_iter=2000,
                    random_state=config.RANDOM_STATE,
                ),
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
                self._result_path(
                    f"node_classification_report_{model_name.lower().replace(' ', '_')}.csv"
                ),
                index=True,
            )

        comparison_df = pd.DataFrame(rows).sort_values("F1-Macro", ascending=False).reset_index(drop=True)
        comparison_df.to_csv(self._result_path("node_classification_comparison.csv"), index=False)
        return comparison_df

    def recursive_feature_elimination(self, n_features_to_select: int = 6) -> pd.DataFrame:
        """Rank features with RFE."""
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

        ranking_df.to_csv(self._result_path("node_classification_rfe.csv"), index=False)
        return ranking_df

    def predict_test_labels(self) -> pd.DataFrame:
        """Save predictions on the test set."""
        if self.best_model is None:
            self.run_experiments()

        _, _, x_test, y_test, feature_columns = self._split_features()
        x_eval = x_test
        if isinstance(self.best_model, LogisticRegression) and self.scaler is not None:
            x_eval = self.scaler.transform(x_test)

        predicted_codes = self.best_model.predict(x_eval)
        prediction_df = self.test_df[["node", "dominant_genre"]].copy()
        prediction_df["predicted_genre"] = self.label_encoder.inverse_transform(predicted_codes)
        prediction_df["is_correct"] = prediction_df["dominant_genre"] == prediction_df["predicted_genre"]
        prediction_df.to_csv(self._result_path("node_classification_predictions.csv"), index=False)
        return prediction_df


if __name__ == "__main__":
    classifier = NodeClassifier()
    try:
        classifier.build_dataset()
        classifier.prepare_train_test_split()
        print(classifier.run_experiments())
        print(classifier.recursive_feature_elimination())
        print(classifier.predict_test_labels().head(10))
    finally:
        classifier.close()
