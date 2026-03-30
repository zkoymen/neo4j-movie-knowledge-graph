"""ReFeX-style movie classification experiment."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
from src.movie_node_classification import MovieNodeClassifier

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MovieRefexExperiment:
    """Expand movie features recursively and train classifiers."""

    def __init__(self) -> None:
        self.dataset: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.best_model = None
        self.best_feature_names: list[str] = []
        self.label_encoder = LabelEncoder()
        self.scaler: StandardScaler | None = None

    def build_refex_dataset(self, iterations: int = 1) -> pd.DataFrame:
        """Build a ReFeX-style movie feature table."""
        classifier = MovieNodeClassifier()
        try:
            base_df = classifier.extract_movie_features()
            graph = classifier.graph
        finally:
            classifier.close()

        feature_df = base_df.copy()
        # We do not expand every numeric column.
        # A smaller seed set keeps the recursive explosion under control.
        working_columns = [
            "degree",
            "betweenness_centrality",
            "closeness_centrality",
            "pagerank",
            "clustering_coefficient",
            "actor_count",
            "director_count",
            "rating_count",
            "avg_rating",
            "imdb_rating",
            "imdb_votes",
        ]

        feature_metadata = []

        for iteration in range(1, iterations + 1):
            current_columns = working_columns.copy()
            new_rows = []

            for node in feature_df["node"]:
                neighbors = list(graph.neighbors(node)) if graph.has_node(node) else []
                row = {"node": node}

                for column in current_columns:
                    if not neighbors:
                        row[f"movie_refex_{iteration}_mean_{column}"] = 0.0
                        row[f"movie_refex_{iteration}_sum_{column}"] = 0.0
                        continue

                    neighbor_values = (
                        feature_df.loc[feature_df["node"].isin(neighbors), column]
                        .fillna(0.0)
                        .astype(float)
                    )
                    row[f"movie_refex_{iteration}_mean_{column}"] = float(neighbor_values.mean())
                    row[f"movie_refex_{iteration}_sum_{column}"] = float(neighbor_values.sum())

                new_rows.append(row)

            new_feature_df = pd.DataFrame(new_rows)
            feature_df = feature_df.merge(new_feature_df, on="node", how="left")

            for column in current_columns:
                feature_metadata.append(
                    {
                        "iteration": iteration,
                        "source_feature": column,
                        "new_feature": f"movie_refex_{iteration}_mean_{column}",
                        "aggregation": "mean",
                    }
                )
                feature_metadata.append(
                    {
                        "iteration": iteration,
                        "source_feature": column,
                        "new_feature": f"movie_refex_{iteration}_sum_{column}",
                        "aggregation": "sum",
                    }
                )

            working_columns.extend([column for column in new_feature_df.columns if column != "node"])

        numeric_df = feature_df.drop(columns=["node", "title", "dominant_genre"]).copy()
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
        dataset_df = pd.concat([feature_df[["node", "title", "dominant_genre"]], numeric_df], axis=1)
        dataset_df = dataset_df.fillna(0.0)

        dataset_df.to_csv(RESULTS_DIR / "movie_refex_classification_dataset.csv", index=False)
        pd.DataFrame(feature_metadata).to_csv(RESULTS_DIR / "movie_refex_feature_metadata.csv", index=False)
        pd.DataFrame(
            [
                {
                    "rows": len(dataset_df),
                    "feature_columns": len(dataset_df.columns) - 2,
                    "iterations": iterations,
                }
            ]
        ).to_csv(RESULTS_DIR / "movie_refex_feature_summary.csv", index=False)
        (
            dataset_df["dominant_genre"]
            .value_counts()
            .rename_axis("genre")
            .reset_index(name="movie_count")
            .to_csv(RESULTS_DIR / "movie_refex_label_distribution.csv", index=False)
        )

        self.dataset = dataset_df
        return dataset_df

    def prepare_train_test_split(self, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split ReFeX dataset into train and test."""
        if self.dataset is None:
            self.build_refex_dataset()

        train_df, test_df = train_test_split(
            self.dataset,
            test_size=test_size,
            random_state=config.RANDOM_STATE,
            stratify=self.dataset["dominant_genre"],
        )
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        self.train_df.to_csv(RESULTS_DIR / "movie_refex_train.csv", index=False)
        self.test_df.to_csv(RESULTS_DIR / "movie_refex_test.csv", index=False)
        return self.train_df, self.test_df

    def _split_features(self):
        """Prepare numpy arrays for model training."""
        if self.train_df is None or self.test_df is None:
            self.prepare_train_test_split()

        feature_columns = [
            column for column in self.train_df.columns if column not in {"node", "title", "dominant_genre"}
        ]
        x_train = self.train_df[feature_columns].values
        x_test = self.test_df[feature_columns].values
        y_train = self.label_encoder.fit_transform(self.train_df["dominant_genre"])
        y_test = self.label_encoder.transform(self.test_df["dominant_genre"])
        return x_train, y_train, x_test, y_test, feature_columns

    def run_experiments(self) -> pd.DataFrame:
        """Train three models on ReFeX features."""
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
                RESULTS_DIR / f"movie_refex_report_{model_name.lower().replace(' ', '_')}.csv",
                index=True,
            )

        comparison_df = pd.DataFrame(rows).sort_values("F1-Macro", ascending=False).reset_index(drop=True)
        comparison_df.to_csv(RESULTS_DIR / "movie_refex_comparison.csv", index=False)
        return comparison_df

    def recursive_feature_elimination(self, n_features_to_select: int = 6) -> pd.DataFrame:
        """Rank ReFeX movie features."""
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
        ranking_df.to_csv(RESULTS_DIR / "movie_refex_rfe.csv", index=False)
        return ranking_df

    def predict_test_labels(self) -> pd.DataFrame:
        """Save predicted labels on the ReFeX test split."""
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
        prediction_df.to_csv(RESULTS_DIR / "movie_refex_predictions.csv", index=False)
        return prediction_df


if __name__ == "__main__":
    experiment = MovieRefexExperiment()
    experiment.build_refex_dataset()
    experiment.prepare_train_test_split()
    print(experiment.run_experiments())
    print(experiment.recursive_feature_elimination())
    print(experiment.predict_test_labels().head(10))
