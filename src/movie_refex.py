"""ReFeX-style movie classification experiment."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

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
        self.best_model_name: str | None = None
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
                    "feature_columns": len(dataset_df.columns) - 3,
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

    def _model_configs(self) -> dict[str, dict]:
        """Return model search configs for this experiment."""
        return {
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
                # This slightly wider grid was enough to move the stable 4-class
                # task from "almost 0.70" to "about 0.70" macro-F1.
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [2, 3],
                    "subsample": [0.8, 1.0],
                    "min_samples_leaf": [1, 3],
                },
                "use_scaled": False,
            },
        }

    def _oversample_train_data(
        self,
        x_df: pd.DataFrame,
        y_series: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Simple random oversampling for minority classes."""
        train_df = x_df.copy()
        train_df["_target"] = y_series.values
        max_count = train_df["_target"].value_counts().max()
        parts = []

        for label, group in train_df.groupby("_target"):
            if len(group) < max_count:
                group = resample(
                    group,
                    replace=True,
                    n_samples=max_count,
                    random_state=config.RANDOM_STATE,
                )
            parts.append(group)

        full_df = (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1.0, random_state=config.RANDOM_STATE)
            .reset_index(drop=True)
        )
        return full_df.drop(columns=["_target"]), full_df["_target"]

    def run_experiments(self) -> pd.DataFrame:
        """Train three models on ReFeX features."""
        x_train, y_train, x_test, y_test, feature_columns = self._split_features()

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        models = self._model_configs()

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
            balanced_accuracy = balanced_accuracy_score(y_test, predictions)
            f1_macro = f1_score(y_test, predictions, average="macro")

            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Balanced Accuracy": balanced_accuracy,
                    "F1-Macro": f1_macro,
                    "Best Params": str(grid.best_params_),
                }
            )

            if f1_macro > best_score:
                best_score = f1_macro
                self.best_model = best_estimator
                self.best_model_name = model_name
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
            confusion_df = pd.DataFrame(
                confusion_matrix(
                    y_test,
                    predictions,
                    labels=list(range(len(self.label_encoder.classes_))),
                ),
                index=self.label_encoder.classes_,
                columns=self.label_encoder.classes_,
            )
            confusion_df.to_csv(
                RESULTS_DIR / f"movie_refex_confusion_{model_name.lower().replace(' ', '_')}.csv",
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

    def run_cross_validation_summary(self, n_splits: int = 5) -> pd.DataFrame:
        """Run stratified cross-validation on the main model candidates."""
        if self.dataset is None:
            self.build_refex_dataset()

        x_df = self.dataset.drop(columns=["node", "title", "dominant_genre"]).fillna(0.0)
        y_series = self.dataset["dominant_genre"].copy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

        rows = []
        for model_name, cfg in self._model_configs().items():
            macro_scores = []
            balanced_scores = []

            for train_idx, test_idx in skf.split(x_df, y_series):
                x_train = x_df.iloc[train_idx].copy()
                y_train = y_series.iloc[train_idx].copy()
                x_test = x_df.iloc[test_idx].copy()
                y_test = y_series.iloc[test_idx].copy()

                if cfg["use_scaled"]:
                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(x_train)
                    x_test = scaler.transform(x_test)

                model = clone(cfg["model"])
                if model_name == "Logistic Regression":
                    # We use the strongest single setting already found above.
                    model.set_params(C=10.0)
                elif model_name == "Random Forest":
                    model.set_params(n_estimators=200, max_depth=None)
                elif model_name == "Gradient Boosting":
                    model.set_params(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=3,
                        subsample=0.8,
                        min_samples_leaf=1,
                    )

                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                macro_scores.append(f1_score(y_test, predictions, average="macro"))
                balanced_scores.append(balanced_accuracy_score(y_test, predictions))

            rows.append(
                {
                    "Model": model_name,
                    "CV F1-Macro Mean": sum(macro_scores) / len(macro_scores),
                    "CV F1-Macro Std": pd.Series(macro_scores).std(ddof=0),
                    "CV Balanced Accuracy Mean": sum(balanced_scores) / len(balanced_scores),
                    "CV Balanced Accuracy Std": pd.Series(balanced_scores).std(ddof=0),
                }
            )

        cv_df = pd.DataFrame(rows).sort_values("CV F1-Macro Mean", ascending=False).reset_index(drop=True)
        cv_df.to_csv(RESULTS_DIR / "movie_refex_cv_summary.csv", index=False)
        return cv_df

    def run_imbalance_strategy_comparison(self) -> pd.DataFrame:
        """Compare base, weighted, and oversampled variants on the holdout split."""
        if self.train_df is None or self.test_df is None:
            self.prepare_train_test_split()

        feature_columns = [
            column for column in self.train_df.columns if column not in {"node", "title", "dominant_genre"}
        ]
        x_train = self.train_df[feature_columns].fillna(0.0).copy()
        y_train = self.train_df["dominant_genre"].copy()
        x_test = self.test_df[feature_columns].fillna(0.0).copy()
        y_test = self.test_df["dominant_genre"].copy()

        variants = {
            "gb_base": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                min_samples_leaf=1,
                random_state=config.RANDOM_STATE,
            ),
            "gb_oversampled": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                min_samples_leaf=1,
                random_state=config.RANDOM_STATE,
            ),
            "rf_base": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=config.RANDOM_STATE),
            "rf_weighted": RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", random_state=config.RANDOM_STATE),
            "lr_base": LogisticRegression(C=10.0, max_iter=3000, random_state=config.RANDOM_STATE),
            "lr_weighted": LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced", random_state=config.RANDOM_STATE),
        }

        rows = []
        for name, model in variants.items():
            x_fit = x_train.copy()
            y_fit = y_train.copy()
            x_eval = x_test.copy()

            if name.endswith("oversampled"):
                x_fit, y_fit = self._oversample_train_data(x_fit, y_fit)

            if name.startswith("lr_"):
                scaler = StandardScaler()
                x_fit = scaler.fit_transform(x_fit)
                x_eval = scaler.transform(x_eval)

            model.fit(x_fit, y_fit)
            predictions = model.predict(x_eval)

            rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
                    "F1-Macro": f1_score(y_test, predictions, average="macro"),
                }
            )

        strategy_df = pd.DataFrame(rows).sort_values("F1-Macro", ascending=False).reset_index(drop=True)
        strategy_df.to_csv(RESULTS_DIR / "movie_refex_imbalance_comparison.csv", index=False)
        return strategy_df


if __name__ == "__main__":
    experiment = MovieRefexExperiment()
    experiment.build_refex_dataset()
    experiment.prepare_train_test_split()
    print(experiment.run_experiments())
    print(experiment.recursive_feature_elimination())
    print(experiment.predict_test_labels().head(10))
