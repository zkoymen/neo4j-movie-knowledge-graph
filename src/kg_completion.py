"""Knowledge graph completion with PyKEEN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from pykeen.pipeline import pipeline
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

import config

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class KGCompletionExperiment:
    """Run a small KG completion experiment on the movie graph."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self.triples_df: pd.DataFrame | None = None
        self.training = None
        self.testing = None
        self.validation = None
        self.pipeline_result = None

    def close(self) -> None:
        """Close Neo4j driver."""
        self.driver.close()

    def export_triples(self) -> pd.DataFrame:
        """Export a semantic subgraph as triples."""
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            actor_result = session.run(
                """
                MATCH (a:Actor)-[:ACTED_IN]->(:Movie)
                WITH a, count(*) AS movie_count
                WHERE movie_count >= $min_movie_count
                ORDER BY movie_count DESC, a.name
                LIMIT $max_actors
                WITH collect(a) AS actor_subset
                UNWIND actor_subset AS a
                MATCH (a)-[:ACTED_IN]->(m:Movie)
                RETURN a.name AS head, 'ACTED_IN' AS relation, m.title AS tail
                """,
                min_movie_count=config.CORE_ACTOR_MIN_MOVIES,
                max_actors=min(config.CORE_ACTOR_MAX_ACTORS, 1200),
            )
            acted_in_df = pd.DataFrame([dict(record) for record in actor_result])

            movie_titles = acted_in_df["tail"].drop_duplicates().tolist()
            if not movie_titles:
                raise ValueError("KG completion could not collect any movie triples.")

            directed_result = session.run(
                """
                MATCH (d:Director)-[:DIRECTED]->(m:Movie)
                WHERE m.title IN $movie_titles
                RETURN d.name AS head, 'DIRECTED' AS relation, m.title AS tail
                """,
                movie_titles=movie_titles,
            )
            directed_df = pd.DataFrame([dict(record) for record in directed_result])

            genre_result = session.run(
                """
                MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
                WHERE m.title IN $movie_titles
                RETURN m.title AS head, 'IN_GENRE' AS relation, g.name AS tail
                """,
                movie_titles=movie_titles,
            )
            genre_df = pd.DataFrame([dict(record) for record in genre_result])

        triples_df = pd.concat([acted_in_df, directed_df, genre_df], ignore_index=True).drop_duplicates()
        triples_df = triples_df.head(config.KG_COMPLETION_MAX_TRIPLES).reset_index(drop=True)
        triples_df.to_csv(RESULTS_DIR / "kg_completion_triples.csv", index=False)

        self.triples_df = triples_df
        return triples_df

    def build_triples_factory(self):
        """Build train, test, and validation factories."""
        if self.triples_df is None:
            self.export_triples()

        triples = self.triples_df[["head", "relation", "tail"]].astype(str).to_numpy(dtype=str)
        if len(triples) == 0:
            raise ValueError("KG completion triples are empty.")

        tf = TriplesFactory.from_labeled_triples(triples)
        training, testing, validation = tf.split(
            ratios=[0.8, 0.1, 0.1],
            random_state=config.RANDOM_STATE,
        )

        self.training = training
        self.testing = testing
        self.validation = validation
        return training, testing, validation

    def run_experiment(self) -> pd.DataFrame:
        """Train a small TransE model and save summary metrics."""
        if self.training is None or self.testing is None or self.validation is None:
            self.build_triples_factory()

        self.pipeline_result = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            model="TransE",
            random_seed=config.RANDOM_STATE,
            training_kwargs={
                "num_epochs": config.KG_COMPLETION_EPOCHS,
                "batch_size": 256,
            },
            device="cpu",
        )

        metric_rows = []
        metrics = getattr(self.pipeline_result, "metric_results", None)
        if metrics is not None:
            metric_dict = metrics.to_dict()
            flat_metrics = {}
            for outer_key, outer_value in metric_dict.items():
                if isinstance(outer_value, dict):
                    for inner_key, inner_value in outer_value.items():
                        flat_metrics[f"{outer_key}.{inner_key}"] = inner_value
                else:
                    flat_metrics[outer_key] = outer_value
            metric_rows.append(flat_metrics)

        metrics_df = pd.DataFrame(metric_rows if metric_rows else [{"status": "completed"}])
        metrics_df.to_csv(RESULTS_DIR / "kg_completion_metrics.csv", index=False)
        return metrics_df

    def predict_missing_genres(self, max_movies: int = 10) -> pd.DataFrame:
        """Predict possible missing IN_GENRE links for some movies."""
        if self.pipeline_result is None:
            self.run_experiment()

        if self.triples_df is None:
            self.export_triples()

        genre_triples = self.triples_df[self.triples_df["relation"] == "IN_GENRE"].copy()
        known_genres = genre_triples.groupby("head")["tail"].apply(set).to_dict()

        movie_candidates = (
            genre_triples["head"]
            .value_counts()
            .index.tolist()[:max_movies]
        )

        rows = []
        for movie_title in movie_candidates:
            predictions = predict_target(
                model=self.pipeline_result.model,
                head=movie_title,
                relation="IN_GENRE",
                triples_factory=self.training,
            )

            prediction_df = predictions.df.copy()
            label_column = "tail_label" if "tail_label" in prediction_df.columns else prediction_df.columns[0]
            score_column = "score" if "score" in prediction_df.columns else prediction_df.columns[-1]

            filtered_df = prediction_df.loc[
                ~prediction_df[label_column].isin(known_genres.get(movie_title, set()))
            ].head(3)

            for _, row in filtered_df.iterrows():
                rows.append(
                    {
                        "movie": movie_title,
                        "predicted_genre": row[label_column],
                        "score": row[score_column],
                    }
                )

        prediction_df = pd.DataFrame(rows).sort_values(["movie", "score"], ascending=[True, False])
        prediction_df = prediction_df.head(config.KG_COMPLETION_TOP_PREDICTIONS).reset_index(drop=True)
        prediction_df.to_csv(RESULTS_DIR / "kg_completion_predictions.csv", index=False)
        return prediction_df


if __name__ == "__main__":
    experiment = KGCompletionExperiment()
    try:
        print(experiment.export_triples().head())
        print(experiment.run_experiment())
        print(experiment.predict_missing_genres())
    finally:
        experiment.close()
