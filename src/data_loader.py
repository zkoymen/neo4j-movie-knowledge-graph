"""Neo4j data loading helpers for OMDb based movie import."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, List, Tuple

import requests
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

import config

OMDB_TIMEOUT_SECONDS = 20
CACHE_DIR = Path("outputs/raw/omdb")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SchemaStats:
    """Simple container to print and inspect schema counts."""

    node_counts: List[Tuple[str, int]]
    relationship_counts: List[Tuple[str, int]]


def _split_csv_field(value: str) -> list[str]:
    """Split OMDb comma separated values into clean list items."""
    if not value or value == "N/A":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _slugify(value: str) -> str:
    """Make a filesystem-safe cache name."""
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _parse_title_seed(raw_value: str) -> tuple[str, int | None]:
    """Parse 'Title::Year' or just 'Title'."""
    if "::" not in raw_value:
        return raw_value.strip(), None

    title, year = raw_value.split("::", 1)
    parsed_year = _safe_int(year.strip())
    return title.strip(), parsed_year


def _safe_int(value: str) -> int | None:
    """Convert text to int when possible."""
    if not value or value == "N/A":
        return None
    digits = "".join(char for char in value if char.isdigit())
    return int(digits) if digits else None


def _safe_float(value: str) -> float | None:
    """Convert text to float when possible."""
    if not value or value == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _normalize_rating(value: str) -> float | None:
    """
    Normalize ratings to a 0-10 scale.

    Examples:
    - 8.7/10 -> 8.7
    - 88% -> 8.8
    - 73/100 -> 7.3
    """
    if not value or value == "N/A":
        return None

    if value.endswith("%"):
        return round(float(value[:-1]) / 10.0, 1)

    if "/" in value:
        raw_score, max_score = value.split("/", 1)
        return round((float(raw_score) / float(max_score)) * 10.0, 1)

    try:
        return float(value)
    except ValueError:
        return None


class MovieGraphLoader:
    """Load and enrich the movie knowledge graph from OMDb."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self.http = requests.Session()

    def close(self) -> None:
        """Close both Neo4j and HTTP sessions."""
        self.driver.close()
        self.http.close()

    def test_connection(self) -> None:
        """Run a tiny query to check Neo4j connectivity."""
        with self.driver.session() as session:
            record = session.run("RETURN 1 AS ok").single()
        if not record or record["ok"] != 1:
            raise RuntimeError("Neo4j connection test failed.")

    def clear_database(self) -> None:
        """Remove old graph so each run starts clean."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def _get_target_titles(self) -> list[tuple[str, int | None]]:
        """Read movie titles from config."""
        raw_titles = config.OMDB_MOVIE_TITLES
        return [_parse_title_seed(title) for title in raw_titles.split("|") if title.strip()]

    def fetch_movie_from_omdb(self, title: str, year: int | None = None) -> dict[str, Any]:
        """Fetch one movie payload from OMDb."""
        if not config.OMDB_API_KEY:
            raise RuntimeError("OMDB_API_KEY is missing in .env.")

        cache_name = _slugify(title)
        if year is not None:
            cache_name = f"{cache_name}_{year}"
        cache_path = CACHE_DIR / f"{cache_name}.json"

        if config.OMDB_USE_CACHE and cache_path.exists():
            print(f"Using cached OMDb response: {title}")
            return json.loads(cache_path.read_text(encoding="utf-8"))

        params = {
            "apikey": config.OMDB_API_KEY,
            "t": title,
            "type": "movie",
            "plot": "short",
            "r": "json",
        }
        if year is not None:
            params["y"] = year

        response = self.http.get(
            config.OMDB_BASE_URL,
            params=params,
            timeout=OMDB_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("Response") == "False":
            raise RuntimeError(f"OMDb could not find movie: {title} | {payload.get('Error')}")

        cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return payload

    def _merge_movie_payload(self, payload: dict[str, Any]) -> None:
        """Write one OMDb movie payload into Neo4j."""
        title = payload.get("Title", "").strip()
        imdb_id = payload.get("imdbID", "").strip()
        if not title or not imdb_id:
            return

        year = _safe_int(payload.get("Year", ""))
        imdb_rating = _safe_float(payload.get("imdbRating", ""))
        runtime = _safe_int(payload.get("Runtime", ""))

        with self.driver.session() as session:
            session.run(
                """
                MERGE (m:Movie {imdb_id: $imdb_id})
                SET m.title = $title,
                    m.year = $year,
                    m.rating = $rating,
                    m.rated = $rated,
                    m.released = $released,
                    m.runtime = $runtime,
                    m.plot = $plot,
                    m.poster = $poster,
                    m.metascore = $metascore,
                    m.imdb_votes = $imdb_votes,
                    m.box_office = $box_office,
                    m.production = $production,
                    m.website = $website,
                    m.type = $type
                """,
                imdb_id=imdb_id,
                title=title,
                year=year,
                rating=imdb_rating,
                rated=payload.get("Rated"),
                released=payload.get("Released"),
                runtime=runtime,
                plot=payload.get("Plot"),
                poster=payload.get("Poster"),
                metascore=_safe_int(payload.get("Metascore", "")),
                imdb_votes=payload.get("imdbVotes"),
                box_office=payload.get("BoxOffice"),
                production=payload.get("Production"),
                website=payload.get("Website"),
                type=payload.get("Type"),
            )

            for actor_name in _split_csv_field(payload.get("Actors", "")):
                session.run(
                    """
                    MERGE (a:Actor {name: $actor_name})
                    MERGE (m:Movie {imdb_id: $imdb_id})
                    MERGE (a)-[:ACTED_IN]->(m)
                    """,
                    actor_name=actor_name,
                    imdb_id=imdb_id,
                )

            for director_name in _split_csv_field(payload.get("Director", "")):
                session.run(
                    """
                    MERGE (d:Director {name: $director_name})
                    MERGE (m:Movie {imdb_id: $imdb_id})
                    MERGE (d)-[:DIRECTED]->(m)
                    """,
                    director_name=director_name,
                    imdb_id=imdb_id,
                )

            for genre_name in _split_csv_field(payload.get("Genre", "")):
                session.run(
                    """
                    MERGE (g:Genre {name: $genre_name})
                    MERGE (m:Movie {imdb_id: $imdb_id})
                    MERGE (m)-[:IN_GENRE]->(g)
                    """,
                    genre_name=genre_name,
                    imdb_id=imdb_id,
                )

            for country_name in _split_csv_field(payload.get("Country", "")):
                session.run(
                    """
                    MERGE (c:Country {name: $country_name})
                    MERGE (m:Movie {imdb_id: $imdb_id})
                    MERGE (m)-[:IN_COUNTRY]->(c)
                    """,
                    country_name=country_name,
                    imdb_id=imdb_id,
                )

            for rating_item in payload.get("Ratings", []):
                source_name = rating_item.get("Source", "").strip()
                raw_rating = rating_item.get("Value", "").strip()
                normalized_rating = _normalize_rating(raw_rating)
                if not source_name or normalized_rating is None:
                    continue

                session.run(
                    """
                    MERGE (u:User {name: $source_name})
                    MERGE (m:Movie {imdb_id: $imdb_id})
                    MERGE (u)-[r:RATED]->(m)
                    SET r.rating = $rating,
                        r.raw_value = $raw_value
                    """,
                    source_name=source_name,
                    imdb_id=imdb_id,
                    rating=normalized_rating,
                    raw_value=raw_rating,
                )

    def load_omdb_movies_dataset(self) -> None:
        """Fetch target movies from OMDb and import them into Neo4j."""
        titles = self._get_target_titles()
        for title, year in titles:
            year_suffix = f" ({year})" if year is not None else ""
            print(f"Fetching from OMDb: {title}{year_suffix}")
            payload = self.fetch_movie_from_omdb(title, year=year)
            self._merge_movie_payload(payload)

    def verify_schema(self) -> SchemaStats:
        """Return node and relationship counts by type."""
        with self.driver.session() as session:
            node_result = session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(*) AS count
                ORDER BY label
                """
            )
            relationship_result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY type
                """
            )

            node_counts = [(record["label"], record["count"]) for record in node_result]
            relationship_counts = [(record["type"], record["count"]) for record in relationship_result]

        return SchemaStats(
            node_counts=node_counts,
            relationship_counts=relationship_counts,
        )

    def print_schema_stats(self, stats: SchemaStats) -> None:
        """Print schema stats in a friendly way."""
        print("\n=== Node Counts ===")
        for label, count in stats.node_counts:
            print(f"- {label}: {count}")

        print("\n=== Relationship Counts ===")
        for rel_type, count in stats.relationship_counts:
            print(f"- {rel_type}: {count}")


def run_phase1_load() -> None:
    """Helper function to run the current import flow."""
    loader = MovieGraphLoader()
    try:
        print("Checking Neo4j connection...")
        loader.test_connection()
        print("Connection OK.")

        print("Clearing old data...")
        loader.clear_database()

        print("Loading OMDb movies dataset...")
        loader.load_omdb_movies_dataset()

        stats = loader.verify_schema()
        loader.print_schema_stats(stats)
        print("\nData loading finished.")
    except Neo4jError as exc:
        print("Neo4j error during loading:")
        print(f"- {exc}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    run_phase1_load()
