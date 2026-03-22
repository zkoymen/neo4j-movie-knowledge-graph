"""Simple project configuration loader."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# We keep .env in project root.
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

# If .env exists, load it. If not, defaults below are used.
load_dotenv(dotenv_path=ENV_PATH, override=False)

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
DATASET_NAME = os.getenv("DATASET_NAME", "recommendations_dump")
CORE_ACTOR_MIN_MOVIES = int(os.getenv("CORE_ACTOR_MIN_MOVIES", "3"))
CORE_ACTOR_MAX_ACTORS = int(os.getenv("CORE_ACTOR_MAX_ACTORS", "1500"))
APPROX_BETWEENNESS_K = int(os.getenv("APPROX_BETWEENNESS_K", "200"))
LINK_PREDICTION_MAX_ACTORS = int(os.getenv("LINK_PREDICTION_MAX_ACTORS", "800"))
MOVIE_SIMILARITY_MAX_MOVIES = int(os.getenv("MOVIE_SIMILARITY_MAX_MOVIES", "1200"))
MOVIE_SIMILARITY_MIN_SCORE = int(os.getenv("MOVIE_SIMILARITY_MIN_SCORE", "3"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
