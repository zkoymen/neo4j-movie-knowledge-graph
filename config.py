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
