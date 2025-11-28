"""Centralized configuration and constants for the glossary tool."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()
MODEL = os.getenv("MODEL", "gpt-4o-mini").strip()
MODEL_P1_CTX = "gpt-4o-mini"
MODEL_P2_NOCTX = "gpt-5"

MIN_FREQ = 2
NGRAM_MAX = 4
OUTPUT_PHASE1 = "Glossary_Phase1.csv"
OUTPUT_PHASE2 = "Glossary_Normalized.csv"
OUTPUT_PHASE3 = "Glossary_Final.csv"
CAPITAL_PRESENCE_REQUIRED = True
BATCH = 20
BATCH_P1 = 20
BATCH_P2 = 80
TIMEOUT = 180
RETRY_LIMIT = 3
RETRY_DELAY = 3


def require_api_credentials() -> None:
    """Ensure BASE_URL and API_KEY are available before hitting the API."""
    if not BASE_URL or not API_KEY:
        raise RuntimeError("Missing BASE_URL or API_KEY in .env")

