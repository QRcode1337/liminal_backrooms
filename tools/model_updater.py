"""
Model Updater - Validates curated list against live OmniRoute IDs.

OmniRoute owns routing and credentials. This script only checks that curated
model IDs still exist on GET {OMNIROUTE_BASE_URL}/models.

Features:
- Fast startup (5 second timeout)
- Caches validation results for 24 hours
- Silently falls back to curated list if OmniRoute is unreachable
- Keeps media models (Sora) that are not OmniRoute chat IDs

Usage:
    python tools/model_updater.py
    python tools/model_updater.py --force
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

CACHE_FILE = Path(__file__).parent.parent / "models_cache.json"
CACHE_MAX_AGE_HOURS = 24
MEDIA_MODEL_PREFIXES = ("sora-",)

DEFAULT_OMNIROUTE_BASE_URL = "http://127.0.0.1:20128/v1"


def omniroute_base_url() -> str:
    return os.getenv("OMNIROUTE_BASE_URL", DEFAULT_OMNIROUTE_BASE_URL).rstrip("/")


def omniroute_headers() -> dict:
    headers = {"Accept": "application/json"}
    api_key = os.getenv("OMNIROUTE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def is_media_model_id(model_id: str) -> bool:
    return bool(model_id) and model_id.lower().startswith(MEDIA_MODEL_PREFIXES)


def fetch_available_model_ids(timeout: float = 5.0) -> set | None:
    """Fetch all available model IDs from OmniRoute.

    Returns a set of model IDs, or None on failure.
    """
    if not HAS_REQUESTS:
        print("[ModelUpdater] requests library not available")
        return None

    url = f"{omniroute_base_url()}/models"
    try:
        start_time = time.time()
        response = requests.get(url, headers=omniroute_headers(), timeout=timeout)
        elapsed = time.time() - start_time

        if response.status_code != 200:
            print(f"[ModelUpdater] OmniRoute returned status {response.status_code}")
            return None

        data = response.json()
        models = data.get("data", [])
        model_ids = {m.get("id", "") for m in models if m.get("id")}

        print(f"[ModelUpdater] Fetched {len(model_ids)} OmniRoute IDs in {elapsed:.2f}s")
        return model_ids

    except requests.exceptions.Timeout:
        print(f"[ModelUpdater] OmniRoute timeout after {timeout}s")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ModelUpdater] Request error: {e}")
        return None
    except Exception as e:
        print(f"[ModelUpdater] Unexpected error: {e}")
        return None


def load_cached_ids() -> set | None:
    """Load cached model IDs if fresh."""
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        if cache_data.get("source") != "omniroute":
            print("[ModelUpdater] Cache is not from OmniRoute, refreshing...")
            return None

        cached_at = cache_data.get("cached_at", "")
        if cached_at:
            cached_time = datetime.fromisoformat(cached_at)
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            if age_hours > CACHE_MAX_AGE_HOURS:
                print(f"[ModelUpdater] Cache is {age_hours:.1f}h old, refreshing...")
                return None

        return set(cache_data.get("model_ids", []))

    except Exception as e:
        print(f"[ModelUpdater] Error loading cache: {e}")
        return None


def save_cached_ids(model_ids: set) -> bool:
    """Save model IDs to cache."""
    try:
        cache_data = {
            "cached_at": datetime.now().isoformat(),
            "source": "omniroute",
            "model_ids": list(model_ids),
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        return True
    except Exception as e:
        print(f"[ModelUpdater] Error saving cache: {e}")
        return False


def get_available_ids() -> set | None:
    """Get available model IDs from cache or OmniRoute."""
    cached = load_cached_ids()
    if cached:
        print(f"[ModelUpdater] Using {len(cached)} cached OmniRoute IDs")
        return cached

    ids = fetch_available_model_ids()
    if ids:
        save_cached_ids(ids)
        return ids

    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            print("[ModelUpdater] Using stale cache")
            return set(cache_data.get("model_ids", []))
        except Exception:
            pass

    return None


def validate_models(curated_models: dict) -> dict:
    """Validate curated model list against live OmniRoute IDs.

    Removes IDs OmniRoute no longer exposes. Keeps Sora media IDs.
    Returns the validated model dict with providers sorted alphabetically.
    """
    available_ids = get_available_ids()

    if available_ids is None:
        print("[ModelUpdater] Validation skipped (no OmniRoute data), using curated list as-is")
        return curated_models

    validated = {}
    removed_count = 0
    kept_count = 0

    for tier, providers in curated_models.items():
        tier_providers = {}

        for provider, models in providers.items():
            provider_models = {}

            for display_name, model_id in models.items():
                if is_media_model_id(model_id) or model_id in available_ids:
                    provider_models[display_name] = model_id
                    kept_count += 1
                else:
                    print(f"[ModelUpdater] [X] Removed (not on OmniRoute): {model_id}")
                    removed_count += 1

            if provider_models:
                tier_providers[provider] = provider_models

        validated[tier] = dict(sorted(tier_providers.items(), key=lambda x: x[0].lower()))

    if removed_count > 0:
        print(f"[ModelUpdater] Validated: {kept_count} kept, {removed_count} removed")
    else:
        print(f"[ModelUpdater] All {kept_count} curated models validated [OK]")

    return validated


# OmniRoute provider prefixes shown as selectable endpoints in the model picker.
# Wrappers like nous/, cursor/, no-think/ are omitted so the list stays usable.
ENDPOINT_PREFIXES = {
    "agy": "Antigravity",
    "cx": "Codex (cx)",
    "codex": "Codex",
    "xai": "xAI",
    "grok-cli": "Grok CLI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "openai": "OpenAI",
    "cerebras": "Cerebras",
    "gh": "GitHub Models",
    "nvidia": "NVIDIA",
    "auto": "Combos",
}

_SKIP_ID_PARTS = (
    "imagine-image",
    "imagine-video",
    "embedding",
    "tts",
    "transcribe",
    "whisper",
    "moderation",
    "dall-e",
    "realtime",
    "computer-use",
    "babbage",
    "davinci",
    "search-api",
    "sora",
)

_ACRONYMS = {"gpt", "glm", "oss", "cli", "grok", "api", "llm", "vl"}


def is_chat_endpoint_id(model_id: str) -> bool:
    low = (model_id or "").lower()
    if "/" not in low:
        return False
    return not any(part in low for part in _SKIP_ID_PARTS)


def pretty_model_name(model_id: str) -> str:
    slug = model_id.split("/", 1)[-1]
    parts = []
    for part in slug.replace("_", "-").split("-"):
        if not part:
            continue
        lower = part.lower()
        if lower in _ACRONYMS:
            parts.append(lower.upper())
        elif any(ch.isdigit() for ch in part) or "." in part:
            parts.append(part)
        else:
            parts.append(part.capitalize())
    return " ".join(parts) or model_id


def build_endpoint_catalog(available_ids: set | None = None) -> dict:
    """Group live OmniRoute IDs by provider prefix for the Endpoints picker."""
    if available_ids is None:
        available_ids = get_available_ids()
    if not available_ids:
        return {}

    grouped = {label: {} for label in ENDPOINT_PREFIXES.values()}
    counts = {label: 0 for label in ENDPOINT_PREFIXES.values()}

    for model_id in sorted(available_ids):
        if not is_chat_endpoint_id(model_id):
            continue
        prefix = model_id.split("/", 1)[0]
        label = ENDPOINT_PREFIXES.get(prefix)
        if not label:
            continue
        display = f"{pretty_model_name(model_id)} [{prefix}]"
        grouped[label][display] = model_id
        counts[label] += 1

    catalog = {label: models for label, models in grouped.items() if models}
    total = sum(len(models) for models in catalog.values())
    print(f"[ModelUpdater] Endpoint catalog: {total} models across {len(catalog)} OmniRoute prefixes")
    return catalog


def merge_endpoint_catalog(curated_models: dict, endpoints: dict | None = None) -> dict:
    """Put live OmniRoute endpoints first; drop the static OmniRoute shortcut group."""
    if endpoints is None:
        endpoints = build_endpoint_catalog()
    ordered = {}
    if endpoints:
        ordered["Endpoints"] = endpoints
    for tier, providers in curated_models.items():
        if endpoints and tier == "OmniRoute":
            continue
        ordered[tier] = providers
    return ordered


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate AI models against live OmniRoute IDs")
    parser.add_argument("--force", "-f", action="store_true", help="Force refresh cache")
    args = parser.parse_args()

    if args.force and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print("Cache cleared")

    print("\n═══ FETCHING OMNIROUTE MODEL IDS ═══\n")
    ids = get_available_ids()

    if ids:
        print(f"\n[OK] {len(ids)} models available on OmniRoute")
        prefixes = sorted({mid.split("/", 1)[0] for mid in ids if "/" in mid})
        print(f"\nProviders ({len(prefixes)}):")
        for prefix in prefixes[:30]:
            print(f"  - {prefix}")
        if len(prefixes) > 30:
            print(f"  ... and {len(prefixes) - 30} more")
    else:
        print("\n[X] Could not fetch OmniRoute models")
