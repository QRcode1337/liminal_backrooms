"""
Model Updater (Hybrid Mode) - Validates curated list against OpenRouter API

This script validates your hand-curated model list against the live API,
removing any models that no longer exist (404) while preserving your curation.

Features:
- Fast startup (5 second timeout)
- Caches validation results for 24 hours  
- Silently falls back to curated list if network unavailable

Usage:
    # Run directly to see validation results
    python tools/model_updater.py
    
    # Import in config.py
    from tools.model_updater import validate_models
    AI_MODELS = validate_models(CURATED_MODELS)
"""

import json
import time
from datetime import datetime
from pathlib import Path

# Try to import requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Cache file location (in project root, not tools folder)
CACHE_FILE = Path(__file__).parent.parent / "models_cache.json"
CACHE_MAX_AGE_HOURS = 24

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"


def fetch_available_model_ids(timeout: float = 5.0) -> set | None:
    """
    Fetch all available model IDs from OpenRouter API.
    
    Returns a set of model IDs, or None on failure.
    """
    if not HAS_REQUESTS:
        print("[ModelUpdater] requests library not available")
        return None
    
    try:
        start_time = time.time()
        response = requests.get(OPENROUTER_API_URL, timeout=timeout)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"[ModelUpdater] API returned status {response.status_code}")
            return None
        
        data = response.json()
        models = data.get("data", [])
        
        # Extract just the IDs
        model_ids = {m.get("id", "") for m in models if m.get("id")}
        
        print(f"[ModelUpdater] Fetched {len(model_ids)} model IDs in {elapsed:.2f}s")
        return model_ids
        
    except requests.exceptions.Timeout:
        print(f"[ModelUpdater] API timeout after {timeout}s")
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
        
        # Check cache age
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
            "model_ids": list(model_ids),
        }
        
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        
        return True
    except Exception as e:
        print(f"[ModelUpdater] Error saving cache: {e}")
        return False


def get_available_ids() -> set | None:
    """Get available model IDs from cache or API."""
    # Try cache first
    cached = load_cached_ids()
    if cached:
        print(f"[ModelUpdater] Using {len(cached)} cached model IDs")
        return cached
    
    # Fetch from API
    ids = fetch_available_model_ids()
    if ids:
        save_cached_ids(ids)
        return ids
    
    # Try stale cache as last resort
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            print("[ModelUpdater] Using stale cache")
            return set(cache_data.get("model_ids", []))
        except Exception as e:
            print(f"[ModelUpdater] Failed to read stale cache: {e}")

    return None


def validate_models(curated_models: dict) -> dict:
    """
    Validate curated model list against OpenRouter API.

    Removes any models that no longer exist.
    Returns the validated model dict with providers sorted alphabetically.

    Args:
        curated_models: Your hand-curated AI_MODELS dict

    Returns:
        Validated dict with 404'd models removed, providers sorted A-Z
    """
    available_ids = get_available_ids()

    if available_ids is None:
        print("[ModelUpdater] Validation skipped (no API data), using curated list as-is")
        return curated_models

    # Deep copy and validate
    validated = {}
    removed_count = 0
    kept_count = 0

    for tier, providers in curated_models.items():
        tier_providers = {}

        for provider, models in providers.items():
            provider_models = {}

            for display_name, model_id in models.items():
                if model_id in available_ids:
                    provider_models[display_name] = model_id
                    kept_count += 1
                else:
                    print(f"[ModelUpdater] [X] Removed (404): {model_id}")
                    removed_count += 1

            # Only add provider if it has models
            if provider_models:
                tier_providers[provider] = provider_models

        # Sort providers alphabetically within each tier
        validated[tier] = dict(sorted(tier_providers.items(), key=lambda x: x[0].lower()))

    if removed_count > 0:
        print(f"[ModelUpdater] Validated: {kept_count} kept, {removed_count} removed")
    else:
        print(f"[ModelUpdater] All {kept_count} curated models validated [OK]")

    return validated


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate AI models against OpenRouter API")
    parser.add_argument("--force", "-f", action="store_true", help="Force refresh cache")
    args = parser.parse_args()
    
    if args.force:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            print("Cache cleared")
    
    print("\n═══ FETCHING MODEL IDS ═══\n")
    ids = get_available_ids()
    
    if ids:
        print(f"\n[OK] {len(ids)} models available on OpenRouter")
        
        # Show some free models
        free_models = [m for m in ids if ":free" in m]
        print(f"\nFree models ({len(free_models)}):")
        for m in sorted(free_models)[:20]:
            print(f"  - {m}")
        if len(free_models) > 20:
            print(f"  ... and {len(free_models) - 20} more")
    else:
        print("\n[X] Could not fetch models")