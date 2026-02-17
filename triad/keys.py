"""API key management for Triad Orchestrator.

Handles loading, saving, and validating provider API keys.
Keys are stored in ~/.triad/keys.env and loaded with this priority:
  1. Environment variables (highest — already set in shell)
  2. ~/.triad/keys.env (user's saved keys from `triad setup`)
  3. .env in current directory (project-level)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory for user-level Triad configuration
TRIAD_HOME = Path.home() / ".triad"
KEYS_FILE = TRIAD_HOME / "keys.env"

# Provider definitions: (env_var, display_name, description, signup_url)
PROVIDERS = [
    (
        "ANTHROPIC_API_KEY",
        "Anthropic (Claude)",
        "Best for refactoring & verification",
        "https://console.anthropic.com/settings/keys",
    ),
    (
        "OPENAI_API_KEY",
        "OpenAI (GPT-4o, o3)",
        "Best for fast implementation",
        "https://platform.openai.com/api-keys",
    ),
    (
        "GEMINI_API_KEY",
        "Google (Gemini)",
        "Best for architecture (1M context)",
        "https://aistudio.google.com/apikey",
    ),
    (
        "XAI_API_KEY",
        "xAI (Grok)",
        "Strong reasoning & coding",
        "https://console.x.ai",
    ),
]

# Map env var to a lightweight LiteLLM model ID for validation
_VALIDATION_MODELS: dict[str, str] = {
    "ANTHROPIC_API_KEY": "anthropic/claude-sonnet-4-5-20250929",
    "OPENAI_API_KEY": "gpt-4o-mini",
    "GEMINI_API_KEY": "gemini/gemini-2.5-flash",
    "XAI_API_KEY": "xai/grok-3",
}

_VALIDATION_DISPLAY: dict[str, str] = {
    "ANTHROPIC_API_KEY": "Claude Sonnet 4.5",
    "OPENAI_API_KEY": "GPT-4o Mini",
    "GEMINI_API_KEY": "Gemini 2.5 Flash",
    "XAI_API_KEY": "Grok 3",
}

# Map env var to provider short name
PROVIDER_NAMES: dict[str, str] = {
    "ANTHROPIC_API_KEY": "Anthropic",
    "OPENAI_API_KEY": "OpenAI",
    "GEMINI_API_KEY": "Google",
    "XAI_API_KEY": "xAI",
}

# Triad Pro API key (not a model provider — used for dashboard event forwarding)
PRO_KEY_ENV = "TRIAD_PRO_KEY"


def load_keys_env() -> None:
    """Load API keys from ~/.triad/keys.env and .env into os.environ.

    Respects priority: existing env vars are NOT overwritten.
    Load order (later files don't overwrite earlier):
      1. Environment variables (already in os.environ)
      2. ~/.triad/keys.env
      3. .env in current working directory
    """
    files = [KEYS_FILE, Path.cwd() / ".env"]
    for env_file in files:
        if env_file.is_file():
            _load_env_file(env_file)


def _load_env_file(path: Path) -> None:
    """Parse a simple KEY=VALUE .env file and set vars that aren't already set."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            # Don't overwrite existing env vars
            if key and not os.environ.get(key):
                os.environ[key] = value
                logger.debug("Loaded %s from %s", key, path)
    except OSError:
        logger.debug("Could not read %s", path)


def save_keys(keys: dict[str, str]) -> Path:
    """Save API keys to ~/.triad/keys.env.

    Args:
        keys: Mapping of env var name to key value (only non-empty saved).

    Returns:
        Path to the saved file.
    """
    TRIAD_HOME.mkdir(parents=True, exist_ok=True)

    lines = ["# Triad Orchestrator API Keys", f"# Saved by `triad setup`", ""]
    for env_var, value in keys.items():
        if value:
            lines.append(f"{env_var}={value}")

    KEYS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Restrict permissions on Unix (best-effort)
    try:
        KEYS_FILE.chmod(0o600)
    except OSError:
        pass

    return KEYS_FILE


def clear_keys() -> bool:
    """Remove ~/.triad/keys.env if it exists.

    Returns:
        True if the file was removed, False if it didn't exist.
    """
    if KEYS_FILE.is_file():
        KEYS_FILE.unlink()
        return True
    return False


def get_configured_keys() -> dict[str, str]:
    """Return a dict of env_var -> value for all known provider keys.

    Checks os.environ after loading keys.env files.
    """
    load_keys_env()
    return {
        env_var: os.environ.get(env_var, "")
        for env_var, _, _, _ in PROVIDERS
    }


def has_any_key() -> bool:
    """Check if at least one provider API key is configured anywhere."""
    load_keys_env()
    return any(os.environ.get(env_var) for env_var, _, _, _ in PROVIDERS)


async def validate_key(env_var: str, api_key: str) -> tuple[bool, str]:
    """Validate an API key by making a tiny LiteLLM call.

    Args:
        env_var: The environment variable name (e.g. ANTHROPIC_API_KEY).
        api_key: The actual key value to test.

    Returns:
        Tuple of (success, detail_message).
    """
    import litellm

    model = _VALIDATION_MODELS.get(env_var)
    if not model:
        return False, "Unknown provider"

    display = _VALIDATION_DISPLAY.get(env_var, model)

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 5,
        "timeout": 15.0,
        "api_key": api_key,
    }

    # xAI needs api_base
    if env_var == "XAI_API_KEY":
        kwargs["api_base"] = "https://api.x.ai/v1"

    try:
        await litellm.acompletion(**kwargs)
        return True, f"Connected ({display})"
    except litellm.AuthenticationError:
        return False, "Invalid key (401 Unauthorized)"
    except litellm.BadRequestError as e:
        return False, f"Bad request: {e}"
    except litellm.RateLimitError:
        # Key is valid but provider is rate-limiting — treat as degraded
        raise
    except litellm.ServiceUnavailableError:
        raise
    except Exception as e:
        return False, f"Error: {str(e)[:80]}"
