import logging
import os
from typing import Any, Dict

logger = logging.getLogger("atwz")

def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def load_env(path: str = ".env") -> Dict[str, str]:
    """Load simple KEY=VALUE .env file into a dict (non-robust)."""
    env: Dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def safe_get(d: Dict[Any, Any], key: Any, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default
