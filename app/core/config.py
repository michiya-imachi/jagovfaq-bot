import os
from pathlib import Path

from dotenv import load_dotenv


def resolve_repo_root() -> Path:
    # app/core/config.py -> app/core -> app -> repo_root
    return Path(__file__).resolve().parents[2]


def resolve_index_dir() -> Path:
    # Prefer explicit path
    env_dir = os.getenv("INDEX_DIR")
    if env_dir:
        return Path(env_dir)

    # Prefer local data/index in repository root
    base = resolve_repo_root()
    p1 = base / "data" / "index"
    if p1.exists():
        return p1

    # Fallback to current working directory
    return Path("data/index")


def load_env_files() -> None:
    # Load .env from repository root first, then from current working directory.
    local_env = resolve_repo_root() / ".env"
    if local_env.exists():
        load_dotenv(local_env)
    load_dotenv()


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default
