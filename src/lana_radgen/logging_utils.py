import logging
import os
from pathlib import Path


def _strip_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def load_project_env() -> None:
    current = Path.cwd().resolve()
    candidates = [current / ".env", Path(__file__).resolve().parents[2] / ".env"]
    env_path = next((path for path in candidates if path.exists()), None)
    if env_path is None:
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _strip_env_value(value))

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)


def configure_logging(level: str = "INFO") -> None:
    load_project_env()
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", level.lower())
