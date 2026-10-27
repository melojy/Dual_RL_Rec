"""DPFR configuration, validation, reproducibility, and atomic file I/O.

Edit config.json for paths and hyperparameters. Relative paths are resolved
against the configuration file. API secrets are read only from the environment.
"""
from __future__ import annotations
import copy
import hashlib
import os
import random
import tempfile
import json
import math
from pathlib import Path
from typing import Any, Iterator


def default_config() -> dict:
    return {
        "direction": "YELP_P_N", "seed": 2026, "device": "auto",
        "deterministic": True, "num_threads": 4, "output_dir": "outputs/yelp_p_n",
        "data": {
            "interactions": "data/source_users.jsonl", "metadata": "data/poi_metadata.json",
            "profiles": "data/profiles.jsonl", "candidates": "data/candidates.json",
            "source_key": "source_records", "source_region": "Philadelphia",
            "target_region": "New Orleans", "region_fields": ["domain", "city", "state"],
            "feature_fields": ["features", "categories", "attributes"],
            "record_order": "timestamp_if_complete", "rating_aggregation": "latest",
            "cached_profiles_scope": "source_only",
        },
        "model": {
            "dimension": 128, "vocab_size": 20000, "max_tokens": 128,
            "heads": 8, "layers": 4, "dropout": 0.2, "temperature": 0.1,
            "actor_hidden": [256, 128],
        },
        "training": {
            "episodes": 6000, "decision_steps": 20, "feedback_users": 50,
            "source_candidates": 100, "max_visited_candidates": 10,
            "warmup_epochs": 3, "warmup_batch_size": 64,
            "feedback_epochs": 3, "feedback_batch_size": 64,
            "encoder_warmup_lr": 1e-4, "encoder_feedback_lr": 1e-5,
            "encoder_weight_decay": 0.01, "actor_lr": 1e-4, "critic_lr": 3e-4,
            "ppo_epochs": 4, "ppo_clip": 0.2, "value_clip": 0.2,
            "gamma": 0.99, "gae_lambda": 0.95, "normalize_advantage": True,
            "residual_start": 1.0, "residual_end": 0.1,
            "encoder_grad_clip": 1.0, "policy_grad_clip": 0.5,
            "checkpoint_every": 500,
        },
        "inference": {"top_k": 20, "encode_batch_size": 256},
        "profiling": {
            "base_url": "https://api.deepseek.com", "model": "",
            "api_key_env": "DEEPSEEK_API_KEY", "temperature": 0.0,
            "max_output_tokens": 2048, "max_input_chars": 180000,
            "timeout_seconds": 120, "retries": 3,
        },
    }


def _merge(base: dict, update: dict, prefix: str = "") -> dict:
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration field: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be an object")
            _merge(base[key], value, prefix + key + ".")
        else:
            base[key] = value
    return base


def validate_config(c: dict) -> None:
    m, t, d, inf, prof = (c[k] for k in ("model", "training", "data", "inference", "profiling"))
    if type(c["seed"]) is not int or c["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(c["num_threads"]) is not int or c["num_threads"] < 1:
        raise ValueError("num_threads must be positive")
    if not isinstance(c["direction"], str) or not c["direction"].strip():
        raise ValueError("direction must be a nonempty run label")
    if type(c["deterministic"]) is not bool or type(t["normalize_advantage"]) is not bool:
        raise ValueError("deterministic and normalize_advantage must be booleans")
    for key in ("source_region", "target_region", "source_key"):
        if not isinstance(d[key], str) or not d[key].strip():
            raise ValueError(f"data.{key} must be a nonempty string")
    if d["source_region"].casefold() == d["target_region"].casefold():
        raise ValueError("source_region and target_region must differ")
    for key in ("region_fields", "feature_fields"):
        if not isinstance(d[key], list) or not d[key] or any(not isinstance(x, str) for x in d[key]):
            raise ValueError(f"data.{key} must be a nonempty list of field names")
    if len(set(d["feature_fields"])) != len(d["feature_fields"]):
        raise ValueError("feature_fields cannot contain duplicate names")
    forbidden = {"stars", "rating", "ratings", "avg_rating", "average_rating", "reviews",
                 "review", "review_count", "num_of_reviews", "checkins", "visit_count"}
    if set(d["feature_fields"]) & forbidden:
        raise ValueError("feature_fields must contain static attributes, not ratings/reviews/behavior statistics")
    if d["record_order"] not in ("timestamp_if_complete", "input"):
        raise ValueError("record_order must be timestamp_if_complete or input")
    if d["rating_aggregation"] not in ("latest", "mean"):
        raise ValueError("rating_aggregation must be latest or mean")
    if d["cached_profiles_scope"] != "source_only":
        raise ValueError("Profile caches must be declared source_only")
    for key in ("dimension", "vocab_size", "max_tokens", "heads", "layers"):
        if type(m[key]) is not int or m[key] < 1:
            raise ValueError(f"model.{key} must be positive")
    if m["dimension"] % 2 or m["dimension"] % m["heads"]:
        raise ValueError("dimension must be even and divisible by heads")
    if m["vocab_size"] < 8 or not 0 <= m["dropout"] < 1 or m["temperature"] <= 0:
        raise ValueError("Invalid vocabulary size, dropout or temperature")
    if not isinstance(m["actor_hidden"], list) or len(m["actor_hidden"]) != 2 or any(type(v) is not int or v < 1 for v in m["actor_hidden"]):
        raise ValueError("actor_hidden must contain two positive integers")
    for key in ("episodes", "decision_steps", "feedback_users", "source_candidates", "warmup_batch_size",
                "feedback_epochs", "feedback_batch_size", "ppo_epochs"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("warmup_epochs", "max_visited_candidates", "checkpoint_every"):
        if type(t[key]) is not int or t[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    if max(t["decision_steps"], t["max_visited_candidates"]) > t["source_candidates"]:
        raise ValueError("decision_steps / max_visited_candidates exceeds source_candidates")
    for key in ("actor_lr", "critic_lr", "encoder_warmup_lr", "encoder_feedback_lr", "ppo_clip", "value_clip", "encoder_grad_clip", "policy_grad_clip"):
        if not isinstance(t[key], (int, float)) or isinstance(t[key], bool) or t[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("encoder_weight_decay", "residual_start", "residual_end"):
        if not isinstance(t[key], (int, float)) or t[key] < 0:
            raise ValueError(f"training.{key} must be nonnegative")
    if not 0 <= t["gamma"] <= 1 or not 0 <= t["gae_lambda"] <= 1:
        raise ValueError("gamma and gae_lambda must be in [0,1]")
    for key in ("top_k", "encode_batch_size"):
        if type(inf[key]) is not int or inf[key] < 1:
            raise ValueError(f"inference.{key} must be positive")
    for key in ("max_output_tokens", "max_input_chars", "timeout_seconds", "retries"):
        if type(prof[key]) is not int or prof[key] < 1:
            raise ValueError(f"profiling.{key} must be a positive integer")
    if not isinstance(prof["base_url"], str) or not prof["base_url"].startswith("https://"):
        raise ValueError("profiling.base_url must be an HTTPS URL")
    if not isinstance(prof["model"], str) or not isinstance(prof["api_key_env"], str) or not prof["api_key_env"]:
        raise ValueError("Set a model string and an API key environment-variable name")
    if isinstance(prof["temperature"], bool) or not isinstance(prof["temperature"], (int, float)) or not 0 <= prof["temperature"] <= 2:
        raise ValueError("profiling.temperature must be a number in [0,2]")
    for section in (m, t, inf, prof):
        for key, value in section.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite value for {key}")


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    c = _merge(copy.deepcopy(default_config()), raw)
    for key in ("interactions", "metadata", "profiles", "candidates"):
        value = c["data"][key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"data.{key} must be a file path")
        p = Path(value).expanduser()
        c["data"][key] = str((p if p.is_absolute() else path.parent / p).resolve())
    p = Path(c["output_dir"]).expanduser()
    c["output_dir"] = str((p if p.is_absolute() else path.parent / p).resolve())
    validate_config(c)
    # A run directory must not overlap with input files it writes or overwrites.
    reserved = {str(Path(c["output_dir"]) / name) for name in
                ("config.json", "model.pt", "training_state.pt", "prototypes.jsonl", "recommendations.json", "training_log.jsonl")}
    if set(c["data"][k] for k in ("interactions", "metadata", "profiles", "candidates")) & reserved:
        raise ValueError("An input file overlaps with a reserved output file")
    return c


# ---------------------------------------------------------------------------
# File formats and reproducible runtime setup
# ---------------------------------------------------------------------------

def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def iter_rows(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path.name}:{i}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path.name}:{i}")
                yield row
    else:
        data = read_json(path)
        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    raise ValueError("Each JSON record must be an object")
                yield row
        elif isinstance(data, dict) and "user_id" in data:
            yield data
        elif isinstance(data, dict):
            if "profiles" in data:
                data = data["profiles"]
            for uid, row in data.items():
                if not isinstance(row, dict):
                    raise ValueError("Expected user-ID-to-object mapping")
                if "user_id" in row and str(row["user_id"]) != str(uid):
                    raise ValueError("Mismatched outer and inner user IDs")
                yield {**row, "user_id": str(uid)}
        else:
            raise ValueError("Expected JSON list, user mapping, or JSONL")


def set_seed(seed: int, deterministic: bool, threads: int = 4) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if threads > 0:
        torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def device_for(name: str):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install the matching PyTorch build or use cpu.")
    return device


def save_torch(path: str | Path, payload: dict) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "wb") as f:
            # File-handle serialization avoids embedding the random temporary filename.
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_jsonl(path: str | Path, rows) -> None:
    """Stream JSON Lines to an atomic replacement, with no giant JSON string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


