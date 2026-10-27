"""DPFR data loading, candidate sampling, state transitions, and source rewards.

The source-training loader reads only source_records. Target inference uses a
separate candidate loader and a reward-free RankingState. Candidate retrieval
is external: this module accepts the user-to-POI-ID lists supplied by the user.
"""
from __future__ import annotations
import ast
import json
import math
import re
import random
import torch
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from Config import iter_rows, read_json, stable_hash

ID_KEYS = ("business_id", "gmap_id", "poi_id", "item_id", "id")
RATING_KEYS = ("rating", "stars", "review_rating", "score")


def identifier(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ID_KEYS:
            if key in obj and obj[key] is not None:
                return identifier(obj[key])
        raise ValueError("POI record is missing a recognized ID")
    if obj is None or isinstance(obj, (list, bool)):
        raise ValueError("Invalid POI identifier")
    out = str(obj).strip()
    if not out:
        raise ValueError("Empty POI identifier")
    return out


def user_id(value: Any) -> str:
    if value is None or isinstance(value, (bool, list, dict)) or not str(value).strip():
        raise ValueError("A nonempty user_id is required")
    return str(value).strip()


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(x))).strip()


def unique_features(values: list[str]) -> list[str]:
    seen, out = set(), []
    for x in values:
        x = clean_text(x)
        if x and x.casefold() not in seen:
            seen.add(x.casefold())
            out.append(x)
    return out


def _literal(x: Any) -> Any:
    if isinstance(x, str):
        text = x.strip()
        if (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")):
            try:
                return json.loads(text)
            except (ValueError, RecursionError):
                try:
                    return ast.literal_eval(text)
                except (ValueError, SyntaxError, RecursionError):
                    pass
        if (text.startswith("u'") and text.endswith("'")) or (
            text.startswith("'") and text.endswith("'")):
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                pass
    return x


def _flatten(x: Any, prefix: str = "") -> list[str]:
    x = _literal(x)
    if x is None:
        return []
    if isinstance(x, dict):
        out = []
        for k in sorted(x, key=str):
            p = f"{prefix}.{clean_text(k)}" if prefix else clean_text(k)
            out.extend(_flatten(x[k], p))
        return out
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten(v, prefix))
        return out
    val = "true" if x is True else "false" if x is False else clean_text(x)
    return [f"{prefix}={val}" if prefix else val] if val else []


def features_from_metadata(meta: dict, fields: list[str]) -> list[str]:
    """One feature schema for POIs/profiles; attribute VALUES are not discarded."""
    out = []
    for field in fields:
        value = meta.get(field)
        if field == "categories":
            value = value if value is not None else meta.get("category")
            value = _literal(value)
            if isinstance(value, str):
                value = value.split(",")
        out.extend(_flatten(value))
    return unique_features(out)


def profile_features(value: Any) -> list[str]:
    if isinstance(value, dict):
        if any(k in value for k in ("features", "categories", "attributes", "category")):
            return features_from_metadata(value, ["features", "categories", "attributes"])
        return unique_features(_flatten(value))
    if isinstance(value, list):
        return unique_features(_flatten(value))
    raise ValueError("Each profile must be a feature list or a legacy feature object")


def read_rating(obj: dict) -> float | None:
    for key in RATING_KEYS:
        if key in obj and obj[key] is not None and obj[key] != "":
            if isinstance(obj[key], bool):
                raise ValueError("Boolean is not a rating")
            value = float(obj[key])
            if not math.isfinite(value) or not 1 <= value <= 5:
                raise ValueError(f"Rating outside [1,5]: {value}")
            return value
    return None


@dataclass(frozen=True)
class Record:
    poi_id: str
    rating: float | None
    review: str
    timestamp: str | int | float | None = None


def parse_records(raw: list, order: str) -> list[Record]:
    if not isinstance(raw, list):
        raise ValueError("User records must be a list")
    records = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("Each interaction must be an object")
        review = next((row[k] for k in ("text", "review", "review_text", "content")
                       if row.get(k) is not None), "")
        timestamp = next((row[k] for k in ("timestamp", "time", "date")
                          if row.get(k) not in (None, "")), None)
        records.append(Record(identifier(row), read_rating(row), clean_text(review), timestamp))
    if order == "timestamp_if_complete" and records and all(r.timestamp is not None for r in records):
        def key(r):
            try:
                return float(r.timestamp)
            except (TypeError, ValueError):
                dt = datetime.fromisoformat(str(r.timestamp).replace("Z", "+00:00"))
                return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
        try:
            records.sort(key=key)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Unrecognized timestamps. Fix them or set record_order='input'.") from exc
    return records


def aggregate_ratings(records: list[Record], policy: str) -> dict[str, float]:
    result = {}
    groups = defaultdict(list)
    for r in records:
        if r.rating is not None:
            result[r.poi_id] = r.rating
            groups[r.poi_id].append(r.rating)
    if policy == "mean":
        return {p: sum(v) / len(v) for p, v in groups.items()}
    return result


def load_metadata(path: str) -> dict[str, dict]:
    obj = read_json(path)
    if isinstance(obj, list):
        result = {}
        for row in obj:
            pid = identifier(row)
            if pid in result:
                raise ValueError(f"Duplicate metadata ID: {pid}")
            result[pid] = row
        return result
    if not isinstance(obj, dict) or not all(isinstance(v, dict) for v in obj.values()):
        raise ValueError("Metadata must be an ID-to-object JSON mapping or object list")
    result = {}
    for key, value in obj.items():
        pid = identifier(key)
        if pid in result:
            raise ValueError(f"Duplicate normalized metadata ID: {pid}")
        result[pid] = value
    return result


def belongs(meta: dict, region: str, fields: list[str]) -> bool:
    return any(clean_text(meta.get(k)).casefold() == region.casefold() for k in fields)


@dataclass
class SourceData:
    """Cannot contain a target history or target rating field by construction."""
    records: dict[str, list[Record]]
    features: dict[str, list[str]]
    ratings: dict[str, dict[str, float]]
    pool: list[str]
    direction: str
    fingerprint: str
    user_hashes: dict[str, str]


def load_source(c: dict) -> SourceData:
    d = c["data"]
    metadata = load_metadata(d["metadata"])
    source_meta = {p: m for p, m in metadata.items() if belongs(m, d["source_region"], d["region_fields"])}
    if not source_meta:
        raise ValueError(f"No source POIs match region {d['source_region']!r}. Check region_fields.")
    features = {p: features_from_metadata(m, d["feature_fields"]) for p, m in source_meta.items()}
    if any(not values for values in features.values()):
        raise ValueError("Source POI metadata contains empty features; preprocess the static attributes first")
    records, ratings = {}, {}
    for row in iter_rows(d["interactions"]):
        uid = user_id(row.get("user_id"))
        if uid in records:
            raise ValueError(f"Duplicate user row: {uid}")
        # Only source_key is read. Other interaction fields are not consumed.
        if d["source_key"] not in row:
            raise ValueError(f"Missing source record field: {d['source_key']}")
        seq = parse_records(row[d["source_key"]], d["record_order"])
        if not seq:
            raise ValueError(f"Empty source history for user {uid}")
        if any(r.poi_id not in source_meta for r in seq):
            raise ValueError(f"Source history has missing metadata or wrong-domain POIs for {uid}")
        records[uid] = seq
        ratings[uid] = aggregate_ratings(seq, d["rating_aggregation"])
    if not records:
        raise ValueError("No source users")
    hashes = {u: stable_hash({"direction": c["direction"],
                             "records": [{"poi_id": r.poi_id, "review": r.review} for r in records[u]],
                             "features": {r.poi_id: features[r.poi_id] for r in records[u]}})
              for u in records}
    fingerprint = stable_hash({"users": hashes, "source_features": features,
                               "training_records": {u: [asdict(r) for r in rs] for u, rs in records.items()}})
    return SourceData(records, features, ratings, sorted(source_meta), c["direction"], fingerprint, hashes)


@dataclass
class CandidateData:
    ids: dict[str, list[str]]
    features: dict[str, list[str]]
    report: dict


def load_candidates(c: dict) -> CandidateData:
    raw = read_json(c["data"]["candidates"])
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Candidates must be a nonempty user-ID-to-list JSON object")
    meta = load_metadata(c["data"]["metadata"])
    ids, used = {}, set()
    report = {"users": len(raw), "duplicate_positions": 0, "missing_metadata_positions": 0,
              "short_lists": 0, "raw_length_histogram": {}, "unique_length_histogram": {}}
    for uid, items in raw.items():
        uid = user_id(uid)
        if not isinstance(items, list):
            raise ValueError("Candidate entries must be lists")
        seq = []
        for item in items:
            pid = identifier(item)
            if pid not in meta:
                report["missing_metadata_positions"] += 1
                raise ValueError(f"Candidate metadata missing for {pid}")
            if not belongs(meta[pid], c["data"]["target_region"], c["data"]["region_fields"]):
                raise ValueError(f"Candidate {pid} does not belong to target region")
            seq.append(pid)
        uniq = list(dict.fromkeys(seq))
        if not uniq:
            raise ValueError(f"No candidates remain for {uid}")
        report["duplicate_positions"] += len(seq) - len(uniq)
        report["short_lists"] += int(len(uniq) < c["inference"]["top_k"])
        for key, val in (("raw_length_histogram", len(seq)), ("unique_length_histogram", len(uniq))):
            report[key][str(val)] = report[key].get(str(val), 0) + 1
        if uid in ids:
            raise ValueError(f"Duplicate normalized candidate user: {uid}")
        ids[str(uid)] = uniq
        used.update(uniq)
    features = {p: features_from_metadata(meta[p], c["data"]["feature_fields"]) for p in used}
    if any(not values for values in features.values()):
        raise ValueError("Target candidate metadata contains empty features")
    return CandidateData(ids, features, report)


# ---------------------------------------------------------------------------
# Candidate sampling and recommendation environment
# ---------------------------------------------------------------------------

def sample_source_candidates(source: SourceData, uid: str, cfg: dict, rng: random.Random):
    visited = sorted({r.poi_id for r in source.records[uid]})
    n = min(cfg["max_visited_candidates"], len(visited))
    included = rng.sample(visited, n)
    # Unvisited sampling excludes the FULL history, not only the sampled visits.
    visited_set = set(visited)
    need = cfg["source_candidates"] - len(included)
    if need >= len(source.pool) - len(visited_set):
        noise = [p for p in source.pool if p not in visited_set]
    else:
        noise, seen = [], set()
        while len(noise) < need:
            pid = source.pool[rng.randrange(len(source.pool))]
            if pid not in visited_set and pid not in seen:
                seen.add(pid)
                noise.append(pid)
    result = included + noise
    rng.shuffle(result)
    if not result:
        raise ValueError("No source candidates available")
    return result


def preference_reward(rating: float | None, delta: float) -> float:
    if not math.isfinite(delta):
        raise ValueError("Prototype similarity difference must be finite")
    if rating is not None:
        if isinstance(rating, bool) or not math.isfinite(rating) or not 1 <= rating <= 5:
            raise ValueError("Original source ratings must lie in [1,5]")
        if rating > 3.0:
            return 10.0
        if rating < 3.0:
            return -10.0
    # Includes equality exactly as Eq. (14) in the supplied PDF.
    return 1.0 if delta >= 0.0 else -1.0


class RankingState:
    """No ratings/reviews/ground truth: safe for target inference."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor, candidates: torch.Tensor, steps: int):
        if candidates.ndim != 2 or len(candidates) == 0:
            raise ValueError("Ranking requires a nonempty candidate matrix")
        if x.ndim != 1 or x.shape != y.shape or x.shape[0] != candidates.shape[1]:
            raise ValueError("User prototypes and candidate vectors have incompatible dimensions")
        if x.device != y.device or x.device != candidates.device:
            raise ValueError("Prototypes and candidate vectors must be on the same device")
        if type(steps) is not int or steps < 1:
            raise ValueError("Recommendation length must be a positive integer")
        self.x, self.y, self.candidates = x, y, candidates
        self.max_steps = min(steps, len(candidates))
        self.mask = torch.ones(len(candidates), dtype=torch.bool, device=candidates.device)
        self.selected: list[int] = []
        self.sum_selected = torch.zeros_like(x)

    def vector(self):
        mean = self.sum_selected / len(self.selected) if self.selected else torch.zeros_like(self.x)
        return torch.cat([self.x, self.y, mean])

    def select(self, index: int) -> bool:
        if self.done or index < 0 or index >= len(self.mask) or not bool(self.mask[index]):
            raise ValueError("Invalid, repeated or post-terminal action")
        self.mask[index] = False
        self.selected.append(index)
        self.sum_selected = self.sum_selected + self.candidates[index]
        return self.done

    @property
    def done(self) -> bool:
        return len(self.selected) >= self.max_steps


class SourceRecommendationEnvironment:
    """One source-user episode, with explicit observed and auxiliary feedback.

    State S_t uses the mean of candidates selected BEFORE decision t. Known
    ratings above/below three always override the prototype-based direction.
    The state machine itself never reads user reviews or target-domain labels.
    """

    def __init__(self, user: str, candidate_ids: list[str], positive: torch.Tensor,
                 negative: torch.Tensor, candidate_vectors: torch.Tensor,
                 ratings: dict[str, float], steps: int):
        if len(candidate_ids) != len(candidate_vectors):
            raise ValueError("Candidate IDs and vectors must have the same length")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Source candidates must be unique")
        self.user = user
        self.candidate_ids = list(candidate_ids)
        self.ratings = dict(ratings)
        self.state = RankingState(positive.detach(), negative.detach(),
                                  candidate_vectors.detach(), steps)

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a detached state, candidate matrix and a fresh availability mask."""
        return self.state.vector(), self.state.candidates, self.state.mask.clone()

    def step(self, action: int) -> tuple[str, float, bool]:
        """Select one available candidate and return (POI ID, raw reward, done)."""
        if self.state.done or not 0 <= action < len(self.candidate_ids):
            raise ValueError("Cannot execute an invalid or post-terminal action")
        if not bool(self.state.mask[action]):
            raise ValueError("A POI cannot be recommended twice in the same episode")
        pid = self.candidate_ids[action]
        delta = float(torch.dot(self.state.candidates[action],
                                self.state.x - self.state.y))
        reward = preference_reward(self.ratings.get(pid), delta)
        done = self.state.select(action)
        return pid, reward, done

    @property
    def done(self) -> bool:
        return self.state.done
