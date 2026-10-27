"""Exact Appendix-A prompts, source-only inputs, resumable optional API calls."""
from __future__ import annotations
import json
import os
import re
import time
import urllib.error
import urllib.request
import warnings
import argparse
from pathlib import Path
from urllib.parse import urlsplit
from Config import load_config
from EnvironmentRL import SourceData, profile_features, user_id
from Config import atomic_json, iter_rows, stable_hash, read_json


SYSTEM_PROMPT = """Per-POI analysis. For a given user, examine the deduplicated features of each visited POI together with its corresponding review. Identify liked and disliked features based on the review. A single POI may contain both positive and negative features.

Feature completion. Prioritize the features provided in the POI records. Additional features may be included only when explicitly supported by the corresponding review.

User-level aggregation. After processing all visited POIs, consolidate and deduplicate the extracted features into two user-level profiles, global_liked and global_disliked. Preserve the feature representation format of the POIs and return only JSON."""
USER_PROMPT_TEMPLATE = """Source-domain POI visits of the current user:

{poi_blocks}"""
POI_BLOCK_TEMPLATE = """[POI {index}]
Features: {features}
Review: {review}"""

def prompt_text(name: str) -> str:
    return {"system.txt": SYSTEM_PROMPT,
            "user.txt": USER_PROMPT_TEMPLATE,
            "poi_block.txt": POI_BLOCK_TEMPLATE}[name]


def prompt_hash() -> str:
    return stable_hash([prompt_text(n) for n in ("system.txt", "user.txt", "poi_block.txt")])


def build_messages(source: SourceData, uid: str) -> list[dict]:
    blocks = []
    for i, r in enumerate(source.records[uid], 1):
        blocks.append(prompt_text("poi_block.txt").format(
            index=i,
            features=json.dumps(source.features[r.poi_id], ensure_ascii=False),
            # JSON quoting keeps multi-line reviews inside one explicit input field.
            review=json.dumps(r.review, ensure_ascii=False)))
    return [{"role": "system", "content": prompt_text("system.txt")},
            {"role": "user", "content": prompt_text("user.txt").format(poi_blocks="\n\n".join(blocks))}]


def normalized_profile(row: dict, strict_output: bool = False) -> dict:
    required = {"global_liked", "global_disliked"}
    if not required.issubset(row):
        raise ValueError("Profile missing global_liked/global_disliked")
    if strict_output and set(row) != required:
        raise ValueError("Generated JSON must contain exactly global_liked and global_disliked")
    return {k: profile_features(row[k]) for k in sorted(required)}


def load_profiles(c: dict, source: SourceData) -> tuple[dict, dict]:
    path = Path(c["data"]["profiles"])
    if not path.is_file():
        raise FileNotFoundError(f"Precomputed profile file missing: {path}. Training never calls an API.")
    profiles, hashes = {}, {}
    ignored_score_rows, verified = 0, 0
    for row in iter_rows(path):
        uid = user_id(row.get("user_id"))
        if uid not in source.records:
            continue
        if uid in profiles:
            raise ValueError(f"Duplicate profile user: {uid}")
        prov = row.get("provenance", {})
        if prov:
            if prov.get("direction") != c["direction"] or prov.get("source_hash") != source.user_hashes[uid]:
                raise ValueError(f"Profile provenance disagrees with source direction/records for {uid}")
            if prov.get("scope") != "source_only":
                raise ValueError("Profile is not source-only")
            if prov.get("prompt_hash") not in (None, prompt_hash()):
                raise ValueError("Cached profile prompt differs from the bundled template")
            verified += 1
        profiles[uid] = normalized_profile(row)
        ignored_score_rows += int("poi_scores" in row)
        hashes[uid] = stable_hash(profiles[uid])
    missing = set(source.records) - profiles.keys()
    if missing:
        raise ValueError(f"Missing profiles for {len(missing)} source users; no silent user exclusion")
    if verified < len(profiles):
        warnings.warn("Cached profiles have no complete provenance manifest. Reused under the explicit "
                      "source_only declaration; their extraction scope is not independently verified.")
    report = {"users": len(profiles), "provenance_verified": verified,
              "ignored_legacy_poi_scores": ignored_score_rows,
              "empty_positive": sum(not p["global_liked"] for p in profiles.values()),
              "empty_negative": sum(not p["global_disliked"] for p in profiles.values()),
              "fingerprint": stable_hash(hashes), "prompt_hash": prompt_hash()}
    return profiles, report


def request_profile(messages: list[dict], cfg: dict) -> tuple[dict, dict]:
    model = cfg["model"] or os.getenv("DPFR_LLM_MODEL", "")
    if not model:
        raise ValueError("Set profiling.model or DPFR_LLM_MODEL to your exact DeepSeek-V4 API model ID")
    secret = os.getenv(cfg["api_key_env"], "")
    if not secret:
        raise ValueError(f"Environment variable {cfg['api_key_env']} is not set")
    base = cfg["base_url"].rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("The LLM API endpoint must be HTTPS without credentials in the URL")
    if parsed.query or parsed.fragment:
        raise ValueError("Configure an API base URL without query or fragment")
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    body = {"model": model, "messages": messages,
            "temperature": cfg["temperature"], "max_tokens": cfg["max_output_tokens"],
            "response_format": {"type": "json_object"}}
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + secret,
                                              "Content-Type": "application/json"})
    last = "request failed"
    for attempt in range(cfg["retries"]):
        try:
            with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
                obj = json.load(response)
            choice = obj["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("LLM output truncated; increase max_output_tokens")
            content = choice["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            value = normalized_profile(json.loads(content), strict_output=True)
            return value, {"requested_model": model, "response_model": obj.get("model", model),
                           "usage": obj.get("usage", {}), "temperature": cfg["temperature"]}
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(f"Profile request failed: {last}. Credentials and response body omitted.") from None
        except (urllib.error.URLError, TimeoutError):
            last = "network timeout or connection failure"
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            last = "invalid, empty or truncated JSON response"
        if attempt + 1 < cfg["retries"]:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Profile generation failed after {cfg['retries']} attempts: {last}")


def generate_profiles(c: dict, allow_api: bool = False, dry_run: str | None = None) -> dict:
    from EnvironmentRL import load_source
    source = load_source(c)
    dest = Path(c["data"]["profiles"])
    if dry_run:
        uid = sorted(source.records)[0]
        atomic_json(dry_run, {"direction": c["direction"], "messages": build_messages(source, uid),
                              "output_schema": {"global_liked": [], "global_disliked": []}})
        return {"dry_run": str(dry_run), "api_calls": 0}
    if not allow_api:
        raise ValueError("API calls are paid and send source reviews externally; explicitly pass --allow-api")
    protected = {Path(c["data"][k]).resolve() for k in ("metadata", "interactions", "candidates")}
    if dest.resolve() in protected:
        raise ValueError("Profile output must not overwrite input records, metadata or candidates")
    completed = {}
    if dest.exists():
        for row in iter_rows(dest):
            uid = user_id(row.get("user_id"))
            if uid in completed:
                raise ValueError("Duplicate cached profile row")
            completed[uid] = row
    missing = [u for u in sorted(source.records) if u not in completed]
    # Check existing provenance, without regenerating or overwriting cached entries.
    for u in set(source.records) & completed.keys():
        row = completed[u]
        normalized_profile(row)
        p = row.get("provenance", {})
        if p and (p.get("scope") != "source_only" or p.get("direction") != c["direction"] or p.get("source_hash") != source.user_hashes[u] or p.get("prompt_hash") not in (None, prompt_hash())):
            raise ValueError("Existing cache belongs to a different source scope")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # JSONL: append per completed user. An incomplete trailing line fails loudly on restart.
    if dest.suffix.lower() != ".jsonl":
        raise ValueError("Generated profile caches must use .jsonl; precomputed .json is supported for training")
    if missing and dest.exists() and dest.stat().st_size:
        with dest.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) not in (b"\n", b"\r")
        if needs_newline:
            with dest.open("ab") as handle:
                handle.write(b"\n")
    calls = 0
    for index, uid in enumerate(missing, 1):
        messages = build_messages(source, uid)
        if sum(len(m["content"]) for m in messages) > c["profiling"]["max_input_chars"]:
            raise ValueError("Profile input exceeds max_input_chars. No reviews were silently truncated.")
        p, api = request_profile(messages, c["profiling"])
        row = {"user_id": uid, **p, "provenance": {
            "scope": "source_only", "direction": c["direction"],
            "source_hash": source.user_hashes[uid], "prompt_hash": prompt_hash(), **api}}
        with dest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        calls += 1
        print(f"Profiles completed: {index}/{len(missing)} (source-only cache)", flush=True)
    return {"users": len(source.records), "new_profiles": calls, "reused_profiles": len(source.records) - calls}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract source-only positive/negative user profiles")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--preview", metavar="FILE", help="Write one real input prompt without calling the API")
    parser.add_argument("--allow-api", action="store_true", help="Allow source review text to be sent to the configured API")
    args = parser.parse_args()
    c = load_config(args.config)
    if args.preview:
        out = Path(args.preview).expanduser().resolve()
        protected = {Path(args.config).expanduser().resolve()}
        protected.update(Path(c["data"][key]).resolve()
                         for key in ("interactions", "metadata", "profiles", "candidates"))
        protected.update(Path(c["output_dir"]) / name for name in
                         ("model.pt", "training_state.pt", "prototypes.jsonl", "config.json"))
        if out in protected:
            parser.error("Prompt preview must not overwrite a data, config or model file")
    report = generate_profiles(c, allow_api=args.allow_api, dry_run=args.preview)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
