"""Train DPFR, export dual user prototypes, and recommend target candidates.

Examples:
    python Train.py --config config.json --check
    python Train.py --config config.json
    python Train.py --config config.json --mode predict
    python Train.py --config config.json --resume outputs/yelp_p_n/training_state.pt
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import torch
from Config import (load_config, validate_config, atomic_json, atomic_jsonl,
                    device_for, save_torch, set_seed, stable_hash)
from EncoderRL import (SharedEncoder, FeatureTokenizer, encode_features,
                       RepresentationCache, train_encoder)
from AgentRL import ActorCritic, PPOAgent, residual_weight_at
from EnvironmentRL import (load_source, load_candidates, RankingState,
                           SourceRecommendationEnvironment, sample_source_candidates)
from EmotionParser import load_profiles, prompt_hash

class Trainer:
    """DPFR source training. Neither target candidates nor target labels are read."""
    def __init__(self, config: dict):
        validate_config(config)
        self.c, self.t = config, config["training"]
        self.out = Path(config["output_dir"])
        set_seed(config["seed"], config["deterministic"], config["num_threads"])
        self.device = device_for(config["device"])
        self.source = load_source(config)
        self.profiles, self.profile_report = load_profiles(config, self.source)
        self.tokenizer = FeatureTokenizer(config["model"]["vocab_size"], config["model"]["max_tokens"])
        self.encoder = SharedEncoder(config["model"]).to(self.device)
        self.policy = ActorCritic(config["model"]).to(self.device)
        self.ppo = PPOAgent(self.policy, self.t)
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.t["encoder_warmup_lr"],
                                           weight_decay=self.t["encoder_weight_decay"])
        self.cache = RepresentationCache(self.encoder, self.tokenizer, self.source, self.profiles, self.device)
        self.rng, self.encoder_rng = random.Random(config["seed"] + 17), random.Random(config["seed"] + 31)
        self.order = sorted(self.source.records)
        self.rng.shuffle(self.order)
        self.user_position = self.episode = 0
        self.feedback = []
        self.warmed = False
        self.fingerprint = stable_hash({
            "format": "dpfr-core-training-v1", "source": self.source.fingerprint,
            "profiles": self.profile_report["fingerprint"], "model": config["model"],
            "seed": config["seed"], "deterministic": config["deterministic"],
            "training": {k: v for k, v in self.t.items() if k != "checkpoint_every"},
            "feature_fields": config["data"]["feature_fields"],
        })

    def log(self, values):
        with (self.out / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"episode": self.episode, **values}, allow_nan=False) + "\n")

    def checkpoint(self):
        save_torch(self.out / "training_state.pt", {
            "format": "dpfr-core-training-v1", "fingerprint": self.fingerprint,
            "episode": self.episode, "warmed": self.warmed,
            "encoder": self.encoder.state_dict(), "policy": self.policy.state_dict(),
            "encoder_optimizer": self.optimizer.state_dict(), "ppo_optimizer": self.ppo.optimizer.state_dict(),
            "feedback": self.feedback, "user_order": self.order, "user_position": self.user_position,
            "sampler_rng": self.rng.getstate(), "encoder_rng": self.encoder_rng.getstate(),
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        })

    def resume(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != "dpfr-core-training-v1" or saved["fingerprint"] != self.fingerprint:
            raise ValueError("Training checkpoint, source data or configuration does not match")
        self.encoder.load_state_dict(saved["encoder"])
        self.policy.load_state_dict(saved["policy"])
        self.optimizer.load_state_dict(saved["encoder_optimizer"])
        self.ppo.optimizer.load_state_dict(saved["ppo_optimizer"])
        self.episode, self.warmed = saved["episode"], saved["warmed"]
        self.feedback = saved["feedback"]
        self.order, self.user_position = saved["user_order"], saved["user_position"]
        self.rng.setstate(saved["sampler_rng"])
        self.encoder_rng.setstate(saved["encoder_rng"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if self.device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        self.cache.clear()
        log = self.out / "training_log.jsonl"
        if log.exists():
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            log.write_text("".join(json.dumps(row) + "\n" for row in rows if row.get("episode", 0) <= self.episode), encoding="utf-8")

    def warmup(self):
        records = [(u, p, 10.0 if value > 3 else -10.0)
                   for u in sorted(self.source.records) for p, value in self.source.ratings[u].items() if value != 3.0]
        if self.t["warmup_epochs"] and not records:
            raise ValueError("Encoder warm-up needs observed source ratings above/below 3")
        for epoch in range(self.t["warmup_epochs"]):
            losses = train_encoder(self.encoder, self.tokenizer, self.source, self.profiles, records,
                                   self.optimizer, self.t, self.c["model"], self.device, 0.0, 1,
                                   self.t["warmup_batch_size"], self.encoder_rng)
            self.log({"stage": "warmup", "warmup_epoch": epoch + 1, **losses})
            print(f"Warm-up {epoch + 1}/{self.t['warmup_epochs']}: loss={losses['loss']:.5f}", flush=True)
        for group in self.optimizer.param_groups:
            group["lr"] = self.t["encoder_feedback_lr"]
        self.warmed = True
        self.cache.clear()

    def rollout(self, uid, candidates, weight):
        x, y, z = self.cache.get(uid, candidates)
        environment = SourceRecommendationEnvironment(
            uid, candidates, x, y, z, self.source.ratings[uid], self.t["decision_steps"])
        rollout, records = self.ppo.collect_rollout(environment, weight)
        self.feedback.extend(records)
        return rollout

    @torch.no_grad()
    def export(self):
        # Recompute EVERY user's two vectors once under the FINAL source encoder.
        users = sorted(self.profiles)
        x = encode_features(self.encoder, self.tokenizer,
                            [self.profiles[u]["global_liked"] for u in users], self.device)
        y = encode_features(self.encoder, self.tokenizer,
                            [self.profiles[u]["global_disliked"] for u in users], self.device)
        payload = {
            "format": "dpfr-core-inference-v1", "direction": self.c["direction"],
            "source_region": self.c["data"]["source_region"], "model_config": self.c["model"],
            "tokenizer": self.tokenizer.state_dict(),
            "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
            "actor": {k: v.detach().cpu() for k, v in self.policy.actor.state_dict().items()},
            "user_ids": users, "positive_prototypes": x, "negative_prototypes": y,
            "residual_weight": self.t["residual_end"], "episode": self.episode,
            "source_fingerprint": self.source.fingerprint, "run_fingerprint": self.fingerprint,
            "feature_fields": self.c["data"]["feature_fields"], "prompt_hash": prompt_hash(),
        }
        save_torch(self.out / "model.pt", payload)
        atomic_jsonl(self.out / "prototypes.jsonl", (
            {"user_id": u, "positive_prototype": x[i].tolist(), "negative_prototype": y[i].tolist()}
            for i, u in enumerate(users)))

    def train(self, resume: str | None = None) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        if resume:
            if Path(resume).expanduser().resolve().parent != self.out.resolve():
                raise ValueError("Resume in the checkpoint's original output directory")
            self.resume(resume)
        elif any((self.out / name).exists() for name in ("training_log.jsonl", "model.pt", "training_state.pt")):
            raise FileExistsError("Existing training outputs; use --resume or a new output_dir")
        atomic_json(self.out / "config.json", self.c)
        if not self.warmed:
            self.warmup()
            self.checkpoint()
        try:
            while self.episode < self.t["episodes"]:
                uid = self.order[self.user_position]
                self.user_position += 1
                if self.user_position == len(self.order):
                    self.rng.shuffle(self.order)
                    self.user_position = 0
                candidates = sample_source_candidates(self.source, uid, self.t, self.rng)
                weight = residual_weight_at(self.episode, self.t["episodes"],
                                            self.t["residual_start"], self.t["residual_end"])
                rollout = self.rollout(uid, candidates, weight)
                losses = self.ppo.update(rollout)  # Update policy before moving the feature space.
                self.episode += 1
                row = {"stage": "joint", **losses, "return": float(rollout.rewards.sum()),
                       "steps": len(rollout.actions), "residual_weight": weight}
                boundary = self.episode % self.t["feedback_users"] == 0
                if boundary and self.feedback:
                    result = train_encoder(self.encoder, self.tokenizer, self.source, self.profiles,
                                           self.feedback, self.optimizer, self.t, self.c["model"], self.device,
                                           self.episode / self.t["episodes"], self.t["feedback_epochs"],
                                           self.t["feedback_batch_size"], self.encoder_rng)
                    row.update({f"encoder_{k}": v for k, v in result.items()})
                    row["feedback_records"] = len(self.feedback)
                    self.feedback.clear()
                    self.cache.clear()
                self.log(row)
                if boundary or self.episode == 1 or self.episode == self.t["episodes"]:
                    print(f"{self.c['direction']} {self.episode}/{self.t['episodes']} "
                          f"return={row['return']:.2f} actor={losses['actor_loss']:.5f}", flush=True)
                if self.t["checkpoint_every"] and self.episode % self.t["checkpoint_every"] == 0:
                    self.checkpoint()
        except KeyboardInterrupt:
            print("Interrupted. Resume the last completed training_state.pt; partial updates were not saved.", flush=True)
            raise
        # Keep an incomplete final block in the resume state; no extra off-schedule update.
        self.checkpoint()
        self.export()
        print(f"Saved {self.out / 'model.pt'} and {self.out / 'prototypes.jsonl'}", flush=True)
        return self.out / "model.pt"


# ---------------------------------------------------------------------------
# Frozen target-domain inference and result export
# ---------------------------------------------------------------------------

class FrozenRecommender:
    def __init__(self, checkpoint: str | Path, device="cpu"):
        self.device = device_for(str(device))
        self.artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if self.artifact.get("format") != "dpfr-core-inference-v1":
            raise ValueError("Use model.pt exported by this release, not training_state.pt or legacy weights")
        a = self.artifact
        self.tokenizer = FeatureTokenizer.from_state(a["tokenizer"])
        self.encoder = SharedEncoder(a["model_config"]).to(self.device).eval()
        self.encoder.load_state_dict(a["encoder"])
        self.policy = ActorCritic(a["model_config"]).to(self.device).eval()
        self.policy.actor.load_state_dict(a["actor"])
        self.encoder.requires_grad_(False)
        self.policy.requires_grad_(False)
        self.users = {u: i for i, u in enumerate(a["user_ids"])}
        self.x, self.y, self.weight = a["positive_prototypes"], a["negative_prototypes"], a["residual_weight"]
        if len(self.users) != len(a["user_ids"]) or self.x.shape != self.y.shape or self.x.shape != (len(self.users), a["model_config"]["dimension"]):
            raise ValueError("Invalid prototype bank in checkpoint")
        if not torch.isfinite(self.x).all() or not torch.isfinite(self.y).all():
            raise ValueError("Checkpoint contains non-finite user prototypes")

    @torch.no_grad()
    def recommend(self, uid: str, ids: list[str], vectors: torch.Tensor, top_k: int) -> list[str]:
        if uid not in self.users:
            raise ValueError(f"User {uid} has no source-domain prototypes in this model")
        if len(ids) != len(set(ids)) or len(ids) != len(vectors):
            raise ValueError("Candidate IDs must be unique and aligned with their vectors")
        i = self.users[uid]
        z = vectors.to(self.device)
        state = RankingState(self.x[i].to(self.device), self.y[i].to(self.device), z, top_k)
        while not state.done:
            action = self.policy.greedy_action(state.vector(), z, state.mask, self.weight)
            # No sampling, tie noise, Critic evaluation or reward computation.
            state.select(action)
        return [ids[i] for i in state.selected]


def predict(c: dict, checkpoint: str | Path, output: str | Path | None = None,
            overwrite: bool = False) -> Path:
    set_seed(c["seed"], c["deterministic"], c["num_threads"])
    model = FrozenRecommender(checkpoint, c["device"])
    if model.artifact["source_region"].casefold() != c["data"]["source_region"].casefold():
        raise ValueError("Configured source region does not match the source prototype bank")
    if model.artifact["feature_fields"] != c["data"]["feature_fields"]:
        raise ValueError("POI feature fields differ from the trained encoder")
    candidates = load_candidates(c)
    missing = set(candidates.ids) - model.users.keys()
    if missing:
        raise ValueError(f"{len(missing)} candidate users lack source prototypes: {sorted(missing)[:3]}")
    out = Path(output or Path(c["output_dir"]) / "recommendations.json").expanduser().resolve()
    protected = {Path(checkpoint).resolve(), *(Path(c['data'][key]).resolve() for key in
                  ("interactions", "metadata", "profiles", "candidates"))}
    protected.update(Path(c["output_dir"]).resolve() / name for name in
                     ("model.pt", "prototypes.jsonl", "training_state.pt", "config.json", "training_log.jsonl"))
    if out in protected:
        raise ValueError("Prediction output must not overwrite data, prototypes, configuration or weights")
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists. Choose another output or explicitly pass --overwrite")
    print(f"External candidates: {len(candidates.ids)} users; "
          f"removed {candidates.report['duplicate_positions']} duplicate positions", flush=True)
    # Encode each distinct target POI once. Source profiles/records are not loaded.
    ids = sorted(candidates.features)
    vectors = encode_features(model.encoder, model.tokenizer,
                              [candidates.features[p] for p in ids], model.device,
                              c["inference"]["encode_batch_size"])
    index = {p: i for i, p in enumerate(ids)}
    results = {}
    for n, (uid, user_ids) in enumerate(candidates.ids.items(), 1):
        z = vectors[torch.tensor([index[p] for p in user_ids], dtype=torch.long)]
        results[uid] = {"recommended_pois": model.recommend(uid, user_ids, z, c["inference"]["top_k"])}
        if n % 100 == 0 or n == len(candidates.ids):
            print(f"Recommendations: {n}/{len(candidates.ids)} users", flush=True)
    atomic_json(out, results)
    print(f"Saved {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="DPFR source training and frozen target recommendation")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--mode", choices=("train", "predict"), default="train")
    parser.add_argument("--resume", help="Resume from training_state.pt")
    parser.add_argument("--check", action="store_true", help="Validate source data and cached profiles only")
    parser.add_argument("--checkpoint", help="Inference model.pt; default: output_dir/model.pt")
    parser.add_argument("--candidates", help="Override external target candidate JSON")
    parser.add_argument("--metadata", help="Override static POI metadata JSON for prediction")
    parser.add_argument("--output", help="Override recommendation JSON output")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing an existing recommendation file")
    args = parser.parse_args()
    c = load_config(args.config)
    if args.mode == "predict":
        if args.resume or args.check:
            parser.error("--resume and --check are training options")
        for key in ("candidates", "metadata"):
            value = getattr(args, key)
            if value:
                c["data"][key] = str(Path(value).expanduser().resolve())
        checkpoint = args.checkpoint or Path(c["output_dir"]) / "model.pt"
        predict(c, checkpoint, args.output, args.overwrite)
        return
    if any((args.checkpoint, args.candidates, args.metadata, args.output, args.overwrite)):
        parser.error("Candidate/output overrides require --mode predict")
    if args.check:
        source = load_source(c)
        _, report = load_profiles(c, source)
        print(f"Source users: {len(source.records)}; source POIs: {len(source.pool)}; "
              f"profiles: {report['users']}; dimension: {c['model']['dimension']}")
        print("Source validation passed. No API request or target-label access was performed.")
        return
    Trainer(c).train(args.resume)


if __name__ == "__main__":
    main()
