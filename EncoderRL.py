"""DPFR shared Transformer encoder and dual-prototype representation learning.

A single feature stream is used for positive profiles, negative profiles and POIs.
The three objectives are signed preference alignment, prototype separation and
cross-user uniformity. All returned representations have unit L2 norm.
"""
from __future__ import annotations
import hashlib
import math
import re
import unicodedata
import torch
from torch import nn
from torch.nn import functional as F


class FeatureTokenizer:
    """Deterministic hashing, unlike Python's process-randomized hash().

    Not a pretrained language tokenizer. Algorithm/version travel with checkpoints.
    0: padding; 1: empty feature sequence; >=2: BLAKE2b-hashed lexical tokens.
    """
    algorithm = "nfkc_regex_blake2b8_v1"

    def __init__(self, vocab_size: int, max_tokens: int):
        if vocab_size < 8 or max_tokens < 1:
            raise ValueError("Invalid tokenizer vocabulary size or maximum length")
        self.vocab_size, self.max_tokens = vocab_size, max_tokens

    def encode(self, features: list[str]) -> list[int]:
        text = unicodedata.normalize("NFKC", " ; ".join(features)).casefold()
        words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        ids = [2 + int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "big")
               % (self.vocab_size - 2) for w in words[:self.max_tokens]]
        return ids or [1]

    def batch(self, sequences: list[list[str]], device=None) -> torch.Tensor:
        if not sequences:
            raise ValueError("Cannot tokenize an empty batch")
        ids = [self.encode(s) for s in sequences]
        result = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long, device=device)
        for i, seq in enumerate(ids):
            result[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        return result

    def state_dict(self) -> dict:
        return {"algorithm": self.algorithm, "vocab_size": self.vocab_size, "max_tokens": self.max_tokens}

    @classmethod
    def from_state(cls, state):
        if state["algorithm"] != cls.algorithm:
            raise ValueError("Unknown tokenizer algorithm")
        return cls(state["vocab_size"], state["max_tokens"])


class SharedEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["dimension"]
        if d <= 0 or d % 2 or d % cfg["heads"]:
            raise ValueError("Encoder dimension must be even and divisible by heads")
        self.dimension = d
        self.embedding = nn.Embedding(cfg["vocab_size"], d, padding_idx=0)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        pos = torch.arange(cfg["max_tokens"] + 1).unsqueeze(1)
        freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(cfg["max_tokens"] + 1, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer("position", pe)
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], d * 4, cfg["dropout"],
                                           activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        self.input_dropout = nn.Dropout(cfg["dropout"])
        self.projector = nn.Sequential(nn.Linear(d, 2 * d), nn.LayerNorm(2 * d), nn.GELU(),
                                       nn.Dropout(cfg["dropout"]), nn.Linear(2 * d, d))
        # Avoid identical initial parameters in cloned Transformer layers.
        for block in self.transformer.layers:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long or token_ids.dim() != 2 or token_ids.shape[1] + 1 > self.position.shape[0]:
            raise ValueError("Invalid token batch shape or sequence too long")
        x = self.embedding(token_ids) * math.sqrt(self.dimension)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1)
        x = self.input_dropout(x + self.position[:x.shape[1]])
        # Padding embeddings alone do NOT mask attention after positional encoding.
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device),
                         token_ids.eq(0)], dim=1)
        x = self.transformer(x.contiguous(), src_key_padding_mask=pad)
        return F.normalize(self.projector(x[:, 0]), dim=-1, eps=1e-8)


def scheduled_weights(progress: float) -> tuple[float, float, float]:
    progress = min(max(float(progress), 0.0), 1.0)
    decay = 0.5 * (1 + math.cos(math.pi * progress))
    return 0.4 - 0.2 * decay, 0.4 + 0.2 * decay, 0.2


def _uniformity(x: torch.Tensor, temperature: float) -> torch.Tensor:
    """log mean exp over ordered DISTINCT-user pairs in this minibatch."""
    if len(x) < 2:
        return x.sum() * 0.0
    logits = (x @ x.T) / temperature
    vals = logits[~torch.eye(len(x), dtype=torch.bool, device=x.device)]
    return torch.logsumexp(vals, dim=0) - math.log(vals.numel())


def matching_probabilities(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                           temperature: float) -> torch.Tensor:
    """Return [P_like, P_dislike] for each aligned user/POI pair."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be finite and strictly positive")
    if x.shape != y.shape or x.shape != z.shape or x.ndim != 2:
        raise ValueError("Prototype and POI batches must share shape [pairs,d]")
    positive = (x * z).sum(dim=-1)
    negative = (y * z).sum(dim=-1)
    return torch.stack((positive, negative), dim=-1).div(temperature).softmax(dim=-1)


def prototype_separation(x: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
    """L_sep = E[softplus(x^T y / tau)] over distinct users in the batch."""
    return F.softplus((x * y).sum(dim=-1) / temperature).mean()


def encoder_objective(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                      user_indices: torch.Tensor, rewards: torch.Tensor,
                      temperature: float, progress: float) -> dict[str, torch.Tensor]:
    """Combine the three paper objectives without rescaling source rewards.

    x/y contain one pair of prototypes per DISTINCT user. user_indices map each
    selected POI in z to its user; repeated feedback from one user therefore does
    not accidentally become an inter-user uniformity pair.
    """
    if len(x) == 0 or len(z) == 0 or x.shape != y.shape:
        raise ValueError("The encoder objective requires nonempty aligned inputs")
    if rewards.shape != (len(z),) or user_indices.shape != (len(z),):
        raise ValueError("One reward and user index are required per feedback POI")
    if user_indices.dtype != torch.long:
        raise ValueError("Feedback user indices must be int64")
    if bool((user_indices < 0).any()) or bool((user_indices >= len(x)).any()):
        raise ValueError("Feedback references a user outside the prototype batch")
    if rewards.requires_grad or not bool(torch.isfinite(rewards).all()):
        raise ValueError("Rewards are finite constants, not differentiable labels")
    probabilities = matching_probabilities(x[user_indices], y[user_indices], z, temperature)
    alignment = -(rewards * (probabilities[:, 0] - probabilities[:, 1])).mean()
    separation = prototype_separation(x, y, temperature)
    uniformity = _uniformity(x, temperature) + _uniformity(y, temperature)
    wa, ws, wu = scheduled_weights(progress)
    return {"loss": wa * alignment + ws * separation + wu * uniformity,
            "alignment": alignment, "separation": separation, "uniformity": uniformity}


# ---------------------------------------------------------------------------
# Batched feature extraction and encoder refinement
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_features(encoder, tokenizer, feature_lists, device, batch_size=256):
    was_training = encoder.training
    encoder.eval()
    result = []
    try:
        for i in range(0, len(feature_lists), batch_size):
            result.append(encoder(tokenizer.batch(feature_lists[i:i + batch_size], device)).cpu())
    finally:
        encoder.train(was_training)
    if not result:
        return torch.empty(0, encoder.dimension)
    return torch.cat(result)


def profile_pair_features(profiles, uid):
    return profiles[uid]["global_liked"], profiles[uid]["global_disliked"]


def train_encoder(encoder, tokenizer, source, profiles, feedback, optimizer,
                  cfg, model_cfg, device, progress, epochs, batch_size, rng):
    if not feedback:
        return {"loss": 0.0, "alignment": 0.0, "separation": 0.0, "uniformity": 0.0, "batches": 0}
    encoder.train()
    totals = {"loss": 0.0, "alignment": 0.0, "separation": 0.0, "uniformity": 0.0}
    count = 0
    for _ in range(epochs):
        order = list(range(len(feedback)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            records = [feedback[k] for k in order[start:start + batch_size]]
            users = list(dict.fromkeys(r[0] for r in records))
            items = list(dict.fromkeys(r[1] for r in records))
            user_index = {u: i for i, u in enumerate(users)}
            item_index = {p: i for i, p in enumerate(items)}
            pairs = [profile_pair_features(profiles, u) for u in users]
            feature_lists = [p[0] for p in pairs]
            feature_lists += [p[1] for p in pairs]
            feature_lists += [source.features[p] for p in items]
            all_vectors = encoder(tokenizer.batch(feature_lists, device))
            n = len(users)
            x = all_vectors[:n]
            y = all_vectors[n:2 * n]
            item_vectors = all_vectors[2 * n:]
            uidx = torch.tensor([user_index[r[0]] for r in records], device=device)
            pidx = torch.tensor([item_index[r[1]] for r in records], device=device)
            rewards = torch.tensor([r[2] for r in records], dtype=torch.float32, device=device)
            result = encoder_objective(x, y, item_vectors[pidx], uidx, rewards,
                                       model_cfg["temperature"], progress)
            if not all(torch.isfinite(v).all() for v in result.values()):
                raise FloatingPointError("Non-finite encoder objective")
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg["encoder_grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            for key in totals:
                totals[key] += float(result[key].detach())
            count += 1
    return {**{k: v / max(count, 1) for k, v in totals.items()}, "batches": count}


class RepresentationCache:
    """Detached CPU cache invalidated on EVERY encoder update."""
    def __init__(self, encoder, tokenizer, source, profiles, device):
        self.encoder, self.tokenizer, self.source = encoder, tokenizer, source
        self.profiles, self.device = profiles, device
        self.items, self.users = {}, {}

    def clear(self):
        self.items.clear()
        self.users.clear()

    def get(self, uid, candidate_ids):
        missing = list(dict.fromkeys(p for p in candidate_ids if p not in self.items))
        if missing:
            vectors = encode_features(self.encoder, self.tokenizer,
                                      [self.source.features[p] for p in missing], self.device)
            self.items.update(zip(missing, vectors))
        if uid not in self.users:
            a, b = profile_pair_features(self.profiles, uid)
            vectors = encode_features(self.encoder, self.tokenizer, [a, b], self.device)
            self.users[uid] = (vectors[0], vectors[1])
        x, y = self.users[uid]
        return x.to(self.device), y.to(self.device), torch.stack([self.items[p] for p in candidate_ids]).to(self.device)


