"""DPFR Actor-Critic policy, source rollouts and PPO optimization.

The Actor produces a residual vector, not fixed POI-ID logits. A query formed
as x + w_r * Actor(S) is matched with the current candidates. PPO operates on
collection-time feature snapshots before the encoder is updated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import math
import torch
from torch import nn

if TYPE_CHECKING:
    from EnvironmentRL import SourceRecommendationEnvironment

class ActorCritic(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        d = model_cfg["dimension"]
        self.dimension = d
        self.temperature = model_cfg["temperature"]
        a, b = model_cfg["actor_hidden"]
        self.actor = nn.Sequential(nn.Linear(3 * d, a), nn.LayerNorm(a), nn.Tanh(),
                                   nn.Linear(a, b), nn.Tanh(), nn.Linear(b, d))
        self.critic = nn.Sequential(nn.Linear(3 * d, a), nn.LayerNorm(a), nn.Tanh(),
                                    nn.Linear(a, b), nn.Tanh(), nn.Linear(b, 1))

    def query(self, states: torch.Tensor, residual_weight: float) -> torch.Tensor:
        """Construct q_t = x_i + w_r A_theta(S_t), without an extra normalization."""
        if states.ndim not in (1, 2) or states.shape[-1] != 3 * self.dimension:
            raise ValueError("State must have shape [3d] or [batch,3d]")
        if not math.isfinite(residual_weight) or residual_weight < 0:
            raise ValueError("Residual weight must be finite and nonnegative")
        states = states.unsqueeze(0) if states.ndim == 1 else states
        return states[:, :self.dimension] + residual_weight * self.actor(states)

    def action_logits(self, states: torch.Tensor, candidates: torch.Tensor,
                      mask: torch.Tensor, residual_weight: float) -> torch.Tensor:
        """Score only available candidates; source/target POI IDs are not parameters."""
        query = self.query(states, residual_weight)
        if candidates.ndim not in (2, 3) or candidates.shape[-1] != self.dimension:
            raise ValueError("Candidate vectors must have shape [items,d] or [batch,items,d]")
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("The action mask must be a boolean matrix")
        expected = (query.shape[0], candidates.shape[-2])
        if tuple(mask.shape) != expected:
            raise ValueError("Action mask shape does not match the states/candidates")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("Cannot choose an action when all candidates are masked")
        if candidates.ndim == 2:
            scores = query @ candidates.T
        else:
            if candidates.shape[0] != query.shape[0]:
                raise ValueError("Candidate batch size differs from state batch size")
            scores = torch.bmm(query.unsqueeze(1), candidates.transpose(1, 2)).squeeze(1)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Non-finite candidate matching scores")
        # Only the paper's temperature is applied: there is no extra factor of 10.
        return (scores / self.temperature).masked_fill(~mask, float("-inf"))

    def distribution(self, states: torch.Tensor, candidates: torch.Tensor,
                     mask: torch.Tensor, residual_weight: float):
        logits = self.action_logits(states, candidates, mask, residual_weight)
        return torch.distributions.Categorical(logits=logits)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        """The Critic is used for source policy optimization, not target scoring."""
        return self.critic(states).squeeze(-1)

    @torch.no_grad()
    def greedy_action(self, state: torch.Tensor, candidates: torch.Tensor,
                      mask: torch.Tensor, weight: float) -> int:
        """Deterministic inference. Equal scores follow the supplied candidate order."""
        logits = self.action_logits(state, candidates, mask, weight)
        if len(logits) != 1:
            raise ValueError("greedy_action expects exactly one recommendation state")
        return int(logits.argmax(dim=-1).item())


@dataclass
class Rollout:
    states: torch.Tensor
    candidates: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    residual_weight: float

    def validate(self) -> None:
        """Check trajectory alignment before using old probabilities in PPO."""
        length = len(self.actions)
        if length == 0 or self.states.ndim != 2 or self.masks.ndim != 2:
            raise ValueError("A rollout needs nonempty state and mask matrices")
        if self.candidates.ndim != 2:
            raise ValueError("A rollout uses one fixed candidate matrix")
        if self.states.shape[0] != length or self.masks.shape != (length, len(self.candidates)):
            raise ValueError("States, actions and candidate masks do not align")
        for vector in (self.actions, self.old_logprobs, self.old_values, self.rewards, self.terminals):
            if vector.shape != (length,):
                raise ValueError("Every per-step rollout field must have shape [steps]")
        if self.actions.dtype != torch.long or self.masks.dtype != torch.bool:
            raise ValueError("Actions must be int64 and masks must be boolean")
        if self.terminals.dtype != torch.bool or not bool(self.terminals[-1]):
            raise ValueError("A source-user rollout must end with a terminal step")
        if bool((self.actions < 0).any()) or bool((self.actions >= len(self.candidates)).any()):
            raise ValueError("A rollout contains an out-of-range action")
        if not bool(self.masks.gather(1, self.actions[:, None]).all()):
            raise ValueError("A rollout contains a masked action")
        for tensor in (self.states, self.candidates, self.old_logprobs, self.old_values, self.rewards):
            if tensor.requires_grad or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Stored rollout features must be finite detached snapshots")



def gae(rewards, values, terminals, gamma: float, lam: float, bootstrap: float = 0.0):
    if rewards.ndim != 1 or values.shape != rewards.shape or terminals.shape != rewards.shape:
        raise ValueError("GAE expects aligned one-dimensional trajectories")
    if terminals.dtype != torch.bool or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("Invalid terminal mask or GAE coefficients")
    advantage = torch.zeros_like(rewards)
    running = rewards.new_tensor(0.0)
    next_value = rewards.new_tensor(bootstrap)
    for i in range(len(rewards) - 1, -1, -1):
        keep = (~terminals[i]).to(rewards.dtype)
        delta = rewards[i] + gamma * next_value * keep - values[i]
        running = delta + gamma * lam * keep * running
        advantage[i] = running
        next_value = values[i]
    return advantage, advantage + values


def residual_weight_at(episode: int, total_episodes: int, start: float, end: float) -> float:
    """Linear schedule evaluated once per rollout, then held fixed for its update."""
    progress = min(max(episode / max(total_episodes - 1, 1), 0.0), 1.0)
    return start + progress * (end - start)


class PPOAgent:
    def __init__(self, policy: ActorCritic, cfg: dict):
        self.policy, self.cfg = policy, cfg
        self.optimizer = torch.optim.Adam([
            {"params": policy.actor.parameters(), "lr": cfg["actor_lr"]},
            {"params": policy.critic.parameters(), "lr": cfg["critic_lr"]}])

    def update(self, r: Rollout) -> dict:
        r.validate()
        self.policy.train()
        advantage, returns = gae(r.rewards, r.old_values, r.terminals,
                                 self.cfg["gamma"], self.cfg["gae_lambda"])
        if self.cfg["normalize_advantage"] and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        advantage, returns = advantage.detach(), returns.detach()
        total_a = total_v = 0.0
        for _ in range(self.cfg["ppo_epochs"]):
            # Re-evaluation uses the SAME states, candidate snapshot, mask and w_r.
            dist = self.policy.distribution(r.states, r.candidates, r.masks, r.residual_weight)
            ratio = (dist.log_prob(r.actions) - r.old_logprobs).exp()
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - self.cfg["ppo_clip"], 1 + self.cfg["ppo_clip"]) * advantage
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            values = self.policy.value(r.states)
            value_clipped = r.old_values + (values - r.old_values).clamp(
                -self.cfg["value_clip"], self.cfg["value_clip"])
            critic_loss = 0.5 * torch.maximum((values - returns).square(),
                                             (value_clipped - returns).square()).mean()
            # Paper-aligned: no policy-entropy term and no environment diversity reward.
            loss = actor_loss + critic_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg["policy_grad_clip"],
                                           error_if_nonfinite=True)
            self.optimizer.step()
            total_a += float(actor_loss.detach())
            total_v += float(critic_loss.detach())
        n = self.cfg["ppo_epochs"]
        return {"actor_loss": total_a / n, "critic_loss": total_v / n}

    @torch.no_grad()
    def collect_rollout(self, env: "SourceRecommendationEnvironment", weight: float
                        ) -> tuple[Rollout, list[tuple[str, str, float]]]:
        """Collect one source episode and encoder feedback using frozen features.

        Exploration is sampling from the policy. The raw +10/-10/+1/-1 reward
        is preserved in both the PPO trajectory and the encoder feedback buffer.
        No policy-entropy bonus or separate diversity reward is introduced.
        """
        self.policy.eval()
        snapshots = {key: [] for key in
                     ("states", "masks", "actions", "logprobs", "values", "rewards", "terminals")}
        feedback = []
        while not env.done:
            state, candidates, mask = env.observe()
            distribution = self.policy.distribution(state, candidates, mask, weight)
            action = distribution.sample().squeeze(0)
            value = self.policy.value(state)
            logprob = distribution.log_prob(action).squeeze(0)
            pid, reward, done = env.step(int(action.item()))
            for key, item in (("states", state), ("masks", mask), ("actions", action),
                              ("values", value), ("logprobs", logprob)):
                snapshots[key].append(item.detach().clone())
            snapshots["rewards"].append(reward)
            snapshots["terminals"].append(done)
            feedback.append((env.user, pid, reward))
        if not feedback:
            raise ValueError("Cannot update PPO with an empty episode")
        device = env.state.candidates.device
        rollout = Rollout(
            states=torch.stack(snapshots["states"]),
            candidates=env.state.candidates.detach().clone(),
            masks=torch.stack(snapshots["masks"]),
            actions=torch.stack(snapshots["actions"]),
            old_logprobs=torch.stack(snapshots["logprobs"]),
            old_values=torch.stack(snapshots["values"]),
            rewards=torch.tensor(snapshots["rewards"], dtype=torch.float32, device=device),
            terminals=torch.tensor(snapshots["terminals"], dtype=torch.bool, device=device),
            residual_weight=weight,
        )
        return rollout, feedback

    def state_dict(self) -> dict:
        """Training state; the inference artifact exports only the Actor weights."""
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
