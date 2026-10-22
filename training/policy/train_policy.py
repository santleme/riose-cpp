#!/usr/bin/env python3
"""Synthetic Campo Vivo policy experiment: DQN versus transparent heuristic.

All dynamics and rewards here are authored game mechanics for experimentation.
They are not fitted to animal observations and are not biological claims.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    import gymnasium as gym
    from gymnasium import spaces
    BASE_ENV = gym.Env
    GYMNASIUM_BACKEND = True
except ImportError:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "gymnasium==1.2.1"], check=True)
        import gymnasium as gym
        from gymnasium import spaces
        BASE_ENV = gym.Env
        GYMNASIUM_BACKEND = True
    except Exception:
        # Keeps the exact reset/step contract when package installation is unavailable.
        class _Space:
            def __init__(self, *args, **kwargs): pass
        class spaces:
            Box = _Space
            Discrete = _Space
        class BASE_ENV: pass
        GYMNASIUM_BACKEND = False


OBSERVATION_FIELDS = ["hunger", "thirst", "fatigue", "heat_load", "grass", "water", "shade", "neighbor_presence", "social_stress"]
ACTION_NAMES = ["graze", "drink", "rest", "seek_shade", "follow_herd"]
HORIZON = 240
ENVIRONMENT_VERSION = 2


class CowForagingEnv(BASE_ENV):
    """Small Gymnasium-style environment for a single cow's resource decisions."""
    metadata = {"render_modes": []}

    def __init__(self, horizon: int = HORIZON):
        super().__init__() if GYMNASIUM_BACKEND else None
        self.horizon = horizon
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(OBSERVATION_FIELDS),), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.t = 0
        self.state = np.zeros(len(OBSERVATION_FIELDS), dtype=np.float32)
        self.hot_day = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if GYMNASIUM_BACKEND:
            super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.hot_day = float(self.rng.uniform(0.0, 0.008))
        self.state = np.asarray([
            self.rng.uniform(0.12, 0.4), self.rng.uniform(0.12, 0.4),
            self.rng.uniform(0.08, 0.32), self.rng.uniform(0.15, 0.50),
            self.rng.uniform(0.48, 0.92), self.rng.uniform(0.50, 0.95),
            self.rng.uniform(0.35, 0.95), self.rng.uniform(0.15, 0.95), self.rng.uniform(0.05, 0.42),
        ], dtype=np.float32)
        return self.state.copy(), {"synthetic": True, "seed": seed}

    @staticmethod
    def _potential(state: np.ndarray) -> float:
        hunger, thirst, fatigue, heat, _grass, _water, _shade, _neighbors, social = state
        return -(0.42 * hunger + 0.52 * thirst + 0.18 * fatigue + 0.34 * heat + 0.12 * social)

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < len(ACTION_NAMES):
            raise ValueError(f"Invalid action index: {action}")
        before = self.state.copy()
        hunger, thirst, fatigue, heat, grass, water, shade, neighbors, social = map(float, before)
        resource_cost = 0.0
        unavailable = False
        # Needs accrue each interval before the selected action takes effect.
        hunger = min(1.0, hunger + 0.004)
        thirst = min(1.0, thirst + 0.005)
        fatigue = min(1.0, fatigue + 0.005)
        heat = float(np.clip(heat + self.hot_day + self.rng.normal(0, 0.004), 0, 1))
        neighbors = float(np.clip(neighbors + self.rng.normal(0, 0.012), 0, 1))
        social = min(1.0, social + 0.008)

        if action == 0:  # graze
            if grass > 0.04:
                bite = min(hunger, 0.20 * grass)
                hunger = max(0.0, hunger - bite)
                thirst = min(1.0, thirst + 0.006)
                fatigue = min(1.0, fatigue + 0.018)
                grass = max(0.0, grass - 0.060)
                resource_cost = 0.012 * bite
            else:
                unavailable = True
        elif action == 1:  # drink
            if water > 0.035:
                sip = min(thirst, 0.34 * water)
                thirst = max(0.0, thirst - sip)
                fatigue = min(1.0, fatigue + 0.006)
                water = max(0.0, water - 0.065)
                resource_cost = 0.015 * sip
            else:
                unavailable = True
        elif action == 2:  # rest
            fatigue = max(0.0, fatigue - 0.23)
        elif action == 3:  # seek shade
            if shade > 0.03:
                heat = max(0.0, heat - 0.30 * shade)
                fatigue = max(0.0, fatigue - 0.015)
            else:
                unavailable = True
        else:  # follow herd / reduce social stress
            if neighbors > 0.08:
                social = max(0.0, social - 0.25 * neighbors)
                neighbors = min(1.0, neighbors + 0.18)
                fatigue = min(1.0, fatigue + 0.015)
            else:
                unavailable = True

        # Regrowth/refill and shade/weather vary smoothly, with bounded availability.
        grass = min(1.0, grass + 0.004)
        water = min(1.0, water + 0.003)
        shade = float(np.clip(shade + self.rng.normal(0, 0.006), 0.08, 1.0))
        after = np.asarray([hunger, thirst, fatigue, heat, grass, water, shade, neighbors, social], dtype=np.float32)
        # Potential-difference shaping plus an explicit ongoing welfare cost.
        # This prevents a policy from treating moderate chronic needs as free.
        after_potential = self._potential(after)
        reward = 0.35 * (after_potential - self._potential(before)) + 0.65 * after_potential - resource_cost
        if unavailable:
            reward -= 0.035
        if hunger > 0.88:
            reward -= 0.10
        if thirst > 0.88:
            reward -= 0.13
        if heat > 0.88:
            reward -= 0.09
        self.state = after
        self.t += 1
        terminated = False
        truncated = self.t >= self.horizon
        return after.copy(), float(reward), terminated, truncated, {
            "synthetic": True, "action": ACTION_NAMES[action], "unavailable_action": unavailable,
        }


def heuristic_action(obs: np.ndarray) -> int:
    """One-step transparent utility baseline using the same synthetic response rates."""
    hunger, thirst, fatigue, heat, grass, water, shade, neighbors, social = map(float, obs)
    utility = [0.42 * min(hunger + 0.004, 0.20 * grass) if grass > 0.04 else 0.0,
               0.52 * min(thirst + 0.005, 0.34 * water) if water > 0.035 else 0.0,
               0.18 * min(fatigue + 0.005, 0.23),
               0.34 * min(heat, 0.30 * shade) if shade > 0.03 else 0.0,
               0.12 * min(social, 0.25 * neighbors) if neighbors > 0.08 else 0.0]
    return int(np.argmax(utility))


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(OBSERVATION_FIELDS), 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(ACTION_NAMES)))

    def forward(self, x):
        return self.layers(x)


class Replay:
    def __init__(self, capacity: int, seed: int):
        self.data = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, transition): self.data.append(transition)
    def sample(self, batch_size): return self.rng.sample(self.data, batch_size)
    def __len__(self): return len(self.data)


def train_dqn(seed: int, steps: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    env = CowForagingEnv()
    online, target = QNetwork().to(device), QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=8e-4, weight_decay=2e-4)
    replay = Replay(50_000, seed)
    obs, _ = env.reset(seed=seed + 100)
    updates = 0
    for step in range(1, steps + 1):
        epsilon = max(0.04, 1.0 - 0.96 * step / max(1, steps))
        if random.random() < epsilon:
            action = random.randrange(len(ACTION_NAMES))
        else:
            with torch.no_grad():
                action = int(online(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay.add((obs, action, reward, next_obs, float(terminated)))
        obs = next_obs
        if truncated or terminated:
            obs, _ = env.reset(seed=seed + 100 + step)
        if len(replay) >= 512 and step % 4 == 0:
            batch = replay.sample(256)
            states = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
            actions = torch.as_tensor([b[1] for b in batch], dtype=torch.long, device=device)
            rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=device)
            next_states = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=device)
            dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=device)
            with torch.no_grad():
                target_q = rewards + 0.985 * (1.0 - dones) * target(next_states).max(1).values
            predicted_q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(predicted_q, target_q)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 5.0)
            optimizer.step()
            updates += 1
            if updates % 250 == 0:
                target.load_state_dict(online.state_dict())
    return online.eval(), {"seed": seed, "steps": steps, "updates": updates,
                           "epsilon_final": max(0.04, 1.0 - 0.96 * steps / max(1, steps))}


def run_episode(policy, seed: int):
    env = CowForagingEnv()
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    for _ in range(HORIZON):
        action = policy(obs)
        actions[action] += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return {"return": float(total_reward), "final_hunger": float(obs[0]), "final_thirst": float(obs[1]),
            "final_fatigue": float(obs[2]), "final_heat": float(obs[3]),
            "final_neighbor_presence": float(obs[7]), "final_social_stress": float(obs[8]),
            "final_need_mean": float(np.mean(obs[[0, 1, 2, 3, 8]])),
            "grass_left": float(obs[4]), "water_left": float(obs[5]), "action_counts": actions.tolist()}


def summarize(episodes: list[dict]) -> dict:
    keys = ["return", "final_hunger", "final_thirst", "final_fatigue", "final_heat",
            "final_neighbor_presence", "final_social_stress", "final_need_mean", "grass_left", "water_left"]
    return {key: {"mean": float(np.mean([e[key] for e in episodes])),
                  "std": float(np.std([e[key] for e in episodes], ddof=1))} for key in keys}


def export_weights(model: QNetwork, path: Path):
    linears = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    payload = {"schema_version": 1, "model_type": "mlp_q_argmax", "input_fields": OBSERVATION_FIELDS,
               "actions": ACTION_NAMES, "hidden_activation": "relu",
               "layers": [{"weights_out_in": layer.weight.detach().cpu().tolist(),
                           "bias": layer.bias.detach().cpu().tolist()} for layer in linears],
               "training_scope": "synthetic_game_dynamics_only"}
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--eval-episodes", type=int, default=20)
    default_out = Path("/kaggle/working/training/policy/outputs") if Path("/kaggle/working").is_dir() else Path("training/policy/outputs")
    parser.add_argument("--output-dir", type=Path, default=default_out)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training = train_dqn(args.seed, args.steps, device)
    eval_seeds = [1_000_000 + i for i in range(args.eval_episodes)]
    learned = lambda obs: int(model(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item())
    reference = lambda obs: heuristic_action(obs)
    learned_episodes = [run_episode(learned, seed) for seed in eval_seeds]
    heuristic_episodes = [run_episode(reference, seed) for seed in eval_seeds]
    learned_summary, heuristic_summary = summarize(learned_episodes), summarize(heuristic_episodes)
    paired_return_delta = [a["return"] - b["return"] for a, b in zip(learned_episodes, heuristic_episodes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "seed": args.seed, "actions": ACTION_NAMES,
                "observation_fields": OBSERVATION_FIELDS}, args.output_dir / "policy.pt")
    export_weights(model, args.output_dir / "policy_weights.json")
    report = {
        "experiment": "synthetic_cow_resource_policy_dqn",
        "claim_scope": "The policy learns only the authored synthetic environment. It does not infer causal real-cow behavior.",
        "environment": {"api": "Gymnasium reset/step", "version": ENVIRONMENT_VERSION,
            "gymnasium_installed": GYMNASIUM_BACKEND,
            "observation_fields": OBSERVATION_FIELDS, "actions": ACTION_NAMES, "horizon_steps": HORIZON,
            "reward": "0.35 * (potential(after)-potential(before)) + 0.65 * potential(after) - resource depletion cost - unavailable-action penalty - high-need penalties; see train_policy.py"},
        "training": {**training, "device": str(device), "cuda_available": torch.cuda.is_available(),
                     "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "eval_seeds_disjoint": True, "evaluation_seeds": eval_seeds},
        "evaluation": {"dqn": learned_summary, "heuristic": heuristic_summary,
                       "paired_return_delta_mean": float(np.mean(paired_return_delta)),
                       "paired_return_delta_std": float(np.std(paired_return_delta, ddof=1)),
                       "dqn_episodes": learned_episodes, "heuristic_episodes": heuristic_episodes},
        "artifact": "policy_weights.json",
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"device": str(device), "gpu_name": report["training"]["gpu_name"],
        "gymnasium_installed": GYMNASIUM_BACKEND, "episodes": args.eval_episodes,
        "dqn_return": learned_summary["return"], "heuristic_return": heuristic_summary["return"],
        "paired_return_delta_mean": report["evaluation"]["paired_return_delta_mean"],
        "dqn_final_need_mean": learned_summary["final_need_mean"],
        "heuristic_final_need_mean": heuristic_summary["final_need_mean"],
        "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
