"""
Reinforcement-learning trainer: train the fly's connectome network as an agent
that gets REWARDED for correct answers, using REINFORCE (policy gradients).

Unlike the supervised trainers (which are told the right answer), here the
network only receives a scalar reward (+1 correct / 0 wrong) after it commits to
an answer -- a more "agent living in a world" framing. The policy is the
connectome-constrained recurrent network from model.py: its wiring and Dale's
law are fixed, and learning tunes the per-edge gains + I/O + readout to maximise
expected reward.

Because the environment's observations don't depend on the agent's action (the
problem is shown regardless), each episode is a temporally-extended contextual
bandit, so we can roll out a batch of episodes, score them, and apply a single
REINFORCE update with a moving-average baseline for variance reduction.

Needs torch + gymnasium:  pip install torch gymnasium
"""

from __future__ import annotations

import os
import time

import numpy as np

from .connectome import load_connectome
from .envs import CURRICULUM_BY_NAME
from .gym_env import MathFlyEnv
from .io_encoding import IOEncoder
from .model import build_torch_rnn


def train_rl(config: dict):
    import torch

    t0 = time.time()
    conn = load_connectome(config)
    task_name = config.get("task", "add_1digit")
    task = CURRICULUM_BY_NAME[task_name]
    print(f"[connectome] source={conn.source} N={conn.n} synapses={conn.W.nnz}")
    print(f"[rl] task={task_name} (reward +1 correct, chance ~{1/(task.max_answer+1):.1%})")

    enc = IOEncoder(conn.n, n_input=config.get("n_input", 64),
                    n_readout=config.get("n_readout", 256),
                    n_answers=task.max_answer + 1,
                    steps_per_token=config.get("steps_per_token", 4),
                    seed=config.get("seed", 0))
    env = MathFlyEnv(enc, task_name=task_name, seed=config.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_torch_rnn(conn, enc, alpha=config.get("alpha", 0.3)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=config.get("lr", 3e-3))

    n_iters = config.get("n_iters", 300)
    batch = config.get("batch", 32)
    entropy_coef = config.get("entropy_coef", 0.01)
    baseline = 0.0
    beta = 0.9                                   # baseline EMA
    rng = np.random.default_rng(config.get("seed", 0))

    for it in range(n_iters):
        # -- roll out a batch of episodes through the gym env --
        Us, answers = [], []
        for _ in range(batch):
            env.reset(seed=int(rng.integers(0, 2**31)))
            Us.append(env.full_input())
            answers.append(env._answer)
        T = max(u.shape[0] for u in Us)
        U = np.zeros((batch, T, conn.n), dtype=np.float32)
        for i, u in enumerate(Us):
            U[i, -u.shape[0]:] = u               # right-align in time
        U = torch.tensor(U, device=device)
        answers = torch.tensor(answers, device=device)

        # -- policy forward, sample actions --
        logits = net(U)                          # (B, n_answers)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        reward = (actions == answers).float()    # +1 correct / 0 wrong

        # -- REINFORCE with EMA baseline + entropy bonus --
        advantage = reward - baseline
        logp = dist.log_prob(actions)
        loss = -(logp * advantage.detach()).mean() - entropy_coef * dist.entropy().mean()

        opt.zero_grad(); loss.backward(); opt.step()
        baseline = beta * baseline + (1 - beta) * reward.mean().item()

        if it % 25 == 0 or it == n_iters - 1:
            greedy = (logits.argmax(1) == answers).float().mean().item()
            print(f"[rl {it:4d}] reward={reward.mean().item():.2f} "
                  f"greedy_acc={greedy:.1%} baseline={baseline:.2f} loss={loss.item():.3f}")

    out_dir = config.get("out_dir", "runs/rl")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "policy.pt"))
    print(f"[done] {time.time()-t0:.1f}s  saved -> {out_dir}/policy.pt")
    return net
