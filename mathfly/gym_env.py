"""
A Gymnasium environment that wraps the math curriculum as an RL world the fly
network lives in.

Framing: the environment is the *world* (it presents a problem as a stream of
sensory input over time and grades the answer); the fly's connectome network is
the *policy/agent* that receives those observations, integrates them through its
recurrent dynamics, and emits an answer action.

Each episode:
  * reset() samples a problem (a op b) from the active task and builds the timed
    input drive. It returns the first timestep's sensory input as the obs.
  * step(action) advances one timestep. Observations are the per-timestep input
    current the fly "sees". The action taken on the FINAL (answer) timestep is
    the network's answer; reward is +1 if it matches, else 0. All earlier steps
    give reward 0. (Because observations don't depend on the action, this is a
    temporally-extended contextual bandit -- exactly the structure of a
    stimulus->decision task.)

This makes Math-Fly a standard `gymnasium.Env`, so you can plug in any RL
library, or use the built-in REINFORCE trainer in mathfly/rl_train.py.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:  # pragma: no cover
    _HAS_GYM = False
    gym = object  # type: ignore

from .envs import CURRICULUM_BY_NAME
from .io_encoding import IOEncoder


class MathFlyEnv(gym.Env if _HAS_GYM else object):
    """Gymnasium env presenting arithmetic problems as timed sensory input."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, encoder: IOEncoder, task_name: str = "add_1digit",
                 seed: int = 0, reward_wrong: float = 0.0):
        if not _HAS_GYM:
            raise ImportError("gymnasium not installed. pip install gymnasium")
        super().__init__()
        self.enc = encoder
        self.task = CURRICULUM_BY_NAME[task_name]
        self.reward_wrong = reward_wrong
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(encoder.n_neurons,), dtype=np.float32)
        self.action_space = spaces.Discrete(max(2, self.task.max_answer + 1))
        self._U = None
        self._t = 0
        self._answer = None
        self._tokens = None

    def set_task(self, task_name: str):
        self.task = CURRICULUM_BY_NAME[task_name]
        self.action_space = spaces.Discrete(max(2, self.task.max_answer + 1))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)              # seeds the gymnasium base RNG
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        tokens, answer = self.task.sample(self.rng)
        self._tokens = tokens
        self._answer = int(answer)
        self._U = self.enc.encode(tokens).astype(np.float32)   # (T, N)
        self._t = 0
        info = {"answer": self._answer, "tokens": "".join(tokens),
                "is_answer_step": self._is_answer_step()}
        return self._U[0].copy(), info

    def _is_answer_step(self) -> bool:
        return self._t == self._U.shape[0] - 1

    def step(self, action):
        T = self._U.shape[0]
        terminated = self._is_answer_step()
        reward = 0.0
        if terminated:
            reward = 1.0 if int(action) == self._answer else self.reward_wrong
        self._t = min(self._t + 1, T - 1)
        obs = self._U[self._t].copy() if not terminated else np.zeros(
            self._U.shape[1], dtype=np.float32)
        info = {"answer": self._answer, "is_answer_step": self._is_answer_step(),
                "correct": (terminated and int(action) == self._answer)}
        return obs, reward, terminated, False, info

    def full_input(self) -> np.ndarray:
        """The whole (T, N) input drive for the current problem -- handy for a
        recurrent policy that consumes the episode in one forward pass."""
        return self._U

    def render(self):
        return f"{''.join(self._tokens)} (answer={self._answer})"


def make_env(connectome_config: dict | None = None, task_name: str = "add_1digit",
             **enc_kwargs) -> MathFlyEnv:
    """Convenience builder: create a connectome-sized encoder and an env."""
    from .connectome import load_connectome
    conn = load_connectome(connectome_config or {"synthetic_n": 800})
    task = CURRICULUM_BY_NAME[task_name]
    enc = IOEncoder(conn.n, n_answers=task.max_answer + 1, **enc_kwargs)
    env = MathFlyEnv(enc, task_name=task_name)
    env._connectome = conn      # stash so a policy can reuse the same wiring
    return env
