"""Inference side of the DQN agent: this is what runs in the tournament.

Official games run on a single CPU thread with a 0.5 s budget per step, so the
model is kept small and torch is pinned to one thread. Feature extraction costs
a few milliseconds, the forward pass well under one.
"""

import os

import numpy as np
import torch

from .features import ACTIONS, state_to_features
from .model import QNet, greedy_action

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")

# Relative weights for random exploration. Moving is favoured over waiting and
# bombing: uniform sampling makes a fresh agent bomb itself constantly, which
# teaches it to avoid bombs altogether before it ever sees one pay off.
EXPLORE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 0.3, 0.6], dtype=np.float64)


def setup(self):
    """Called once before the first round, in both training and play mode."""
    torch.set_num_threads(1)
    self.device = torch.device(os.environ.get("DQN_DEVICE", "cpu"))
    self.q_net = QNet().to(self.device)
    # read before loading: train.py starts epsilon high only for a fresh net
    self.fresh_model = not os.path.isfile(MODEL_FILE)

    if not self.fresh_model:
        state = torch.load(MODEL_FILE, map_location=self.device)
        self.q_net.load_state_dict(state)
        self.logger.info(f"Loaded model from {MODEL_FILE}.")
    else:
        self.logger.warning(f"No model at {MODEL_FILE}; starting from scratch.")

    self.q_net.eval()
    self.cached_features = None
    self.rng = np.random.default_rng()


def q_values(self, vec):
    with torch.no_grad():
        v = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
        out = self.q_net(v.unsqueeze(0).to(self.device))
    return out.squeeze(0).cpu().numpy()


def act(self, game_state: dict) -> str:
    """Pick an action. Features are cached so train.py need not recompute them.

    ``mask`` gates both the exploration draw and the greedy argmax. It is legal
    *and* safe: the escape search has already removed every action it believes
    is fatal, so neither path can walk into a blast while an alternative exists.
    Only when nothing legal is safe does the mask fall back to legality alone.
    """
    vec, mask = state_to_features(game_state)
    # train.py reads this to build the transition for the step just taken
    self.cached_features = (game_state["round"], game_state["step"], vec, mask)

    epsilon = getattr(self, "epsilon", 0.0) if self.train else 0.0
    if epsilon > 0.0 and self.rng.random() < epsilon:
        weights = EXPLORE_WEIGHTS * mask
        if weights.sum() <= 0:
            weights = EXPLORE_WEIGHTS.copy()
        action = int(self.rng.choice(len(ACTIONS), p=weights / weights.sum()))
        self.logger.debug(f"Exploring: {ACTIONS[action]}")
    else:
        action = greedy_action(q_values(self, vec), mask)
        self.logger.debug(f"Greedy: {ACTIONS[action]}")

    self.last_action_index = action
    return ACTIONS[action]
