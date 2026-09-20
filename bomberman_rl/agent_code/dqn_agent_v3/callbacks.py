import atexit
import os
import pickle

import numpy as np
import torch

from .features import ACTIONS, state_to_features
from .model import QNet, greedy_action

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")
# a path here writes every decision to a pickle for later analysis
TRACE_FILE = os.environ.get("DQN_TRACE")

# moves beat WAIT and BOMB, or a fresh agent only ever bombs itself
EXPLORE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 0.3, 0.6], dtype=np.float64)


def setup(self):
    torch.set_num_threads(1)
    self.device = torch.device(os.environ.get("DQN_DEVICE", "cpu"))
    self.q_net = QNet().to(self.device)

    if os.path.isfile(MODEL_FILE):
        state = torch.load(MODEL_FILE, map_location=self.device)
        self.q_net.load_state_dict(state)
        self.logger.info(f"Loaded model from {MODEL_FILE}.")
    else:
        self.logger.warning(f"No model at {MODEL_FILE}; starting from scratch.")

    self.q_net.eval()
    self.cached_features = None
    self.rng = np.random.default_rng()
    _setup_trace(self)


def _setup_trace(self):
    self.trace = None
    if not TRACE_FILE:
        return
    self.trace = []
    self.trace_round = None
    # play mode has no "game over" callback, so the last round would be lost
    atexit.register(_dump_trace, self)
    self.logger.info(f"Tracing decisions to {TRACE_FILE}.")


def _dump_trace(self):
    with open(TRACE_FILE, "wb") as fh:
        pickle.dump(self.trace, fh)


def _record(self, game_state, action):
    # the framework builds fresh arrays every step, so we can keep the state
    if self.trace_round is not None and game_state["round"] != self.trace_round:
        _dump_trace(self)
    self.trace_round = game_state["round"]
    self.trace.append((game_state["round"], game_state["step"], game_state, action))


def q_values(self, vec, crop):
    with torch.no_grad():
        v = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
        c = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))
        out = self.q_net(v.unsqueeze(0).to(self.device), c.unsqueeze(0).to(self.device))
    return out.squeeze(0).cpu().numpy()


def act(self, game_state: dict) -> str:
    # mask gates the exploration draw and the argmax alike
    vec, crop, mask = state_to_features(game_state)
    # train.py reuses this to build the transition for the step just taken
    self.cached_features = (game_state["round"], game_state["step"], vec, crop, mask)

    epsilon = getattr(self, "epsilon", 0.0) if self.train else 0.0
    if epsilon > 0.0 and self.rng.random() < epsilon:
        weights = EXPLORE_WEIGHTS * mask
        if weights.sum() <= 0:
            weights = EXPLORE_WEIGHTS.copy()
        action = int(self.rng.choice(len(ACTIONS), p=weights / weights.sum()))
        self.logger.debug(f"Exploring: {ACTIONS[action]}")
    else:
        action = greedy_action(q_values(self, vec, crop), mask)
        self.logger.debug(f"Greedy: {ACTIONS[action]}")

    self.last_action_index = action
    if self.trace is not None:
        _record(self, game_state, ACTIONS[action])
    return ACTIONS[action]
