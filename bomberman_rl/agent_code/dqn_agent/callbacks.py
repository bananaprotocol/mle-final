"""
Action selection of the DQN agent.

The agent is a hybrid of three parts:
  1. the Q-network decides where to walk, when to drop a bomb, and whom to
     follow or to avoid;
  2. a hard safety layer removes each action after which no escape path
     exists. The safety layer decides alone only when no action is safe;
  3. a loop-breaker removes the actions that keep the agent on the same two
     tiles. See `break_loop` below.

The framework sets the working directory to this agent directory before it
calls a callback. All file names here are therefore relative.
"""

import os
import random
import time
from collections import deque

import numpy as np
import torch

from .features import (
    ACTIONS,
    BOMB,
    MOVES,
    WAIT,
    action_masks,
    least_bad_action,
    observation,
)
from .model import DQNNet, to_tensors

MODEL_FILE = "model.pt"
PROBE_STEPS = (50, 400)  # steps at which the agent logs the mean act time

# Loop-breaker. The agent looks at the tiles of the last LOOP_WINDOW steps. If
# its own tile occurs LOOP_REVISITS times or more in that window, the agent is
# in a loop.
LOOP_WINDOW = 8
LOOP_REVISITS = 3


def setup(self):
    """
    Prepare the network. The framework calls this one time for each agent.

    In a tournament game `self.train` is False. The agent then uses one CPU
    thread only, and it never explores.
    """
    if not self.train:
        # The thread limit must be set before the first forward pass.
        torch.set_num_threads(1)

    self.device = torch.device("cpu")
    self.model = DQNNet()

    # Defaults for the exploration. train.py replaces them in setup_training.
    self.eps = 0.0
    self.eps_start = 0.0
    self.eps_end = 0.0
    self.n_rounds = 1

    # Counters of the whole game. The act-time probe below uses them.
    self.round = -1
    self.act_time_sum = 0.0
    self.act_steps = 0
    reset_round(self)

    if load_into(self.model, MODEL_FILE, self.logger):
        self.logger.info(f"Loaded the weights from {MODEL_FILE}.")

    self.model.to(self.device)
    self.model.eval()


def load_weights(path):
    """
    Read the network weights from a file.

    The file is a state dictionary, or a checkpoint that holds one under the
    key "model". A state dictionary keeps the weights usable across torch
    versions.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"]
    return blob


def load_into(model, path, logger) -> bool:
    """
    Put the weights of `path` into `model`. Tell if this was successful.

    A file of an older design has other shapes: the global vector grew from 14
    to 16 values, and the head is now a dueling head. Such a file must not stop
    the agent. The network then keeps its random start, and the log says so.
    """
    if not os.path.isfile(path):
        logger.info(f"No {path} found. The network starts at random.")
        return False
    try:
        model.load_state_dict(load_weights(path))
        return True
    except (RuntimeError, KeyError) as error:
        logger.warning(f"{path} does not fit this network ({error}). "
                       "The network starts at random.")
        return False


def epsilon(self, game_round):
    """Give the exploration rate. It falls linearly over the rounds of a stage."""
    if self.n_rounds <= 1:
        return self.eps_end
    part = min(1.0, max(0.0, (game_round - 1) / (self.n_rounds - 1)))
    return self.eps_start + part * (self.eps_end - self.eps_start)


def reset_round(self):
    """Clear the counters of the last round. train.py writes them to the CSV."""
    self.n_gated = 0
    self.n_override = 0
    self.n_loop = 0
    self.n_wait = 0
    self.n_bomb = 0
    self.round_act_time = 0.0
    self.round_act_steps = 0
    self.recent_tiles = deque(maxlen=LOOP_WINDOW)


def act(self, game_state) -> str:
    """
    Choose one action. The tournament gives this function 0.5 s per step.

    The function never raises an error. WAIT is the answer if something goes
    wrong, because WAIT is always legal.
    """
    if game_state is None:
        return "WAIT"

    start = time.perf_counter()
    try:
        action = choose_action(self, game_state)
    except Exception:
        self.logger.exception("act failed. The agent waits.")
        action = "WAIT"

    elapsed = time.perf_counter() - start
    self.act_time_sum += elapsed
    self.act_steps += 1
    self.round_act_time += elapsed
    self.round_act_steps += 1
    if self.act_steps in PROBE_STEPS:
        # The first measurement also holds the one-time start of torch.
        mean_ms = 1000.0 * self.act_time_sum / self.act_steps
        self.logger.info(f"mean act time over {self.act_steps} steps: {mean_ms:.2f} ms")
    return action


def choose_action(self, game_state) -> str:
    """Choose one action, count it, and keep the tile for the loop-breaker."""
    game_round = int(game_state["round"])
    if game_round != self.round:
        self.round = game_round
        reset_round(self)
    if self.train:
        self.eps = epsilon(self, game_round)

    tile = (int(game_state["self"][3][0]), int(game_state["self"][3][1]))
    legal, safe = action_masks(game_state)
    if bool((legal & ~safe).any()):
        self.n_gated += 1

    action = select_action(self, game_state, tile, legal, safe)

    self.recent_tiles.append(tile)
    if action == WAIT:
        self.n_wait += 1
    elif action == BOMB:
        self.n_bomb += 1
    return ACTIONS[action]


def select_action(self, game_state, tile, legal, safe) -> int:
    """Apply the masks and query the network. Gives the index of the action."""
    # No action keeps an escape path open. The safety layer plays alone.
    if not safe.any():
        self.n_override += 1
        self.logger.debug("No escape exists. The safety layer decides.")
        return least_bad_action(game_state, legal)

    if self.train and random.random() < self.eps:
        return int(random.choice(np.flatnonzero(safe)))

    # Only the lines below need the observation. The branches above leave it
    # unbuilt, which is most of the steps while epsilon is high.
    planes, glob, _d_target, _my_danger = observation(game_state)
    with torch.no_grad():
        plane_batch, glob_batch = to_tensors(planes, glob, self.device)
        q_values = self.model(plane_batch, glob_batch)[0].cpu().numpy()

    mask, fired = break_loop(self, tile, safe)
    if fired:
        self.n_loop += 1
    return int(np.where(mask, q_values, -np.inf).argmax())


def break_loop(self, tile, safe):
    """
    Remove the actions that hold the agent on the same tiles.

    A greedy policy is a function of the state alone, and the state holds no
    history. A two-step loop is therefore stable: after UP the best action is
    DOWN, and after DOWN the best action is UP. Standing still is stable for
    the same reason. The agent leaves such a loop only if an action is removed.

    The rule fires when the agent occupied its own tile LOOP_REVISITS times or
    more in the last LOOP_WINDOW steps. It then removes WAIT and the action
    that moves back to the tile of the step before. It keeps the mask
    unchanged if nothing would remain.

    Gives (mask, fired).
    """
    history = self.recent_tiles
    if len(history) < LOOP_WINDOW or history.count(tile) < LOOP_REVISITS:
        return safe, False

    reduced = safe.copy()
    reduced[WAIT] = False
    previous = history[-1]
    for action, (dx, dy) in enumerate(MOVES):
        if (tile[0] + dx, tile[1] + dy) == previous:
            reduced[action] = False
    if not reduced.any():
        return safe, False
    self.logger.debug(f"Loop at {tile}. The loop-breaker removes WAIT and {previous}.")
    return reduced, True
