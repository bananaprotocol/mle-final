"""
Training code of the DQN agent (Double DQN with a replay buffer).

The reward has three parts:
  * the true reward of the game: coins, kills, deaths, survival, step cost;
  * a potential-based shaping term, F = Phi(s') - Phi(s);
  * a small auxiliary reward for crates and for found coins.

The CSV log keeps the three parts in separate columns. Watch them: if the
shaping term is more than about 30 % of the total, the agent learns the
shaping and not the game.

The stage of the curriculum comes from `train_config.json` in this directory.
One file edit sets the scenario name, the number of rounds, the exploration
rate, the auxiliary scale and the checkpoint of the previous stage.

Two details of the framework need care:
  * `game_events_occurred` is not called for the step in which the agent
    dies. `end_of_round` always delivers that step.
  * If the agent survives the last step, the framework delivers that step two
    times. The second delivery holds the full event list of that step and
    SURVIVED_ROUND. The code below adds the round bonus to the buffer slot of
    the first delivery, so that the step counts one time only.

The first version of this file used Phi <= 0 and F = gamma * Phi(s') - Phi(s).
That pays (1 - gamma) * |Phi| for each step in which nothing happens, which
was about 5 times the step cost. The agent then learned to stand still.
"""

import atexit
import csv
import json
import os
import random
from collections import deque
from typing import List

import events as e
import numpy as np
import torch
import torch.nn.functional as functional

from .callbacks import MODEL_FILE, load_into
from .features import (
    ACTIONS,
    BOARD,
    CAP,
    N_ACTIONS,
    N_GLOBAL,
    N_PLANES,
    action_masks,
    observation,
)
from .model import DQNNet, to_tensors

CONFIG_FILE = "train_config.json"
LOG_DIR = "logs"

# Hyperparameters (section 12 of the design document).
GAMMA = 0.97
ALPHA_PBRS = 0.1     # weight of the distance term of the potential
BETA_PBRS = 0.5      # weight of the danger term of the potential
STEP_COST = 0.01
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
WARMUP = 5_000       # number of transitions before the first gradient step
LEARNING_RATE = 2e-4
GRAD_CLIP = 10.0
TARGET_SYNC = 500    # gradient steps between two copies to the target network
CKPT_EVERY = 50      # rounds between two checkpoints
PROBE_SIZE = 512     # number of fixed states for the q_mean column
BEST_WINDOW = 50     # rounds in the mean that selects the best checkpoint

# The shaping term uses the discount 1, not GAMMA. With GAMMA the term pays
# (1 - GAMMA) * |Phi| for each step in which the state does not change. That
# is a reward for idle time. A shift of Phi only moves the drift to another
# part of the board; the discount 1 removes it everywhere. The price is that
# the shaping is no longer exactly policy-invariant. The remaining bias is
# bounded by (1 - GAMMA) * max(Phi) and it points towards the target, which
# is the wanted direction.
GAMMA_SHAPING = 1.0

# The potential is shifted to be non-negative: 0 <= Phi <= PHI_SHIFT. A shift
# by a constant is still a potential, so the argument of Ng et al. (1999)
# holds. The shift also removes a bonus: with Phi <= 0 the terminal step of a
# death received -Phi(s) >= 0, up to +2.9, which cancelled more than half of
# the death penalty.
PHI_SHIFT = ALPHA_PBRS * CAP + BETA_PBRS

# The true reward of the game. The report lists this table.
REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.GOT_KILLED: -5.0,
    e.KILLED_SELF: -5.0,   # a suicide gives GOT_KILLED and KILLED_SELF: -10.0
    e.SURVIVED_ROUND: 2.0,
}

# The auxiliary reward. It is NOT policy-invariant, so it stays small and the
# stage config can scale it down.
#
# Why it is necessary: in `classic` there are about 123 crates and 9 coins, so
# one crate holds a coin with the probability 0.073. A bomb-and-return cycle
# takes about 9 steps, which costs 0.09 of step cost. Without a payoff for the
# crate itself the expected value of a bomb is zero or negative, and the agent
# correctly learns not to bomb. In `loot-crate` the probability is 0.41, which
# is why stage 2 worked and stage 3 did not.
#
# COIN_FOUND fires 4 steps after the bomb drop, so the credit assignment is
# short, and it is a strict precursor of the real +1 of COIN_COLLECTED.
AUX_REWARDS = {
    e.CRATE_DESTROYED: 0.05,
    e.COIN_FOUND: 0.5,
}

DEFAULT_CONFIG = {
    "stage": "stage1",
    "n_rounds": 300,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "warm_start": None,
    "seed": None,
    "aux_scale": 1.0,   # 0.0 removes the auxiliary reward (ablation A10)
}

CSV_COLUMNS = [
    "round", "stage", "reward", "true", "shaping", "aux", "score", "coins",
    "kills", "suicides", "survived", "steps", "loss_mean", "eps", "q_mean",
    "n_override", "n_gated", "n_loop", "n_wait", "n_bomb", "act_ms",
]


# ---------------------------------------------------------------------------
# replay buffer
# ---------------------------------------------------------------------------

class Replay:
    """
    A ring buffer with a fixed capacity.

    The board planes use the type uint8. This keeps 100 000 transitions in
    about 0.5 GB of memory. The sampler makes float values from them.
    """

    def __init__(self, capacity, shape):
        self.capacity = capacity
        self.size = 0
        self.head = 0
        self.planes = np.zeros((capacity, *shape), dtype=np.uint8)
        self.glob = np.zeros((capacity, N_GLOBAL), dtype=np.float32)
        self.next_planes = np.zeros((capacity, *shape), dtype=np.uint8)
        self.next_glob = np.zeros((capacity, N_GLOBAL), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.next_mask = np.zeros((capacity, N_ACTIONS), dtype=bool)

    def push(self, obs, action, reward, next_obs, next_mask, done):
        """Store one transition and give back its slot index."""
        i = self.head
        self.planes[i], self.glob[i] = obs
        self.next_planes[i], self.next_glob[i] = next_obs
        self.action[i] = action
        self.reward[i] = reward
        self.done[i] = done
        self.next_mask[i] = next_mask
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def sample(self, batch_size):
        return np.random.randint(0, self.size, size=batch_size)

    def tensors(self, index, device):
        """Give the tensors of the selected transitions, in the order learn() uses."""
        planes, glob = to_tensors(self.planes[index], self.glob[index], device)
        next_planes, next_glob = to_tensors(
            self.next_planes[index], self.next_glob[index], device)
        rest = (self.action, self.reward, self.done, self.next_mask)
        return (planes, glob, next_planes, next_glob,
                *(torch.from_numpy(array[index]).to(device) for array in rest))


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def setup_training(self):
    """
    Prepare the buffer, the target network and the optimizer.

    The framework calls this function after `setup` in callbacks.py.
    """
    config = dict(DEFAULT_CONFIG)
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE) as handle:
            config.update(json.load(handle))
    self.config = config
    self.stage = str(config["stage"])
    self.n_rounds = int(config["n_rounds"])
    self.eps_start = float(config["eps_start"])
    self.eps_end = float(config["eps_end"])
    self.aux_scale = float(config["aux_scale"])

    if config.get("seed") is not None:
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.logger.info(f"seed = {seed}")

    # Weights of the previous stage. They win over model.pt. Select this file
    # by the score of the checkpoint (ckpt_<stage>_best.pt), not by its round
    # number: stages 3a and 3b of the first run became worse throughout the
    # stage, and the last checkpoint carried that loss into the next stage.
    warm_start = config.get("warm_start")
    if warm_start and load_into(self.model, warm_start, self.logger):
        self.logger.info(f"Warm start from {warm_start}.")

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.model.train()
    self.target = DQNNet(**self.model.options).to(self.device)
    self.target.load_state_dict(self.model.state_dict())
    self.target.eval()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    self.buffer = Replay(BUFFER_SIZE, (N_PLANES, BOARD, BOARD))
    self.grad_steps = 0
    self.probe = None

    # Selection of the best checkpoint of the stage.
    self.true_history = deque(maxlen=BEST_WINDOW)
    self.best_true = None

    # Bookkeeping of the transition that game_events_occurred stored last.
    self.last_slot = None
    self.last_key = None
    start_new_round(self)

    # The framework sets the working directory to this agent directory before
    # each callback, but not for the atexit handler. Keep the directory here.
    self.agent_dir = os.path.abspath(os.getcwd())
    self.csv_path = os.path.join(self.agent_dir, LOG_DIR, f"train_{self.stage}.csv")
    os.makedirs(os.path.join(self.agent_dir, LOG_DIR), exist_ok=True)
    open_csv(self)

    atexit.register(save_checkpoint, self)
    self.logger.info(
        f"stage={self.stage} rounds={self.n_rounds} device={self.device} "
        f"eps {self.eps_start} -> {self.eps_end}"
    )


def open_csv(self):
    """
    Make the CSV file of the stage ready for new lines.

    A file of an older run has an older header. Mixing the two makes the
    columns meaningless, so the old file is kept under a new name.
    """
    if os.path.isfile(self.csv_path):
        with open(self.csv_path, newline="") as handle:
            header = next(csv.reader(handle), [])
        if header == CSV_COLUMNS:
            return
        backup = f"{self.csv_path}.old"
        os.replace(self.csv_path, backup)
        self.logger.warning(f"The CSV file had an old header. It is now {backup}.")
    with open(self.csv_path, "w", newline="") as handle:
        csv.writer(handle).writerow(CSV_COLUMNS)


def start_new_round(self):
    """
    Clear the sums of the last round.

    The counters of callbacks.py (n_gated, n_override, n_loop, n_wait,
    n_bomb, act time) are not cleared here. `choose_action` clears them at the
    first step of the new round, which is after `write_csv_row` has read the
    round that ended.
    """
    self.round_reward = 0.0
    self.round_true = 0.0
    self.round_shaping = 0.0
    self.round_aux = 0.0
    self.round_losses = []
    self.round_coins = 0
    self.round_kills = 0


def count_events(self, events: List[str]):
    """Add the events of one step to the counters of the round."""
    self.round_coins += events.count(e.COIN_COLLECTED)
    self.round_kills += events.count(e.KILLED_OPPONENT)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------

def true_reward(events: List[str]) -> float:
    """Give the true reward of one step: the game objective and the step cost."""
    return sum(REWARDS.get(event, 0.0) for event in events) - STEP_COST


def aux_reward(self, events: List[str]) -> float:
    """Give the auxiliary reward of one step: crates and found coins."""
    if self.aux_scale == 0.0:
        return 0.0
    return self.aux_scale * sum(AUX_REWARDS.get(event, 0.0) for event in events)


def potential(d_target, my_danger) -> float:
    """
    Give Phi(s) of the reward shaping. The result is never negative.

    The first term pulls the agent to the nearest coin, or to a tile next to
    a crate when no coin is visible. The second term punishes a tile that a
    bomb hits in this step or in the next step. PHI_SHIFT makes the result
    non-negative; see the note at the top of this file.
    """
    danger = 1.0 if my_danger <= 1 else 0.0
    return PHI_SHIFT - ALPHA_PBRS * d_target - BETA_PBRS * danger


def shaping(old_target, old_danger, new_target, new_danger) -> float:
    """Give F for a step between two non-terminal states."""
    return (GAMMA_SHAPING * potential(new_target, new_danger)
            - potential(old_target, old_danger))


def add_reward(self, true, shape, aux) -> float:
    """Add the three parts to the sums of the round. Gives the total."""
    self.round_true += true
    self.round_shaping += shape
    self.round_aux += aux
    total = true + shape + aux
    self.round_reward += total
    return total


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """Store one transition and make one gradient step."""
    if old_game_state is None or self_action not in ACTIONS:
        return

    old_planes, old_glob, old_target, old_danger = observation(old_game_state)
    new_planes, new_glob, new_target, new_danger = observation(new_game_state)

    reward = add_reward(
        self,
        true_reward(events),
        shaping(old_target, old_danger, new_target, new_danger),
        aux_reward(self, events),
    )

    # The bootstrap mask of the new state. The safe mask can be empty: no
    # action of the new state keeps an escape path open. The agent is then
    # still alive and still plays, so the transition is NOT terminal. The
    # legal mask is the bootstrap mask in that case; WAIT is always legal, so
    # the mask always holds at least one action.
    legal, safe = action_masks(new_game_state)
    next_mask = safe if safe.any() else legal

    self.last_slot = self.buffer.push(
        (old_planes, old_glob), ACTIONS.index(self_action), reward,
        (new_planes, new_glob), next_mask, False,
    )
    self.last_key = (int(old_game_state["round"]), int(old_game_state["step"]))
    count_events(self, events)
    learn(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Close the round. There are two cases.

    The agent survived the last step. The round stopped at the step limit, or
    nothing was left to do. This is a truncation, not a terminal state: the
    game itself would continue. The framework already delivered this step
    through `game_events_occurred`, so the buffer slot is correct and keeps
    `done = False`; the value of the last state stays bootstrapped. Only the
    SURVIVED_ROUND bonus is missing, because the first delivery did not hold
    that event yet.

    The agent died in the last step. The state is terminal. This is the only
    delivery of that step, so a new transition is stored, with `done = True`
    and Phi(s') = 0.
    """
    if last_action in ACTIONS:
        key = (int(last_game_state["round"]), int(last_game_state["step"]))
        if self.last_key == key and self.last_slot is not None:
            bonus = REWARDS[e.SURVIVED_ROUND] if e.SURVIVED_ROUND in events else 0.0
            if bonus:
                self.buffer.reward[self.last_slot] += bonus
                add_reward(self, bonus, 0.0, 0.0)
        else:
            planes, glob, d_target, my_danger = observation(last_game_state)
            reward = add_reward(
                self,
                true_reward(events),
                -potential(d_target, my_danger),   # Phi(terminal) = 0
                aux_reward(self, events),
            )
            # The next state is never read: done is True, so the bootstrap
            # term is zero. The state itself stands in for it.
            self.buffer.push((planes, glob), ACTIONS.index(last_action), reward,
                             (planes, glob), np.zeros(N_ACTIONS, dtype=bool), True)
            count_events(self, events)
        learn(self)

    self.last_slot = None
    self.last_key = None
    write_csv_row(self, last_game_state, events)
    track_best(self)
    if int(last_game_state["round"]) % CKPT_EVERY == 0:
        save_checkpoint(self)
    start_new_round(self)


# ---------------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------------

def learn(self):
    """Make one Double-DQN gradient step on a batch from the buffer."""
    if self.buffer.size < WARMUP:
        return
    if self.probe is None:
        make_probe(self)

    index = self.buffer.sample(BATCH_SIZE)
    (planes, glob, next_planes, next_glob,
     action, reward, done, next_mask) = self.buffer.tensors(index, self.device)

    with torch.no_grad():
        # Double DQN: the online network picks the action, the target
        # network gives its value. Only safe actions take part.
        online_next = self.model(next_planes, next_glob)
        online_next = online_next.masked_fill(~next_mask, -1e9)
        best = online_next.argmax(dim=1, keepdim=True)
        target_next = self.target(next_planes, next_glob).gather(1, best).squeeze(1)
        target = reward + GAMMA * target_next * (~done)

    q_value = self.model(planes, glob).gather(1, action.unsqueeze(1)).squeeze(1)
    loss = functional.smooth_l1_loss(q_value, target)

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
    self.optimizer.step()

    self.grad_steps += 1
    self.round_losses.append(float(loss.item()))
    if self.grad_steps % TARGET_SYNC == 0:
        self.target.load_state_dict(self.model.state_dict())


def make_probe(self):
    """
    Keep a fixed set of states. The q_mean column uses it every round.

    The tensors are a copy, not a list of buffer slots: a long stage writes
    more than BUFFER_SIZE transitions, and the ring would then silently
    replace the probe states.
    """
    index = self.buffer.sample(min(PROBE_SIZE, self.buffer.size))
    self.probe = to_tensors(
        self.buffer.planes[index], self.buffer.glob[index], self.device)


def probe_q_mean(self) -> float:
    """
    Give the mean of the largest Q-value over the fixed probe states.

    A value that rises while the score stays flat is a sign of an
    overestimation of the Q-values.
    """
    if self.probe is None:
        return 0.0
    with torch.no_grad():
        return float(self.model(*self.probe).max(dim=1).values.mean().item())


# ---------------------------------------------------------------------------
# logging and checkpoints
# ---------------------------------------------------------------------------

def write_csv_row(self, last_game_state, events):
    """Append one line per round to the CSV file for the training curves."""
    act_ms = 1000.0 * self.round_act_time / max(self.round_act_steps, 1)
    losses = self.round_losses
    row = [
        int(last_game_state["round"]),
        self.stage,
        round(self.round_reward, 3),
        round(self.round_true, 3),
        round(self.round_shaping, 3),
        round(self.round_aux, 3),
        int(last_game_state["self"][1]),
        self.round_coins,
        self.round_kills,
        1 if e.KILLED_SELF in events else 0,
        1 if e.SURVIVED_ROUND in events else 0,
        int(last_game_state["step"]),
        round(float(np.mean(losses)), 5) if losses else "",
        round(self.eps, 4),
        round(probe_q_mean(self), 4),
        self.n_override,
        self.n_gated,
        self.n_loop,
        self.n_wait,
        self.n_bomb,
        round(act_ms, 3),
    ]
    with open(self.csv_path, "a", newline="") as handle:
        csv.writer(handle).writerow(row)


def checkpoint_blob(self):
    """
    Give the dictionary that one checkpoint file holds.

    The file holds state dictionaries only. A pickled module breaks when the
    torch version changes, and the tournament image has its own torch.
    """
    return {
        "model": {key: value.cpu() for key, value in self.model.state_dict().items()},
        "target": {key: value.cpu() for key, value in self.target.state_dict().items()},
        "optimizer": self.optimizer.state_dict(),
        "options": self.model.options,
        "grad_steps": self.grad_steps,
        "round": self.round,
        "eps": self.eps,
        "stage": self.stage,
    }


def track_best(self):
    """
    Keep the checkpoint with the best true return of the stage.

    The last checkpoint of a stage is often not the best one. In the first
    run, stages 3a and 3b became worse from round to round, and each next
    stage started from those weights. The mean runs over BEST_WINDOW rounds,
    because one round alone is too noisy.

    Only the true return counts here. A mean over the shaped return would
    select the checkpoint that collects the most shaping.
    """
    if not hasattr(self, "agent_dir"):
        return
    self.true_history.append(self.round_true)
    if len(self.true_history) < BEST_WINDOW:
        return
    mean_true = sum(self.true_history) / len(self.true_history)
    if self.best_true is not None and mean_true <= self.best_true:
        return
    self.best_true = mean_true
    torch.save(checkpoint_blob(self),
               os.path.join(self.agent_dir, f"ckpt_{self.stage}_best.pt"))
    self.logger.info(f"best mean true return {mean_true:.3f} at round {self.round}")


def save_checkpoint(self):
    """Write a numbered checkpoint, the newest checkpoint and the weights."""
    if not hasattr(self, "agent_dir"):  # setup_training did not finish
        return
    blob = checkpoint_blob(self)
    torch.save(blob, os.path.join(self.agent_dir, f"ckpt_{self.stage}_r{self.round}.pt"))
    torch.save(blob, os.path.join(self.agent_dir, "ckpt_latest.pt"))
    torch.save(blob["model"], os.path.join(self.agent_dir, MODEL_FILE))
    self.logger.info(f"checkpoint at round {self.round}, {self.grad_steps} gradient steps")
