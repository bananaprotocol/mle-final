"""Training side of the DQN agent: Double DQN with symmetry augmentation.

Loaded only with ``--train``. Everything lives in memory, so one process is one
run over one scenario; only the weights carry over, through ``model.pt``.
``best.pt`` keeps the best rolling-mean-score network.
"""

import atexit
import csv
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

import events as e
import settings as s

from .callbacks import MODEL_FILE
from .features import (ACTIONS, IDX_CRATES_IN_BLAST, IDX_COIN_DIST, IDX_CRATE_DIST,
                       IDX_ENEMY_DIST, IDX_ENEMY_IN_BLAST, VEC_SIZE, action_mask,
                       state_to_features)
from .model import TRANSFORMS, QNet, transform_batch

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(AGENT_DIR, "train_log.csv")
BEST_FILE = os.path.join(AGENT_DIR, "best.pt")

# --- hyper parameters ------------------------------------------------------
BUFFER_SIZE, BATCH_SIZE = 200_000, 128
GAMMA, LEARNING_RATE = 0.95, 2.5e-4
TAU, LEARN_START = 0.005, 2_000  # soft update rate; transitions before update 1
GRAD_CLIP, UPDATES_PER_ROUND_END = 10.0, 20
SAVE_EVERY, SCORE_WINDOW = 25, 100  # rounds per model save / in the best-score window
# epsilon: linear decay, starting high only when there are no weights on disk
EPS_START_FRESH, EPS_START_WARM, EPS_END, EPS_DECAY_ROUNDS = 1.0, 0.3, 0.05, 400

# --- reward shaping --------------------------------------------------------
# Death dominates on purpose: surviving bombs gates all later play, so it must
# not be drowned out by coin rewards.
GAME_REWARDS = {
    e.COIN_COLLECTED: 10.0, e.KILLED_OPPONENT: 50.0, e.KILLED_SELF: -50.0,
    e.GOT_KILLED: -30.0, e.CRATE_DESTROYED: 2.0, e.COIN_FOUND: 1.0,
    e.INVALID_ACTION: -2.0, e.WAITED: -0.3, e.SURVIVED_ROUND: 5.0,
}
MOVED_TOWARDS_TARGET, MOVED_AWAY_FROM_TARGET = "MOVED_TOWARDS_TARGET", "MOVED_AWAY_FROM_TARGET"
USEFUL_BOMB, USELESS_BOMB = "USEFUL_BOMB", "USELESS_BOMB"
CUSTOM_REWARDS = {MOVED_TOWARDS_TARGET: 1.0, MOVED_AWAY_FROM_TARGET: -1.0,
                  USELESS_BOMB: -3.0}
CRATE_BOMB_BONUS = 2.0  # per crate in the blast, capped at 4 crates
ENEMY_BOMB_BONUS = 10.0


class ReplayBuffer:
    """Flat in-memory ring buffer: 200k x 84 floats is about 67 MB."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.vecs = np.zeros((capacity, VEC_SIZE), dtype=np.float32)
        self.next_vecs = np.zeros_like(self.vecs)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards, self.dones = np.zeros(capacity, np.float32), np.zeros(capacity, np.float32)
        self.pos = self.size = 0

    def __len__(self):
        return self.size

    def push(self, vec, action, reward, next_vec, done):
        i = self.pos
        self.vecs[i], self.actions[i], self.rewards[i] = vec, action, reward
        # terminal: next_vecs is never read back, done kills the bootstrap
        self.next_vecs[i] = 0.0 if next_vec is None else next_vec
        self.dones[i] = float(done)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def rewind(self):  # undo the most recent push so it can be replaced
        self.pos = (self.pos - 1) % self.capacity
        self.size = max(0, self.size - 1)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.vecs[idx], self.actions[idx], self.rewards[idx],
                self.next_vecs[idx], self.dones[idx])


def setup_training(self):
    """Called after `setup` in callbacks.py when running with --train."""
    self.buffer = ReplayBuffer(BUFFER_SIZE)
    self.rng = np.random.default_rng()
    self.target_net = QNet().to(self.device)
    self.target_net.load_state_dict(self.q_net.state_dict())
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.SmoothL1Loss()
    self.eps_start = EPS_START_FRESH if self.fresh_model else EPS_START_WARM
    self.total_rounds, self.score_window, self.best_mean_score = 0, [], None
    self.epsilon = _epsilon(self)
    _reset_round(self)
    # model.pt is all that outlives the process, so save it at exit too: without
    # this the last <SAVE_EVERY rounds of a run would be thrown away
    atexit.register(_save, self)
    self.logger.info(f"Training from {'scratch' if self.fresh_model else 'model.pt'}, "
                     f"eps {self.epsilon:.3f}, device {self.device}.")


def _reset_round(self):
    self.losses, self.round_reward = [], 0.0
    self.round_coins = self.round_kills = 0
    self.last_key, self.last_contribution = None, (0.0, 0, 0)


def _epsilon(self):  # linear decay over the rounds of *this* process
    progress = min(1.0, self.total_rounds / EPS_DECAY_ROUNDS)
    return max(EPS_END, self.eps_start - (self.eps_start - EPS_END) * progress)


def _target_kind_and_distance(vec):
    """Target to chase (coin, then crate, then enemy) and its distance; 1.0 means none."""
    for kind, i in (("coin", IDX_COIN_DIST), ("crate", IDX_CRATE_DIST), ("enemy", IDX_ENEMY_DIST)):
        if vec[i] < 0.999:
            return kind, float(vec[i])
    return None, None


def _add_custom_events(old_vec, action, new_vec, events):
    """Dense shaping from the features we already computed. Nothing punishes
    unsafe moves: the action mask already forbids them, and where it degrades to
    legal-only there was no safe choice to prefer."""
    if action == 5:  # BOMB
        useful = old_vec[IDX_CRATES_IN_BLAST] > 0 or old_vec[IDX_ENEMY_IN_BLAST] > 0.5
        events.append(USEFUL_BOMB if useful else USELESS_BOMB)
    # Potential-based shaping on distance to the current target, only while the
    # target kind is unchanged: switching to a far-off coin must not be punished.
    if new_vec is not None:
        old_kind, old_dist = _target_kind_and_distance(old_vec)
        new_kind, new_dist = _target_kind_and_distance(new_vec)
        if old_kind is not None and old_kind == new_kind:
            if new_dist < old_dist:
                events.append(MOVED_TOWARDS_TARGET)
            elif new_dist > old_dist:
                events.append(MOVED_AWAY_FROM_TARGET)


def reward_from_events(self, events: List[str], old_vec=None, action=None) -> float:
    total = 0.0
    # the environment raises both KILLED_SELF and GOT_KILLED for a suicide;
    # paying both would make bombing look far worse than dying to an opponent
    if e.KILLED_SELF in events:
        events = [ev for ev in events if ev != e.GOT_KILLED]
    for event in events:
        if event in GAME_REWARDS:
            total += GAME_REWARDS[event]
        elif event in CUSTOM_REWARDS:
            total += CUSTOM_REWARDS[event]
    if USEFUL_BOMB in events and old_vec is not None:
        total += CRATE_BOMB_BONUS * (old_vec[IDX_CRATES_IN_BLAST] * 4.0)
        if old_vec[IDX_ENEMY_IN_BLAST] > 0.5:
            total += ENEMY_BOMB_BONUS
    return total


def _cached_or_recompute(self, game_state):  # features the last `act` saw
    cached = getattr(self, "cached_features", None)
    if cached is not None and cached[0] == game_state["round"] and cached[1] == game_state["step"]:
        return cached[2]
    self.logger.warning("Feature cache miss; recomputing.")
    return state_to_features(game_state)[0]


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action is None:
        return
    # the framework passes its own list by reference and reuses it in
    # end_of_round, so shape a copy instead of appending to it
    events = list(events)
    old_vec = _cached_or_recompute(self, old_game_state)
    new_vec = state_to_features(new_game_state)[0]
    action = ACTIONS.index(self_action)
    _add_custom_events(old_vec, action, new_vec, events)
    reward = reward_from_events(self, events, old_vec, action)
    key = (old_game_state["round"], old_game_state["step"])
    _record(self, key, events, reward, (old_vec, action, reward, new_vec, False))
    _optimize(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Final transition of the round; `last_game_state` is what `act` last saw."""
    events = list(events)  # never append to the framework's own list
    if last_action is not None:
        old_vec = _cached_or_recompute(self, last_game_state)
        action = ACTIONS.index(last_action)
        _add_custom_events(old_vec, action, None, events)
        reward = reward_from_events(self, events, old_vec, action)
        # When the agent survives the final step, the environment delivers that
        # step's events twice: once through game_events_occurred and again here
        # with SURVIVED_ROUND appended. _record replaces rather than duplicates.
        key = (last_game_state["round"], last_game_state["step"])
        _record(self, key, events, reward, (old_vec, action, reward, None, True))
    for _ in range(UPDATES_PER_ROUND_END):
        _optimize(self)
    self.total_rounds += 1
    _log_round(self, last_game_state, events)
    self.epsilon = _epsilon(self)
    _reset_round(self)
    if self.total_rounds % SAVE_EVERY == 0:
        _save(self)


def _record(self, key, events, reward, transition):
    """Store one transition, replacing the previous one if it is the same step."""
    if key == self.last_key:
        self.buffer.rewind()
        old_reward, old_coins, old_kills = self.last_contribution
        self.round_reward -= old_reward
        self.round_coins -= old_coins
        self.round_kills -= old_kills
    coins = sum(1 for ev in events if ev == e.COIN_COLLECTED)
    kills = sum(1 for ev in events if ev == e.KILLED_OPPONENT)
    self.round_reward += reward
    self.round_coins += coins
    self.round_kills += kills
    self.buffer.push(*transition)
    self.last_key = key
    self.last_contribution = (reward, coins, kills)


def _optimize(self):
    if len(self.buffer) < max(LEARN_START, BATCH_SIZE):
        return
    vecs, actions, rewards, next_vecs, dones = self.buffer.sample(BATCH_SIZE, self.rng)
    # one random symmetry per batch: cheap, and the whole point of the layout
    k, flip = TRANSFORMS[self.rng.integers(len(TRANSFORMS))]
    vecs, actions = transform_batch(vecs, actions, k, flip)
    next_vecs, _ = transform_batch(next_vecs, None, k, flip)
    next_masks = action_mask(next_vecs)
    vecs_t, actions_t, rewards_t, dones_t, next_vecs_t, next_masks_t = (
        torch.from_numpy(a).to(self.device)
        for a in (vecs, actions, rewards, dones, next_vecs, next_masks))
    with torch.no_grad():
        # Double DQN: online net picks the action, target net scores it
        online_next = self.q_net(next_vecs_t)
        online_next = online_next.masked_fill(~next_masks_t, -float("inf"))
        best = online_next.argmax(dim=1, keepdim=True)
        target_next = self.target_net(next_vecs_t).gather(1, best).squeeze(1)
        targets = rewards_t + GAMMA * (1.0 - dones_t) * target_next
    predicted = self.q_net(vecs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    loss = self.criterion(predicted, targets)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
    self.optimizer.step()
    with torch.no_grad():  # soft target update
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.mul_(1.0 - TAU).add_(param, alpha=TAU)
    self.losses.append(float(loss.item()))


def _save(self, path=MODEL_FILE):
    torch.save({k: v.detach().cpu() for k, v in self.q_net.state_dict().items()}, path)
    self.logger.info(f"Saved {os.path.basename(path)} after {self.total_rounds} rounds.")


def _update_best(self, score):
    """Keep `best.pt` at the best rolling mean score of this process (in memory
    only, so it guards against a late collapse within a run)."""
    self.score_window.append(float(score))
    del self.score_window[:-SCORE_WINDOW]  # no-op while the window is not full
    if len(self.score_window) < SCORE_WINDOW:
        return
    mean_score = float(np.mean(self.score_window))
    if self.best_mean_score is None or mean_score > self.best_mean_score:
        self.logger.info(f"New best: mean score {mean_score:.2f} over the last {SCORE_WINDOW} "
                         f"rounds (previous {self.best_mean_score}).")
        self.best_mean_score = mean_score
        _save(self, BEST_FILE)


def _log_round(self, last_game_state, events):
    # last_game_state["self"][1] predates the final step's pickups, so score is
    # reconstructed from the events we counted instead
    score = self.round_coins * s.REWARD_COIN + self.round_kills * s.REWARD_KILL
    _update_best(self, score)
    row = {
        "round": self.total_rounds, "score": score, "reward": round(self.round_reward, 2),
        "steps": last_game_state["step"] if last_game_state else 0,
        "epsilon": round(self.epsilon, 4),
        "loss": round(float(np.mean(self.losses)), 5) if self.losses else "",
        "coins": self.round_coins, "kills": self.round_kills,
        "suicide": int(e.KILLED_SELF in events), "survived": int(e.SURVIVED_ROUND in events),
    }
    write_header = not os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
