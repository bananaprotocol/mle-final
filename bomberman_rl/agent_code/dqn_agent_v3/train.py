import atexit
import csv
import os
import pickle
from typing import List

import numpy as np
import torch
import torch.nn as nn

import events as e
import settings as s

from .callbacks import MODEL_FILE
from .features import (ACTIONS, CROP_SHAPE, IDX_CRATES_IN_BLAST, IDX_COIN_DIST,
                       IDX_CRATE_DIST, IDX_ENEMY_DIST, IDX_ENEMY_IN_BLAST,
                       VEC_SIZE, legal_action_mask, masks_from_vecs,
                       robust_action_mask, safe_action_mask, state_to_features)
from .model import TRANSFORMS, QNet, transform_batch

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(AGENT_DIR, "train_state.pkl")
LOG_FILE = os.path.join(AGENT_DIR, "train_log.csv")
REPLAY_FILE = os.path.join(AGENT_DIR, "replay.npz")
BEST_FILE = os.path.join(AGENT_DIR, "best.pt")

# hyper parameters
BUFFER_SIZE = 200_000
BATCH_SIZE = 128
GAMMA = 0.95
LEARNING_RATE = 2.5e-4
TAU = 0.005          # soft target update per gradient step
LEARN_START = 2_000  # transitions before the first update
GRAD_CLIP = 10.0
UPDATES_PER_ROUND_END = 20
SAVE_EVERY = 25          # rounds between model saves
BUFFER_SAVE_EVERY = 200  # rounds between buffer saves, ~300 MB each
SCORE_WINDOW = 100       # rounds averaged for the best checkpoint

EPS_END = 0.05
EPS_DECAY_ROUNDS = 400
EPS_START_FIRST_STAGE = 1.0
EPS_START_LATER_STAGE = 0.3

# reward shaping; death outweighs coins on purpose, or coin greed wins
GAME_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    e.KILLED_OPPONENT: 50.0,
    e.KILLED_SELF: -50.0,
    e.GOT_KILLED: -30.0,
    e.CRATE_DESTROYED: 2.0,
    e.COIN_FOUND: 1.0,
    e.INVALID_ACTION: -2.0,
    e.WAITED: -0.3,
    e.SURVIVED_ROUND: 5.0,
}

MOVED_TOWARDS_TARGET = "MOVED_TOWARDS_TARGET"
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
UNSAFE_MOVE = "UNSAFE_MOVE"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
USEFUL_BOMB = "USEFUL_BOMB"
USELESS_BOMB = "USELESS_BOMB"
ENTERED_TRAP = "ENTERED_TRAP"

CUSTOM_REWARDS = {
    MOVED_TOWARDS_TARGET: 1.0,
    MOVED_AWAY_FROM_TARGET: -1.0,
    UNSAFE_MOVE: -10.0,
    SUICIDAL_BOMB: -15.0,
    USELESS_BOMB: -3.0,
    ENTERED_TRAP: -8.0,
}
CRATE_BOMB_BONUS = 2.0     # per crate in the blast, up to 4 crates
ENEMY_BOMB_BONUS = 10.0


class ReplayBuffer:
    """Flat ring buffer. Crops are float16 to halve the footprint."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.vecs = np.zeros((capacity, VEC_SIZE), dtype=np.float32)
        self.crops = np.zeros((capacity,) + CROP_SHAPE, dtype=np.float16)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_vecs = np.zeros((capacity, VEC_SIZE), dtype=np.float32)
        self.next_crops = np.zeros((capacity,) + CROP_SHAPE, dtype=np.float16)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, vec, crop, action, reward, next_vec, next_crop, done):
        i = self.pos
        self.vecs[i] = vec
        self.crops[i] = crop
        self.actions[i] = action
        self.rewards[i] = reward
        if next_vec is None:  # terminal, and `done` kills the bootstrap
            self.next_vecs[i] = 0.0
            self.next_crops[i] = 0.0
        else:
            self.next_vecs[i] = next_vec
            self.next_crops[i] = next_crop
        self.dones[i] = float(done)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def rewind(self):
        self.pos = (self.pos - 1) % self.capacity
        self.size = max(0, self.size - 1)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.vecs[idx], self.crops[idx].astype(np.float32),
                self.actions[idx], self.rewards[idx],
                self.next_vecs[idx], self.next_crops[idx].astype(np.float32),
                self.dones[idx])

    FIELDS = ("vecs", "crops", "actions", "rewards", "next_vecs", "next_crops",
              "dones")

    def save(self, path):
        # temp file first: Ctrl-C during a save left a broken npz behind
        arrays = {name: getattr(self, name)[:self.size] for name in self.FIELDS}
        arrays["pos"] = np.int64(self.pos)
        arrays["size"] = np.int64(self.size)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)

    def load(self, path):
        """Number of transitions restored. 0 if there is nothing usable."""
        if not os.path.isfile(path):
            return 0
        with np.load(path) as data:
            stored = int(data["size"]) if "size" in data else int(len(data["vecs"]))
            n = min(stored, self.capacity, int(len(data["vecs"])))
            if n <= 0:
                return 0
            for name in self.FIELDS:
                target = getattr(self, name)
                source = data[name]
                if source.shape[1:] != target.shape[1:]:
                    raise ValueError(f"{name} has shape {source.shape[1:]}, "
                                     f"expected {target.shape[1:]}")
                target[:n] = source[:n]
            # a full ring keeps its cursor, so the oldest data still goes first
            pos = int(data["pos"]) if ("pos" in data and n == stored) else n
        self.size = n
        self.pos = pos % self.capacity
        return n


def setup_training(self):
    self.stage = int(os.environ.get("DQN_STAGE", 1))
    self.buffer = ReplayBuffer(BUFFER_SIZE)
    self.rng = np.random.default_rng()

    self.target_net = QNet().to(self.device)
    self.target_net.load_state_dict(self.q_net.state_dict())
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.SmoothL1Loss()

    self.total_rounds = 0
    self.stage_round = 0
    self.score_window = []
    self.best_mean_score = None
    previous_stage = None
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE, "rb") as fh:
            saved = pickle.load(fh)
        self.total_rounds = saved.get("total_rounds", 0)
        previous_stage = saved.get("stage")
        if previous_stage == self.stage:
            # counted per stage, and stage 3 runs as many short processes
            self.stage_round = saved.get("stage_round", 0)
            self.score_window = list(saved.get("score_window", []))[-SCORE_WINDOW:]
            self.best_mean_score = saved.get("best_mean_score")
    if previous_stage is not None and previous_stage != self.stage:
        self.logger.info(f"Stage {previous_stage} -> {self.stage}: restarting exploration.")

    _load_buffer(self)
    self.epsilon = _epsilon(self)
    _reset_round(self)
    atexit.register(_save, self, True)
    self.logger.info(f"Training stage {self.stage}, round {self.total_rounds}, "
                     f"stage round {self.stage_round}, epsilon {self.epsilon:.3f}, "
                     f"buffer {len(self.buffer)}, device {self.device}.")


def _reset_round(self):
    self.losses = []
    self.round_reward = 0.0
    self.round_coins = 0
    self.round_kills = 0
    self.last_key = None
    self.last_contribution = (0.0, 0, 0)


def _load_buffer(self):
    try:
        loaded = self.buffer.load(REPLAY_FILE)
    except Exception as exc:  # truncated, stale layout, unreadable
        self.logger.warning(f"Could not load {REPLAY_FILE} ({exc}); starting empty.")
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        return
    if loaded:
        self.logger.info(f"Loaded {loaded} transitions from {REPLAY_FILE}.")
    else:
        self.logger.info("No replay buffer on disk; starting empty.")


def _epsilon(self):
    start = EPS_START_FIRST_STAGE if self.stage == 1 else EPS_START_LATER_STAGE
    progress = min(1.0, self.stage_round / EPS_DECAY_ROUNDS)
    return max(EPS_END, start - (start - EPS_END) * progress)


def _target_kind_and_distance(vec):
    """Target to chase and its distance. Coins, then crates, then opponents."""
    for kind, idx in (("coin", IDX_COIN_DIST), ("crate", IDX_CRATE_DIST),
                      ("enemy", IDX_ENEMY_DIST)):
        if vec[idx] < 0.999:
            return kind, float(vec[idx])
    return None, None


def _robust_options(vec):
    """Top rung of the mask ladder. Empty means we stand in a trap."""
    return legal_action_mask(vec) & safe_action_mask(vec) & robust_action_mask(vec)


def _add_custom_events(self, old_vec, action, new_vec, events):
    """Dense shaping on top of the framework events. `events` must be a copy."""
    # only fires once the mask itself fell back to "merely legal"
    safe = safe_action_mask(old_vec)
    if safe.any() and not safe[action]:
        events.append(UNSAFE_MOVE)

    if action == 5:  # BOMB
        if not safe[5]:
            events.append(SUICIDAL_BOMB)
        crates = old_vec[IDX_CRATES_IN_BLAST] * 4.0
        enemy = old_vec[IDX_ENEMY_IN_BLAST] > 0.5
        if crates > 0 or enemy:
            events.append(USEFUL_BOMB)
        else:
            events.append(USELESS_BOMB)

    # we walked into a corridor an opponent can bomb. A penalty on the
    # non-robust action itself would never fire, as the mask blocks it.
    if new_vec is not None and _robust_options(old_vec).any() \
            and not _robust_options(new_vec).any():
        events.append(ENTERED_TRAP)

    # one kind of target only: take a coin, switch to a far one, and that must
    # not count as a move away
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
    # a suicide raises both; paying both makes our bomb worse than theirs
    if e.KILLED_SELF in events:
        events = [ev for ev in events if ev != e.GOT_KILLED]
    for event in events:
        if event in GAME_REWARDS:
            total += GAME_REWARDS[event]
        elif event in CUSTOM_REWARDS:
            total += CUSTOM_REWARDS[event]
    if USEFUL_BOMB in events and old_vec is not None:
        crates = old_vec[IDX_CRATES_IN_BLAST] * 4.0
        total += CRATE_BOMB_BONUS * crates
        if old_vec[IDX_ENEMY_IN_BLAST] > 0.5:
            total += ENEMY_BOMB_BONUS
    return total


def _cached_or_recompute(self, game_state):
    """Reuse what act() built for this step."""
    cached = getattr(self, "cached_features", None)
    if cached is not None and cached[0] == game_state["round"] and cached[1] == game_state["step"]:
        return cached[2], cached[3]
    self.logger.warning("Feature cache miss; recomputing.")
    vec, crop, _ = state_to_features(game_state)
    return vec, crop


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action is None:
        return

    # the framework reuses this list in end_of_round, so work on a copy
    events = list(events)
    old_vec, old_crop = _cached_or_recompute(self, old_game_state)
    new_vec, new_crop, _ = state_to_features(new_game_state)
    action = ACTIONS.index(self_action)

    _add_custom_events(self, old_vec, action, new_vec, events)
    reward = reward_from_events(self, events, old_vec, action)
    key = (old_game_state["round"], old_game_state["step"])
    _record(self, key, events, reward,
            (old_vec, old_crop, action, reward, new_vec, new_crop, False))
    _optimize(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    events = list(events)
    if last_action is not None:
        old_vec, old_crop = _cached_or_recompute(self, last_game_state)
        action = ACTIONS.index(last_action)
        _add_custom_events(self, old_vec, action, None, events)
        reward = reward_from_events(self, events, old_vec, action)
        # a survived last step arrives twice, here with SURVIVED_ROUND on top,
        # so _record replaces instead of appends
        key = (last_game_state["round"], last_game_state["step"])
        _record(self, key, events, reward,
                (old_vec, old_crop, action, reward, None, None, True))

    for _ in range(UPDATES_PER_ROUND_END):
        _optimize(self)

    self.total_rounds += 1
    self.stage_round += 1
    _log_round(self, last_game_state, events)
    self.epsilon = _epsilon(self)
    _reset_round(self)

    if self.total_rounds % SAVE_EVERY == 0:
        _save(self, save_buffer=self.total_rounds % BUFFER_SAVE_EVERY == 0)


def _record(self, key, events, reward, transition):
    """Push a transition, replacing the last one if it is the same step."""
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

    vecs, crops, actions, rewards, next_vecs, next_crops, dones = \
        self.buffer.sample(BATCH_SIZE, self.rng)

    # one random symmetry per batch, which is what the block layout is for
    k, flip = TRANSFORMS[self.rng.integers(len(TRANSFORMS))]
    vecs, crops, actions = transform_batch(vecs, crops, actions, k, flip)
    next_vecs, next_crops, _ = transform_batch(next_vecs, next_crops, None, k, flip)
    next_masks = masks_from_vecs(next_vecs)

    dev = self.device
    vecs_t = torch.from_numpy(vecs).to(dev)
    crops_t = torch.from_numpy(crops).to(dev)
    actions_t = torch.from_numpy(actions).to(dev)
    rewards_t = torch.from_numpy(rewards).to(dev)
    dones_t = torch.from_numpy(dones).to(dev)
    next_vecs_t = torch.from_numpy(next_vecs).to(dev)
    next_crops_t = torch.from_numpy(next_crops).to(dev)
    next_masks_t = torch.from_numpy(next_masks).to(dev)

    with torch.no_grad():
        # Double DQN: the online net picks the action, the target net scores it
        online_next = self.q_net(next_vecs_t, next_crops_t)
        online_next = online_next.masked_fill(~next_masks_t, -float("inf"))
        best = online_next.argmax(dim=1, keepdim=True)
        target_next = self.target_net(next_vecs_t, next_crops_t).gather(1, best).squeeze(1)
        targets = rewards_t + GAMMA * (1.0 - dones_t) * target_next

    predicted = self.q_net(vecs_t, crops_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    loss = self.criterion(predicted, targets)

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
    self.optimizer.step()

    with torch.no_grad():
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.mul_(1.0 - TAU).add_(param, alpha=TAU)

    self.losses.append(float(loss.item()))


def _cpu_state_dict(self):
    return {k: v.detach().cpu() for k, v in self.q_net.state_dict().items()}


def _save(self, save_buffer=False):
    torch.save(_cpu_state_dict(self), MODEL_FILE)
    with open(STATE_FILE, "wb") as fh:
        pickle.dump({"total_rounds": self.total_rounds, "stage": self.stage,
                     "stage_round": self.stage_round,
                     "score_window": list(self.score_window),
                     "best_mean_score": self.best_mean_score}, fh)
    if save_buffer:
        try:
            self.buffer.save(REPLAY_FILE)
        except Exception as exc:  # a failed write must not kill the run
            self.logger.warning(f"Could not save {REPLAY_FILE}: {exc}")
        else:
            self.logger.info(f"Saved {len(self.buffer)} transitions to {REPLAY_FILE}.")
    self.logger.info(f"Saved model after {self.total_rounds} rounds.")


def _update_best(self, score):
    self.score_window.append(float(score))
    if len(self.score_window) > SCORE_WINDOW:
        del self.score_window[:-SCORE_WINDOW]
    if len(self.score_window) < SCORE_WINDOW or self.stage_round < SCORE_WINDOW:
        return None
    mean_score = float(np.mean(self.score_window))
    if self.best_mean_score is not None and mean_score <= self.best_mean_score:
        return mean_score
    previous = self.best_mean_score
    self.best_mean_score = mean_score
    torch.save(_cpu_state_dict(self), BEST_FILE)
    self.logger.info(
        f"New best at stage {self.stage} round {self.stage_round}: mean score "
        f"{mean_score:.2f} over {SCORE_WINDOW} rounds (was "
        f"{previous if previous is None else round(previous, 2)})")
    return mean_score


def _log_round(self, last_game_state, events):
    # the score in last_game_state is stale for the final step, so count events
    score = self.round_coins * s.REWARD_COIN + self.round_kills * s.REWARD_KILL
    _update_best(self, score)
    steps = last_game_state["step"] if last_game_state else 0
    row = {
        "round": self.total_rounds,
        "stage": self.stage,
        "score": score,
        "reward": round(self.round_reward, 2),
        "steps": steps,
        "epsilon": round(self.epsilon, 4),
        "loss": round(float(np.mean(self.losses)), 5) if self.losses else "",
        "coins": self.round_coins,
        "kills": self.round_kills,
        "suicide": int(e.KILLED_SELF in events),
        "survived": int(e.SURVIVED_ROUND in events),
    }
    write_header = not os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
