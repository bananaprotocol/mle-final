import pickle
from typing import List

import events as e
import numpy as np

from .callbacks import ACTIONS, state_key, valid_actions, _bombs_of, blast_cross


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all of your callbacks and you can set arbitrary values.
    """
    self.alpha = 0.15
    self.gamma = 0.9
    # potential-based shaping coefficient (see _coin_potential below)
    self.coin_shaping = 0.15


def _in_imminent_danger(game_state):
    """
    True if my tile is in a bomb's blast cross with <= 2 ticks left, or in a
    live explosion. Standing on a bomb tile is NOT punished here. That is a
    normal, escapable state right after dropping one (the planner handles it).
    """
    x, y = game_state["self"][3]
    if game_state["explosion_map"][x, y] >= 1:
        return True
    arena = game_state["field"]
    for (px, py), t in _bombs_of(game_state):
        if t <= 2 and (x, y) in blast_cross(px, py, arena):
            return True
    return False


def _coin_potential(self, game_state) -> float:
    """Phi(s) = -c * d(s, nearest coin); 0 when no coin is visible."""
    coins = game_state["coins"]
    x, y = game_state["self"][3]
    if not coins:
        return 0.0
    d = min(abs(int(cx) - x) + abs(int(cy) - y) for cx, cy in coins)
    return -self.coin_shaping * d


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    This is *one* of the places where you could update your agent.

    :param self: The same object that is passed to all of your callbacks.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """
    s = state_key(old_game_state)
    a = ACTIONS.index(self_action)
    r = reward_from_events(self, events)

    # Dense survival signal: being on a blast path that is about to close (or
    # in a live explosion) is bad *now*, independent of who owns the bomb.
    if _in_imminent_danger(new_game_state):
        r -= 0.5

    # Potential-based reward shaping toward the nearest coin
    # (F = gamma*Phi(s') - Phi(s)); provably keeps the optimal policy of the
    # real reward, while giving a dense gradient for navigation.
    phi0 = _coin_potential(self, old_game_state)
    phi1 = _coin_potential(self, new_game_state)
    r += self.gamma * phi1 - phi0

    s_new = state_key(new_game_state)

    q_new = self.q.setdefault(s_new, np.zeros(len(ACTIONS)))

    # Bootstrap over actions that are actually legal in the new state.
    valid = valid_actions(new_game_state)
    best = q_new[[ACTIONS.index(v) for v in valid]].max() if valid else 0.0
    target = r + self.gamma * best
    q_row = self.q.setdefault(s, np.zeros(len(ACTIONS)))
    q_row[a] += self.alpha * (target - q_row[a])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    s = state_key(last_game_state)
    a = ACTIONS.index(last_action)
    q_row = self.q.setdefault(s, np.zeros(len(ACTIONS)))

    # Full event reward for the final step (coins, kills, ...) plus a bonus
    # for surviving the round. Terminal update: no bootstrapping.
    r = reward_from_events(self, events)
    if e.SURVIVED_ROUND in events:
        r += 1.0

    q_row[a] += self.alpha * (r - q_row[a])

    with open("q_table.pkl", "wb") as f:
        pickle.dump(self.q, f)


def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # real game rewards
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.GOT_KILLED: -5.0,
        # suicide gets GOT_KILLED + KILLED_SELF
        e.KILLED_SELF: -5.0,
        # small auxiliary signals
        e.COIN_FOUND: 0.2,       # my bomb revealed a coin
        e.CRATE_DESTROYED: 0.1,  # my bomb cleared a crate
        e.INVALID_ACTION: -0.1,  # don't spam moves you know are blocked
    }
    reward_sum = 0.0
    for event in events:
        reward_sum += game_rewards.get(event, 0.0)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
