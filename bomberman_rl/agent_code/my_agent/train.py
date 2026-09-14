from collections import namedtuple, deque

import pickle
import numpy as np
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS, get_explosion_tiles, count_crates_in_blast, has_escape_route, get_escape_direction

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
PATH_BLOCKED = "PATH_BLOCKED"

SURVIVED_BOMB = "SURVIVED_BOMB"
ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"

SAFE_BOMB = "SAFE_BOMB"
WASTED_BOMB = "WASTED_BOMB"

BOMB_HITS_1_CRATE = "BOMB_HITS_1_CRATE"
BOMB_HITS_2_CRATES = "BOMB_HITS_2_CRATES"
BOMB_HITS_3_PLUS_CRATES = "BOMB_HITS_3_PLUS_CRATES"

MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"

MOVED_TOWARDS_ESCAPE = "MOVED_TOWARDS_ESCAPE"

ALPHA = 0.1
GAMMA = 0.9


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.last_action = None
    self.position_history = deque(maxlen=8)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # CE1/2: Moved to or away from coin?
    old_coin_distance = distance_to_coin(old_game_state)
    new_coin_distance = distance_to_coin(new_game_state)

    if new_coin_distance < old_coin_distance:
        events.append(MOVED_TOWARDS_COIN)
    elif new_coin_distance > old_coin_distance:
        events.append(MOVED_AWAY_FROM_COIN)

    # CE3: Obstacle between agent and current target?
    if is_path_blocked(old_game_state):
        events.append(PATH_BLOCKED)

    # CE4: Entered or escaped danger?
    old_x, old_y = old_game_state["self"][3]
    old_in_danger = (old_x, old_y) in get_explosion_tiles(old_game_state)
    x, y = new_game_state["self"][3]
    in_danger = (x, y) in get_explosion_tiles(new_game_state)

    if old_in_danger and not in_danger:
        events.append(ESCAPED_DANGER)

    elif not old_in_danger and in_danger and e.BOMB_DROPPED not in events:
        events.append(ENTERED_DANGER)

    # CE5: How many crates did we hit
    if e.BOMB_DROPPED in events:
        crates_in_blast = count_crates_in_blast(old_game_state)

        if crates_in_blast == 0:
            events.append(WASTED_BOMB)
        elif crates_in_blast == 1:
            events.append(BOMB_HITS_1_CRATE)
        elif crates_in_blast == 2:
            events.append(BOMB_HITS_2_CRATES)
        elif crates_in_blast >= 3:
            events.append(BOMB_HITS_3_PLUS_CRATES)

    # CE6: Moved towards or away from crate?
    if not old_in_danger:  # only if not in explosion, so running away from bomb is not punished
        old_crate_distance = distance_to_crate(old_game_state)
        new_crate_distance = distance_to_crate(new_game_state)

        if new_crate_distance < old_crate_distance:
            events.append(MOVED_TOWARDS_CRATE)
        elif new_crate_distance > old_crate_distance:
            events.append(MOVED_AWAY_FROM_CRATE)

    # CE7: Bomb placement
    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(SURVIVED_BOMB)

    if e.BOMB_DROPPED in events and has_escape_route(old_game_state):
        events.append(SAFE_BOMB)

    # CE8: Moved towards escape?
    old_x, old_y = old_game_state["self"][3]
    old_in_danger = (old_x, old_y) in get_explosion_tiles(old_game_state)

    if old_in_danger:
        old_escape_dir = get_escape_direction(old_game_state)
        if old_escape_dir != "NO_ESCAPE" and self_action == old_escape_dir:
            events.append("MOVED_TOWARDS_ESCAPE")

    
    state = state_to_features(old_game_state)
    next_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state, self_action, next_state, reward))
    update_q_value(self, state, self_action, reward, next_state)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    state = state_to_features(last_game_state)
    reward = reward_from_events(self, events)

    self.transitions.append(Transition(state, last_action, None, reward))
    update_q_value(self, state, last_action, reward, None)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 0,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -3,
        e.INVALID_ACTION: -2, # prevents agent from moving against wall
        
        # Coin heaven
        MOVED_TOWARDS_COIN: .1,
        MOVED_AWAY_FROM_COIN: -.3,
        # PATH_BLOCKED: -.1,

        # Loot Crate
        ESCAPED_DANGER: .2,
        ENTERED_DANGER: -1,
        # SURVIVED_BOMB: .5,
        # SAFE_BOMB: .5,
        WASTED_BOMB: -.3,

        # BOMB_HITS_1_CRATE: .3,
        # BOMB_HITS_2_CRATES: .5,
        # BOMB_HITS_3_PLUS_CRATES: .7,

        MOVED_TOWARDS_CRATE: .05,
        MOVED_AWAY_FROM_CRATE: -.1,
        # MOVED_TOWARDS_ESCAPE: 0.2,

        e.CRATE_DESTROYED: .3,
        e.BOMB_DROPPED: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def update_q_value(self, state, action, reward, next_state):
    """
        Q-learning update function similar to lecture 4.2
    """
    old_q = self.model.get((state, action), 0.0)

    if next_state is None:
        future_q = 0.0
    else:
        future_q = max(
            self.model.get((next_state, a), 0.0)
            for a in ACTIONS
        )

    new_q = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

    self.model[(state, action)] = new_q


def distance_to_coin(game_state):
    x, y = game_state["self"][3]
    coins = game_state["coins"]

    if not coins:
        return 0

    return min(
        abs(cx - x) + abs(cy - y)
        for cx, cy in coins
    )


def distance_to_crate(game_state):
    x, y = game_state["self"][3]
    crates = []

    for cx in range(game_state["field"].shape[0]):
        for cy in range(game_state["field"].shape[1]):
            if game_state["field"][cx, cy] == 1:
                crates.append((cx, cy))

    if not crates:
        return 0

    return min(
        abs(cx - x) + abs(cy - y)
        for cx, cy in crates
    )


def is_path_blocked(game_state):
    x, y = game_state["self"][3]
    field = game_state["field"]
    
    if game_state["coins"]:
        cx, cy = min(game_state["coins"], key=lambda c: abs(c[0]-x)+abs(c[1]-y))
    else:
        crates = list(zip(*np.where(game_state["field"] == 1)))
        if not crates:
            return False
        cx, cy = min(crates, key=lambda c: abs(c[0]-x)+abs(c[1]-y))
    
    if cx == x and cy != y:
        step = 1 if cy > y else -1
        for ny in range(y + step, cy, step):
            if field[x, ny] == -1:
                return True
    if cy == y and cx != x:
        step = 1 if cx > x else -1
        for nx in range(x + step, cx, step):
            if field[nx, y] == -1:
                return True
    
    return False