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

ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"

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

    # CE5: Entered or escaped danger?
    old_x, old_y = old_game_state["self"][3]
    old_in_danger = (old_x, old_y) in get_explosion_tiles(old_game_state)
    x, y = new_game_state["self"][3]
    in_danger = (x, y) in get_explosion_tiles(new_game_state)

    if old_in_danger and not in_danger:
        events.append(ESCAPED_DANGER)

    elif not old_in_danger and in_danger and e.BOMB_DROPPED not in events:
        events.append(ENTERED_DANGER)

    # CE1/2: Moved to or away from coin?
    old_coin_distance = distance_to_coin(old_game_state)
    new_coin_distance = distance_to_coin(new_game_state)

    if new_coin_distance < old_coin_distance:
        events.append(MOVED_TOWARDS_COIN)
    elif new_coin_distance > old_coin_distance:
        events.append(MOVED_AWAY_FROM_COIN)

    # CE3/4: Moved towards or away from crate?
    old_crate_distance = distance_to_crate(old_game_state)
    new_crate_distance = distance_to_crate(new_game_state)

    if new_crate_distance < old_crate_distance:
        events.append(MOVED_TOWARDS_CRATE)
    elif new_crate_distance > old_crate_distance:
        events.append(MOVED_AWAY_FROM_CRATE)


    # CE7: How many crates did we hit
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

    # CE8: Moved towards escape?
    if old_in_danger:
        old_escape_distance = distance_to_escape(old_game_state)
        new_escape_distance = distance_to_escape(new_game_state)

        if new_escape_distance < old_escape_distance:
            events.append(MOVED_TOWARDS_ESCAPE)
        elif new_escape_distance > old_escape_distance:
            events.append("MOVED_AWAY_FROM_ESCAPE")

    
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
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -8,
        e.INVALID_ACTION: -2, # prevents agent from moving against wall
        
        # Coin heaven
        MOVED_TOWARDS_COIN: .1,
        MOVED_AWAY_FROM_COIN: -.2,

        # Loot Crate
        ESCAPED_DANGER: .2,
        ENTERED_DANGER: -.5,
        WASTED_BOMB: -.5,

        BOMB_HITS_1_CRATE: .2,
        BOMB_HITS_2_CRATES: .3,
        BOMB_HITS_3_PLUS_CRATES: .5,

        MOVED_TOWARDS_CRATE: .05,
        MOVED_AWAY_FROM_CRATE: -.2,
        MOVED_TOWARDS_ESCAPE: .25,

        e.CRATE_DESTROYED: .25,
        e.BOMB_DROPPED: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    # Step penalty: encourage agent to complete tasks quickly or efficiently
    reward_sum -= 0.02
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


def bfs_distance(game_state, targets, blocked=None):
    if not targets:
        return 0

    x, y = game_state["self"][3]
    field = game_state["field"]
    targets = set(targets)
    blocked = set() if blocked is None else set(blocked)

    if (x, y) in targets:
        return 0

    visited = {(x, y)}
    queue = deque([(x, y, 0)])

    while queue:
        cx, cy, dist = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited or (nx, ny) in blocked:
                continue
            if (nx, ny) in targets:
                return dist + 1
            if field[nx, ny] != 0:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny, dist + 1))

    return 999


def distance_to_coin(game_state):
    return bfs_distance(game_state, game_state["coins"])


def distance_to_crate(game_state):
    field = game_state["field"]
    crate_positions = list(zip(*np.where(field == 1)))
    return bfs_distance(game_state, crate_positions)


def distance_to_escape(game_state):
    explosion = get_explosion_tiles(game_state)

    safe_tiles = [
        (x, y)
        for x in range(game_state["field"].shape[0])
        for y in range(game_state["field"].shape[1])
        if game_state["field"][x, y] == 0
        and (x, y) not in explosion
    ]

    return bfs_distance(game_state, safe_tiles, blocked=None)