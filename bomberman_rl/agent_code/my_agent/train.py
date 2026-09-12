from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

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
        e.INVALID_ACTION: -1, # prevents agent from moving against wall
        
        # Coin heaven
        MOVED_TOWARDS_COIN: .1,
        MOVED_AWAY_FROM_COIN: -.2,
        e.WAITED: -.5,
        e.BOMB_DROPPED: -.5,
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
