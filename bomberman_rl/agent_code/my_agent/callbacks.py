import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = {}
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration: random action to learn new interactions
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    # Exploitation: choose action with highest q value
    state = state_to_features(game_state)
    q_values = {a: self.model.get((state, a), 0.0) for a in ACTIONS}
    # return max(q_values, key=q_values.get)

    # Alternatively: Return a random action if multiple actions share the same q-value

    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]
    return random.choice(best_actions)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

     # Load agent position, field
    x, y = game_state["self"][3]
    field = game_state["field"]

    # F1: Direction of closest coin (9 possible states)
    coins = game_state["coins"]

    if coins:
        nearest_coin = min(
            coins,
            key=lambda c: abs(c[0] - x) + abs(c[1] - y)
        )

        dx = nearest_coin[0] - x
        dy = nearest_coin[1] - y
        
        direction_x = np.sign(dx)  # -1 = left, 0 = same as agent x, 1 = right
        direction_y = np.sign(dy)  # -1 = up,  0 = same as agent y, 1 = down
        coin_direction = (direction_x, direction_y)

    else: coin_direction = "NO COIN"

    # F2: Wall or crate in each direction of character? (jeweils True oder False)
    wall_up = field[x, y - 1] != 0
    wall_right = field[x + 1, y] != 0
    wall_down = field[x, y + 1] != 0
    wall_left = field[x - 1, y] != 0

    return (coin_direction, 
            wall_up, wall_right, wall_down, wall_left
    )