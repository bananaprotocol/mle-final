import os
import pickle
import random

import numpy as np
from collections import deque


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
        return np.random.choice(ACTIONS, p=[.15, .15, .15, .15, .1, .3])

    self.logger.debug("Querying model for action.")
    # Exploitation: choose action with highest q value
    state = state_to_features(game_state)
    q_values = {a: self.model.get((state, a), 0.0) for a in ACTIONS}
    # return max(q_values, key=q_values.get)

    if not self.train:
        self.logger.info(
            f"STATE={state} | Q="
            + str({a: self.model.get((state, a), 0.0) for a in ACTIONS})
        )

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

    # F1: Direction of 1. nearest coin, if there are no coins 2. nearest crate (9 possible states)
    navigation_target = get_navigation_target(game_state)

    # F2: Wall or crate in each direction of agent? (jeweils True oder False)
    wall_up = field[x, y - 1] != 0
    wall_right = field[x + 1, y] != 0
    wall_down = field[x, y + 1] != 0
    wall_left = field[x - 1, y] != 0

    # F3: Can the agent drop a bomb? (Or is it on cooldown)
    can_bomb = game_state["self"][2]

    # F4: Is the Agent currently in danger/on an explosion tile?
    in_danger = (x, y) in get_explosion_tiles(game_state)

    # F5: Is there an escape route, if the agent would drop a bomb now?
    has_escape = has_escape_route(game_state)

    # F6: If the agent is currently in danger, direction of escape
    if in_danger:
        escape_direction = get_escape_direction(game_state)
    else:
        escape_direction = "NONE"

    # F7: Place bomb here if it can hit at least 1 crate
    should_bomb_here = count_crates_in_blast(game_state) > 0

    return (
            navigation_target, 
            wall_up, wall_right, wall_down, wall_left,
            can_bomb,
            in_danger,
            has_escape,
            escape_direction,
            should_bomb_here
    )


def get_navigation_target(game_state):
    x, y = game_state["self"][3]
    if game_state["coins"]:
        target = min(game_state["coins"], key=lambda c: abs(c[0]-x)+abs(c[1]-y))
    else:
        crates = list(zip(*np.where(game_state["field"] == 1)))
        if crates:
            target = min(crates, key=lambda c: abs(c[0]-x)+abs(c[1]-y))
        else:
            return "NONE"
    dx, dy = target[0]-x, target[1]-y
    return (np.sign(dx), np.sign(dy))


def has_escape_route(game_state):
    x, y = game_state["self"][3]
    field = game_state["field"]
    explosion = get_explosion_tiles(game_state)

    # Own position -> If agent drops a bomb now, is there a safe route to escape?
    explosion.add((x, y))

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, 4):
            nx, ny = x + dx * i, y + dy * i

            if field[nx][ny] == -1:
                break

            explosion.add((nx, ny))

    queue = deque([(x, y, 0)])
    visited = {(x, y)}

    while queue:
        x, y, distance = queue.popleft()

        if distance >= 4:
            continue

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy

            if field[nx][ny] != 0 or (nx, ny) in visited:
                continue

            visited.add((nx, ny))

            if (nx, ny) not in explosion:
                return True

            queue.append((nx, ny, distance + 1))

    return False


def get_explosion_tiles(game_state):
    field = game_state["field"]
    explosion = set()

    bombs = game_state["bombs"]

    for (x, y), timer in bombs:
        explosion.add((x, y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, 4):
                nx, ny = x + dx * i, y + dy * i

                if field[nx][ny] == -1:
                    break

                explosion.add((nx, ny))

    return explosion


def count_crates_in_blast(game_state):
    field = game_state["field"]
    x, y = game_state["self"][3]
    count = 0

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, 4):
            nx, ny = x + dx * i, y + dy * i

            if field[nx][ny] == -1:
                break

            if field[nx][ny] == 1:
                count += 1

    return count


def get_escape_direction(game_state):
    x, y = game_state["self"][3]
    field = game_state["field"]
    explosion = get_explosion_tiles(game_state)

    queue = deque([(x, y, None, 0)])
    visited = {(x, y)}

    while queue:
        cx, cy, first_direction, distance = queue.popleft()
        if distance >= 4:
            continue
        for dx, dy, direction in [
            (1, 0, "RIGHT"), (-1, 0, "LEFT"), (0, 1, "DOWN"), (0, -1, "UP")
        ]:
            nx, ny = cx + dx, cy + dy
            if field[nx][ny] != 0 or (nx, ny) in visited:
                continue
            new_direction = direction if first_direction is None else first_direction
            if (nx, ny) not in explosion:
                return new_direction
            visited.add((nx, ny))
            queue.append((nx, ny, new_direction, distance + 1))

    return "NO_ESCAPE"