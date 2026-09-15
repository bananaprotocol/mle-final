"""
Observation and safety rules for the DQN agent.

This module contains only pure functions. Each function reads one
`game_state` dictionary and gives a result. No function changes the game.

The module has three groups of functions:
  * the danger model: which tiles a bomb can hit, and when;
  * the observation: 9 board planes and 16 global values for the network;
  * the action masks: which actions are legal, and which actions keep an
    escape path open.

Timing model (taken from environment.py and items.py)
----------------------------------------------------
The world updates each step after all agents moved. Thus:
  * a bomb with the shown timer `t` in the state of step `s` explodes in the
    update of step `s + t`;
  * the blast is lethal in the update of step `s + t` and of step `s + t + 1`;
  * the tile of the agent after `k` moves is examined by the update of step
    `s + k - 1`.
Therefore the tile after `k` moves is lethal if the tile is in the blast of a
bomb with `k = t + 1` or `k = t + 2`. The agent has `t + 1` moves to leave a
blast. A bomb that the agent drops in step `s` explodes in the update of step
`s + BOMB_TIMER`, so the planner gives it the timer `BOMB_TIMER`.
"""

from collections import deque

import numpy as np

# Action order. All masks and all network outputs use this order.
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
UP, RIGHT, DOWN, LEFT, WAIT, BOMB = range(6)
N_ACTIONS = 6

# Tile offset of each move action, in the same order as ACTIONS.
MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0))

# These values must agree with settings.py of the framework.
BOMB_POWER = 3
BOMB_TIMER = 4

BOARD = 17         # board width and height (COLS and ROWS of settings.py)
BFS_HORIZON = 8    # maximum number of moves that the escape planner examines
DANGER_LEVELS = 5  # a danger plane value of 5 means "no bomb is near"
DANGER_PLANE = 7   # index of the danger plane in the plane stack
CAP = 24           # maximum walk distance that the potential function uses

N_PLANES = 9
N_GLOBAL = 16


# ---------------------------------------------------------------------------
# danger model
# ---------------------------------------------------------------------------

def bombs_of(game_state):
    """Give the bombs as [((x, y), timer), ...] with plain integers."""
    return [((int(x), int(y)), int(t)) for (x, y), t in game_state["bombs"]]


def other_tiles(game_state):
    """Give the tiles of the opponents as plain integer pairs."""
    return {(int(x), int(y)) for _n, _s, _b, (x, y) in game_state["others"]}


def blast_cross(x, y, field):
    """
    Give all tiles that a bomb at (x, y) hits.

    The blast is a cross with the length BOMB_POWER in each direction. A
    stone wall (-1) stops the blast. A crate (1) does not stop the blast,
    because the blast destroys the crate.
    """
    width, height = field.shape
    cross = {(x, y)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < width and 0 <= ny < height) or field[nx, ny] == -1:
                break
            cross.add((nx, ny))
    return cross


def danger_ticks(game_state):
    """
    Give the number of steps until a bomb can kill, for each tile.

    A value of 0 means that the tile is lethal in this step. A value of
    DANGER_LEVELS means that no bomb threatens the tile soon.
    """
    field = game_state["field"]
    ticks = np.full(field.shape, DANGER_LEVELS, dtype=np.int8)
    ticks[game_state["explosion_map"] >= 1] = 0
    for (bx, by), timer in bombs_of(game_state):
        value = min(timer, DANGER_LEVELS)
        for tx, ty in blast_cross(bx, by, field):
            if value < ticks[tx, ty]:
                ticks[tx, ty] = value
    return ticks


# ---------------------------------------------------------------------------
# walk distances (coin target and crate target)
# ---------------------------------------------------------------------------

def _walk_map(game_state):
    """
    Walk from the agent tile over the free tiles.

    Stone walls, crates and bombs stop the walk. Opponents do not stop the
    walk, because they move away.

    Returns two arrays:
      dist   number of moves to each tile, or -1 if no path exists;
      first  index of the first move of that path, or -1 for the start tile.
    """
    field = game_state["field"]
    width, height = field.shape
    bomb_tiles = {pos for pos, _t in bombs_of(game_state)}
    x, y = game_state["self"][3]

    dist = np.full(field.shape, -1, dtype=np.int16)
    first = np.full(field.shape, -1, dtype=np.int8)
    dist[x, y] = 0
    queue = deque([(x, y)])

    while queue:
        cx, cy = queue.popleft()
        for action, (dx, dy) in enumerate(MOVES):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if dist[nx, ny] >= 0 or field[nx, ny] != 0 or (nx, ny) in bomb_tiles:
                continue
            dist[nx, ny] = dist[cx, cy] + 1
            first[nx, ny] = action if (cx, cy) == (x, y) else first[cx, cy]
            queue.append((nx, ny))
    return dist, first


def _nearest(dist, first, targets):
    """
    Give (distance, first move) for the nearest tile of `targets`.

    The distance is never more than CAP. The first move is -1 if no target
    is in reach, or if the agent already stands on the nearest target.
    """
    usable = targets & (dist >= 0)
    if not usable.any():
        return CAP, -1
    costs = np.where(usable, dist.astype(np.int32), 10 ** 4)
    tx, ty = np.unravel_index(int(costs.argmin()), dist.shape)
    return min(int(dist[tx, ty]), CAP), int(first[tx, ty])


def _crate_targets(field):
    """Give a mask of the free tiles that touch a crate."""
    crate = field == 1
    near = np.zeros_like(crate)
    near[:-1, :] |= crate[1:, :]
    near[1:, :] |= crate[:-1, :]
    near[:, :-1] |= crate[:, 1:]
    near[:, 1:] |= crate[:, :-1]
    return near & (field == 0)


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------

def observation(game_state):
    """
    Build the input of the network.

    Returns four values:
      planes    uint8 array (9, 17, 17). Plane 7 holds the danger level 0 ... 5.
                Divide plane 7 by DANGER_LEVELS before the forward pass.
      glob      float32 array with 16 global values.
      d_target  walk distance to the current target (a coin, or a crate).
      my_danger number of steps until a bomb can kill on the agent tile.

    The last two values are not part of the network input. The training code
    uses them for the potential function of the reward shaping.
    """
    field = game_state["field"]
    _name, _score, bombs_left, (x, y) = game_state["self"]
    coins = [(int(cx), int(cy)) for cx, cy in game_state["coins"]]
    others = game_state["others"]

    planes = np.zeros((N_PLANES, field.shape[0], field.shape[1]), dtype=np.uint8)
    planes[0] = field == -1
    planes[1] = field == 1
    for cx, cy in coins:
        planes[2, cx, cy] = 1
    for _n, _s, other_bomb, (ox, oy) in others:
        planes[3, ox, oy] = 1
        if other_bomb:
            planes[4, ox, oy] = 1
    planes[5, x, y] = 1
    for (bx, by), _t in bombs_of(game_state):
        planes[6, bx, by] = 1

    ticks = danger_ticks(game_state)
    planes[DANGER_PLANE] = DANGER_LEVELS - ticks
    planes[8] = game_state["explosion_map"] >= 1

    # Walk distances. One search gives both directions and both distances.
    dist, first = _walk_map(game_state)
    coin_mask = np.zeros(field.shape, dtype=bool)
    for cx, cy in coins:
        coin_mask[cx, cy] = True
    d_coin, coin_dir = _nearest(dist, first, coin_mask)
    d_crate, crate_dir = _nearest(dist, first, _crate_targets(field))

    # The target is the nearest coin. If no coin is visible, the target is a
    # tile next to a crate. A constant cap keeps the potential continuous
    # when a crate reveals a coin.
    if coins:
        d_target = d_coin
    elif (field == 1).any():
        d_target = d_crate
    else:
        d_target = CAP

    glob = np.zeros(N_GLOBAL, dtype=np.float32)
    glob[0] = 1.0 if bombs_left else 0.0
    glob[1] = len(others) / 3.0
    glob[2] = len(coins) / 50.0
    glob[3] = int(game_state["step"]) / 400.0
    # A direction of -1 means "no target". It lands on the first slot of the
    # block: glob[4] and glob[9]. The four moves follow in the order of ACTIONS.
    glob[5 + coin_dir] = 1.0    # 4 = no direction, 5 ... 8 = UP ... LEFT
    glob[10 + crate_dir] = 1.0  # 9 = no direction, 10 ... 13 = UP ... LEFT
    # The two walk distances. The network needs them to represent the potential
    # Phi of the reward shaping; a direction alone is not sufficient. Two 3x3
    # convolutions see 5 x 5 tiles, so the network cannot count a path itself.
    glob[14] = d_coin / CAP
    glob[15] = d_crate / CAP

    return planes, glob, int(d_target), int(ticks[x, y])


# ---------------------------------------------------------------------------
# escape planner (the safety layer)
# ---------------------------------------------------------------------------

def _escape_exists(game_state, start, start_depth, extra_bomb=None):
    """
    Tell if the agent can reach a safe tile from `start`.

    `start` is the tile of the agent after `start_depth` moves. A tile is
    safe when no bomb blast can touch it any more. `extra_bomb` adds one new
    bomb at the given tile, which is the bomb that the BOMB action drops.

    The search gives no false "yes": it accepts a path only when every tile
    of the path is out of the fire at its own depth. See the timing model at
    the top of this file.
    """
    field = game_state["field"]
    exp_map = game_state["explosion_map"]
    width, height = field.shape

    bombs = bombs_of(game_state)
    blocked = {pos for pos, _t in bombs} | other_tiles(game_state)
    if extra_bomb is not None:
        bombs = bombs + [(extra_bomb, BOMB_TIMER)]
    crosses = [(t, blast_cross(bx, by, field)) for (bx, by), t in bombs]

    def lethal(tile, depth):
        # A live explosion kills in this step only. It is gone one step later.
        if depth == 1 and exp_map[tile[0], tile[1]] >= 1:
            return True
        return any(tile in cross for t, cross in crosses if depth in (t + 1, t + 2))

    def safe_for_ever(tile, depth):
        # No bomb can hit this tile at this depth or at any later depth.
        return not any(tile in cross for t, cross in crosses if depth <= t + 2)

    if lethal(start, start_depth):
        return False

    seen = {start}
    queue = deque([(start, start_depth)])
    while queue:
        tile, depth = queue.popleft()
        if safe_for_ever(tile, depth):
            return True
        if depth >= BFS_HORIZON:
            continue
        for dx, dy in MOVES:
            nx, ny = tile[0] + dx, tile[1] + dy
            step = (nx, ny)
            if step in seen or not (0 <= nx < width and 0 <= ny < height):
                continue
            if field[nx, ny] != 0 or step in blocked or lethal(step, depth + 1):
                continue
            seen.add(step)
            queue.append((step, depth + 1))
    return False


def action_masks(game_state):
    """
    Give two boolean arrays with 6 entries each.

    `legal` marks the actions that the engine accepts. A move is legal when
    the target tile is free, has no bomb and has no opponent. WAIT is always
    legal. BOMB is legal when the agent still has a bomb.

    `safe` marks the legal actions after which an escape path still exists.
    The agent selects its action from `safe`. `safe` is also the bootstrap
    mask of section 4c of the design: the planner rejects a start tile that
    is lethal in the next update, so `safe` holds no action that kills the
    agent at once.
    """
    field = game_state["field"]
    width, height = field.shape
    _name, _score, bombs_left, (x, y) = game_state["self"]
    bomb_tiles = {pos for pos, _t in bombs_of(game_state)}
    opponents = other_tiles(game_state)

    legal = np.zeros(N_ACTIONS, dtype=bool)
    for action, (dx, dy) in enumerate(MOVES):
        nx, ny = x + dx, y + dy
        legal[action] = (
            0 <= nx < width
            and 0 <= ny < height
            and field[nx, ny] == 0
            and (nx, ny) not in bomb_tiles
            and (nx, ny) not in opponents
        )
    legal[WAIT] = True
    legal[BOMB] = bool(bombs_left)

    # No bomb and no fire: each legal move is safe. Only BOMB needs the planner.
    quiet = not bomb_tiles and not (game_state["explosion_map"] >= 1).any()

    safe = np.zeros(N_ACTIONS, dtype=bool)
    for action in range(N_ACTIONS):
        if not legal[action]:
            continue
        if action == BOMB:
            safe[action] = _escape_exists(game_state, (x, y), 1, extra_bomb=(x, y))
        elif quiet:
            safe[action] = True
        elif action == WAIT:
            safe[action] = _escape_exists(game_state, (x, y), 1)
        else:
            dx, dy = MOVES[action]
            safe[action] = _escape_exists(game_state, (x + dx, y + dy), 1)
    return legal, safe


def least_bad_action(game_state, legal):
    """
    Give the legal action that stays out of the fire for the longest time.

    The agent uses this action only when no action keeps an escape path
    open. BOMB is never used here, because a new bomb makes the state worse.
    """
    ticks = danger_ticks(game_state)
    x, y = game_state["self"][3]
    best, best_ticks = WAIT, -1
    for action in (UP, RIGHT, DOWN, LEFT, WAIT):
        if not legal[action]:
            continue
        if action == WAIT:
            tx, ty = x, y
        else:
            tx, ty = x + MOVES[action][0], y + MOVES[action][1]
        if ticks[tx, ty] > best_ticks:
            best, best_ticks = action, int(ticks[tx, ty])
    return best
