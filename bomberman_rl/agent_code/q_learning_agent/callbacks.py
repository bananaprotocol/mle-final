import os
import pickle
import random
from collections import deque

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# Must match the tournament settings (settings.py).
BOMB_POWER = 3   # blast length per direction (tiles)
BOMB_TIMER = 4   # countdown of a freshly dropped bomb; the first state
                 # snapshot after dropping it shows 3

EPSILON_START = 0.25
EPSILON_MIN = 0.05
BFS_HORIZON = 6  # max moves to search for an escape


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.eps = EPSILON_START
    self.q = {}
    self.current_round = -1

    if os.path.isfile("q_table.pkl"):
        with open("q_table.pkl", "rb") as f:
            self.q = pickle.load(f)
        self.logger.info(f"Loaded Q-table with {len(self.q)} states.")
    else:
        self.logger.info("Starting with empty Q-table.")


# ---------------------------------------------------------------------------
# board geometry / danger model
# ---------------------------------------------------------------------------

def _in_bounds(arena, x, y):
    return 0 <= x < arena.shape[0] and 0 <= y < arena.shape[1]


def blast_cross(x, y, arena):
    """
    All tiles hit by a bomb at (x, y): power BOMB_POWER, stopped by stone
    walls (-1). Crates (1) do NOT stop the blast (see items.Bomb).
    """
    cross = {(x, y)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if not _in_bounds(arena, nx, ny) or arena[nx, ny] == -1:
                break
            cross.add((nx, ny))
    return cross


def _bombs_of(game_state):
    """Normalize game_state['bombs'] to [((x, y), timer), ...] with plain ints."""
    return [((int(px), int(py)), int(t)) for (px, py), t in game_state["bombs"]]


def _self_threatened(game_state):
    """True if my tile is inside some bomb's blast cross or a live explosion."""
    x, y = game_state["self"][3]
    if game_state["explosion_map"][x, y] >= 1:
        return True
    return _in_any_cross(game_state, (x, y))


def _in_any_cross(game_state, tile):
    arena = game_state["field"]
    for (px, py), _t in _bombs_of(game_state):
        if tile in blast_cross(px, py, arena):
            return True
    return False


def _escape_action(game_state, extra_bomb=False):
    """
    Safety layer: BFS over moves to a tile that is in no active bomb's blast
    cross, respecting when each bomb detonates.

    Timing model (verified against environment.py / items.py):
      - a bomb shown with timer t in my state (step s) explodes during world
        update s + t, i.e. AFTER the move I make in that step; the blast
        lingers (still dangerous) for one more update (s + t + 1).
      - my position after k moves (BFS depth k) is the position inspected by
        world update s + k.
    Hence a position at depth k is fatal iff it lies in the cross of a bomb
    with t in {k - 1, k - 2}, i.e. k in {t + 1, t + 2}. Passing through a
    cross before the bomb detonates is fine; after the blast has passed the
    tiles are harmless smoke (and covered by the explosion_map anyway).

    Returns the first move of the shortest safe path, or None if no escape
    within BFS_HORIZON. With extra_bomb=True a freshly dropped own bomb
    (timer BOMB_TIMER - 1 = 3, as the next snapshot will show it) is included.
    This decides whether dropping a bomb is legal at all.
    """
    arena = game_state["field"]
    exp_map = game_state["explosion_map"]
    x, y = game_state["self"][3]
    bombs = _bombs_of(game_state)
    if extra_bomb:
        bombs = bombs + [((x, y), BOMB_TIMER - 1)]

    crosses = [(t, blast_cross(px, py, arena)) for (px, py), t in bombs]
    bomb_tiles = {pos for pos, _t in bombs}
    other_tiles = {xy for (_, _, _, xy) in game_state["others"]}

    def blocked_now(tile):
        tx, ty = tile
        if not _in_bounds(arena, tx, ty):
            return True
        if arena[tx, ty] != 0:
            return True
        if tile in bomb_tiles or tile in other_tiles:
            return True
        if exp_map[tx, ty] >= 1:
            return True
        return False

    def safe_at_depth(tile, k):
        # position after k moves is inspected by world update s + k;
        # bomb with snapshot timer t is dangerous at updates s + t and s + t + 1
        return all(tile not in C for (t, C) in crosses if k in (t + 1, t + 2))

    start = (x, y)
    prev_node = {start: None}
    prev_action = {start: None}
    depth = {start: 0}
    queue = deque([start])

    while queue:
        tile = queue.popleft()
        k = depth[tile]
        # goal: a tile in no bomb's cross (safe once every bomb has passed)
        if all(tile not in C for _t, C in crosses):
            cur = tile
            while prev_node[cur] is not None:
                nxt = prev_node[cur]
                if prev_node[nxt] is None:  # nxt is the start tile
                    return prev_action[cur]
                cur = nxt
            return None
        if k >= BFS_HORIZON:
            continue
        for action, (dx, dy) in DIRS.items():
            ntile = (tile[0] + dx, tile[1] + dy)
            if ntile in depth or blocked_now(ntile) or not safe_at_depth(ntile, k + 1):
                continue
            depth[ntile] = k + 1
            prev_node[ntile] = tile
            prev_action[ntile] = action
            queue.append(ntile)
    return None


def _least_bad_action(game_state, valid):
    """
    Fallback when no full escape exists: the legal move (or WAIT) that keeps
    me out of danger the longest.
    """
    arena = game_state["field"]
    exp_map = game_state["explosion_map"]
    x, y = game_state["self"][3]
    crosses = [(t, blast_cross(px, py, arena)) for (px, py), t in _bombs_of(game_state)]

    def time_to_safety(tile):
        tx, ty = tile
        if exp_map[tx, ty] >= 1:
            return 0
        t_min = 9
        for t, C in crosses:
            if tile in C:
                t_min = min(t_min, t)
        return t_min

    best, best_t = None, -1
    for a in valid:
        if a == "BOMB":
            continue
        tile = (x, y) if a == "WAIT" else (x + DIRS[a][0], y + DIRS[a][1])
        t = time_to_safety(tile)
        if t > best_t:
            best, best_t = a, t
    return best if best is not None else "WAIT"


# ---------------------------------------------------------------------------
# state representation
# ---------------------------------------------------------------------------

def _nearest_offset(targets, x, y):
    if not targets:
        return (9, 9)
    tx, ty = min(targets, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
    return (max(-4, min(4, tx - x)), max(-4, min(4, ty - y)))


def state_key(game_state):
    """
    Ego-centric, position-invariant state so similar local situations map to
    the same key across rounds.

    Danger features (the part that makes survival learnable):
      threat     ticks until some bomb's blast hits my CURRENT tile (5: none)
      danger_adj how many of my 4 neighbours are in a blast cross or live
                 explosion (0..4)
    """
    arena = game_state["field"]
    _, _, bombs_left, (x, y) = game_state["self"]
    coins = [(int(cx), int(cy)) for cx, cy in game_state["coins"]]
    enemies = [(int(ex), int(ey)) for (_, _, _, (ex, ey)) in game_state["others"]]
    bombs = _bombs_of(game_state)
    exp_map = game_state["explosion_map"]

    cdx, cdy = _nearest_offset(coins, x, y)
    edx, edy = _nearest_offset(enemies, x, y)

    crosses = [(t, blast_cross(px, py, arena)) for (px, py), t in bombs]

    threat = 5
    for (t, C) in crosses:
        if (x, y) in C:
            threat = min(threat, t)

    danger_adj = 0
    for dx, dy in DIRS.values():
        nx, ny = x + dx, y + dy
        if _in_bounds(arena, nx, ny) and (
            exp_map[nx, ny] >= 1 or any((nx, ny) in C for _t, C in crosses)
        ):
            danger_adj += 1

    def _dist_free(dx, dy, max_d=4):
        d = 0
        for i in range(1, max_d + 1):
            nx, ny = x + dx * i, y + dy * i
            if not _in_bounds(arena, nx, ny) or arena[nx, ny] != 0:
                break
            d += 1
        return d

    def _is_crate(dx, dy):
        nx, ny = x + dx, y + dy
        if not _in_bounds(arena, nx, ny):
            return 0
        return 1 if arena[nx, ny] == 1 else 0

    dist_up, dist_right = _dist_free(0, -1), _dist_free(1, 0)
    dist_down, dist_left = _dist_free(0, 1), _dist_free(-1, 0)
    crate_up, crate_right = _is_crate(0, -1), _is_crate(1, 0)
    crate_down, crate_left = _is_crate(0, 1), _is_crate(-1, 0)

    return (
        cdx, cdy,
        edx, edy,
        dist_up, dist_right, dist_down, dist_left,
        crate_up, crate_right, crate_down, crate_left,
        1 if bombs_left else 0,
        threat,
        danger_adj,
        min(len(bombs), 3),
        min(8, int(game_state["step"]) // 50 + 1),  # round progress
        min(len(coins), 3),
    )


# ---------------------------------------------------------------------------
# action selection
# ---------------------------------------------------------------------------

def valid_actions(game_state):
    """All legal action strings (adapted from rule_based_agent)."""
    arena = game_state["field"]
    _, _, bombs_left, (x, y) = game_state["self"]
    bomb_xys = {xy for xy, _ in game_state["bombs"]}
    others_xy = [xy for (n, s, b, xy) in game_state["others"]]
    explosion_map = game_state["explosion_map"]

    def safe(i, j):
        if not _in_bounds(arena, i, j):
            return False
        return (
            arena[i, j] == 0
            and explosion_map[i, j] < 1
            and (i, j) not in bomb_xys
            and (i, j) not in others_xy
        )

    valid = []
    if safe(x - 1, y):
        valid.append("LEFT")
    if safe(x + 1, y):
        valid.append("RIGHT")
    if safe(x, y - 1):
        valid.append("UP")
    if safe(x, y + 1):
        valid.append("DOWN")
    if safe(x, y):
        valid.append("WAIT")
    if bombs_left:
        valid.append("BOMB")
    return valid


def act(self, game_state) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if game_state is None:
        return "WAIT"
    x, y = game_state["self"][3]

    # New round: reset bookkeeping that must not leak across rounds.
    rnd = game_state.get("round", 0)
    if rnd != self.current_round:
        self.current_round = rnd

    if self.eps > EPSILON_MIN:
        self.eps = max(EPSILON_MIN, self.eps * 0.999)

    valid = valid_actions(game_state)
    if not valid:
        return "WAIT"

    # --- safety layer -------------------------------------------------
    # My tile is in some bomb's blast cross (or a live explosion): take the
    # planned escape move. Overrides exploration, survival first.
    if _self_threatened(game_state):
        plan = _escape_action(game_state)
        self.logger.info(f"[dbg] threatened at ({x},{y}) bombs={game_state['bombs']} plan={plan}")
        if plan is not None:
            return plan
        return _least_bad_action(game_state, valid)

    # --- learning layer ------------------------------------------------
    # A bomb shown with timer t <= 1 is too close to let the Q-layer step
    # into its cross (t == 0 detonates after this very move). Filter such
    # destinations out of every chosen action; WAIT is safe here because the
    # current tile is by definition not in any cross (else the planner ran).
    arena = game_state["field"]
    fatal = set()
    for (px, py), t in _bombs_of(game_state):
        if t <= 1:
            fatal |= blast_cross(px, py, arena)

    def _dest(a):
        dx, dy = DIRS[a]
        return (x + dx, y + dy)

    valid = [a for a in valid if a in ("WAIT", "BOMB") or _dest(a) not in fatal]
    if not valid:  # defensive: cannot actually happen, see above
        return "WAIT"

    # Never offer BOMB unless there is a provably safe escape from the blast.
    if "BOMB" in valid and _escape_action(game_state, extra_bomb=True) is None:
        valid = [a for a in valid if a != "BOMB"]

    s = state_key(game_state)
    self.q.setdefault(s, np.zeros(len(ACTIONS)))

    if random.random() < self.eps:
        action = random.choice(valid)
    else:
        q_row = self.q[s]
        q_values = q_row[[ACTIONS.index(a) for a in valid]]
        m = q_values.max()
        ties = [a for a, v in zip(valid, q_values) if v >= m - 1e-9]
        if m <= 1e-9:
            # all (near) zero: nothing learned here yet, so don't gamble on a bomb
            ties = [a for a in ties if a != "BOMB"] or ties
        action = random.choice(ties)

    return action
