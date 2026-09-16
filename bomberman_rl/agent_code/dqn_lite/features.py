"""Feature extraction for the DQN agent.

One input is produced per game state: ``vec``, 42 hand-built floats. These are
four 8-feature direction blocks (UP, RIGHT, DOWN, LEFT) plus a 10-feature
global block.

A 6-element action mask is returned alongside. It is the intersection of the
*legal* actions (the environment would reject walking into a wall/crate/bomb/
agent, or bombing without a bomb available) and the *safe* ones (the escape
search sees a survivable continuation). If that intersection is empty the mask
degrades to the legal actions alone, because a mask with no action left would
make both the greedy argmax and the exploration weights meaningless.

Timing conventions, derived from ``environment.do_step``:
  the state handed to ``act`` shows the agent at *offset 0*; after the agent's
  action it is at offset 1, and kills during that step are evaluated against
  offset-1 positions.  A bomb whose state timer is ``t`` therefore kills on
  offsets ``t + 1`` and ``t + 2`` (the explosion stays dangerous for one extra
  evaluation).  ``explosion_map > 0`` means "deadly on offset 1".
"""

import numpy as np

import settings as s

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
# clockwise, so a 90 degree rotation is a cyclic shift of indices 0..3
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

N_DIR_FEATURES = 8
N_GLOBAL_FEATURES = 10
VEC_SIZE = 4 * N_DIR_FEATURES + N_GLOBAL_FEATURES  # 42

# A bomb dropped right now behaves like a state timer of BOMB_TIMER, so the
# latest offset we ever have to reason about is BOMB_TIMER + EXPLOSION_TIMER.
HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1  # 7

# indices into vec
IDX_FREE = 0          # within a direction block
IDX_CRATE = 1
IDX_DANGER = 2
IDX_ESCAPE = 3
IDX_COIN_DIR = 4
IDX_CRATE_DIR = 5
IDX_ENEMY_DIR = 6
IDX_DEAD_END = 7

GLOBAL_OFFSET = 4 * N_DIR_FEATURES  # 32
IDX_BOMB_AVAILABLE = GLOBAL_OFFSET + 0
IDX_DANGER_HERE = GLOBAL_OFFSET + 1
IDX_WAIT_SAFE = GLOBAL_OFFSET + 2
IDX_BOMB_HERE_SAFE = GLOBAL_OFFSET + 3
IDX_CRATES_IN_BLAST = GLOBAL_OFFSET + 4
IDX_ENEMY_IN_BLAST = GLOBAL_OFFSET + 5
IDX_COIN_DIST = GLOBAL_OFFSET + 6
IDX_CRATE_DIST = GLOBAL_OFFSET + 7
IDX_ENEMY_DIST = GLOBAL_OFFSET + 8
IDX_STEP = GLOBAL_OFFSET + 9

MAX_DIST = 20.0  # distances are clipped and normalised by this


def blast_coords(x, y, field):
    """Tiles hit by a bomb at (x, y). Stops at walls, does not turn corners."""
    coords = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def danger_bitmap(field, bombs, explosion_map):
    """Per-tile bitmask of deadly offsets: bit k set => deadly at offset k."""
    bits = np.zeros(field.shape, dtype=np.int32)
    bits[explosion_map > 0] |= 1 << 1
    for (bx, by), timer in bombs:
        mask = (1 << (timer + 1)) | (1 << (timer + 2))
        for (x, y) in blast_coords(bx, by, field):
            bits[x, y] |= mask
    return bits


def danger_scalar(bits_at_tile):
    """Earliest deadly offset -> urgency in (0, 1]; 0 means never deadly."""
    for k in range(1, HORIZON + 1):
        if bits_at_tile & (1 << k):
            return (HORIZON - k) / HORIZON
    return 0.0


def _survives(start, passable, bits, origin):
    """Is there an action sequence from `start` at offset 1 that stays alive?

    Explores the time-expanded graph level by level. `origin` is the agent's
    current tile: standing still on it is legal even when a bomb sits there,
    but it can never be re-entered once left (bombs block movement).
    """
    frontier = {start}
    for t in range(2, HORIZON + 1):
        nxt = set()
        for (x, y) in frontier:
            for nx, ny in ((x, y), (x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
                if not (passable[nx, ny] or ((nx, ny) == (x, y) == origin)):
                    continue
                if bits[nx, ny] & (1 << t):
                    continue
                nxt.add((nx, ny))
        frontier = nxt
        if not frontier:
            return False
    return True


def _escape_from(pos, passable, bits, blocked_at_1, origin):
    """Core search: which of UP/RIGHT/DOWN/LEFT/WAIT can be survived?

    Knows nothing about opponents; whatever pessimism is wanted has already
    been baked into `passable` (permanent obstacles) or `blocked_at_1`
    (obstacles for the first step only).
    """
    x, y = pos
    safe = set()
    candidates = [(i, (x + dx, y + dy)) for i, (dx, dy) in enumerate(DELTAS)]
    candidates.append((4, (x, y)))  # WAIT
    for action, p1 in candidates:
        if not (passable[p1] or p1 == origin):
            continue
        if p1 in blocked_at_1:
            continue
        if bits[p1] & (1 << 1):
            continue
        if _survives(p1, passable, bits, origin):
            safe.add(action)
    return safe


def escape_actions(pos, passable, bits, other_positions, origin, allow_fallback=True):
    """Which of UP/RIGHT/DOWN/LEFT/WAIT can be followed by survival?

    Pessimistic first: opponents are permanent obstacles for the whole search.
    Assuming they step aside is what used to get the agent killed. A
    rule_based_agent parked in the only corridor does not move, and the escape
    route the features promised never opens.

    Only if the pessimistic search finds nothing at all do we fall back to the
    optimistic one, where opponents block just offset 1. "Trapped unless they
    move" is still better served by the moves that survive if they do move than
    by an arbitrary choice.  Callers that are deciding whether to *create* new
    danger (dropping a bomb) pass ``allow_fallback=False``: there is never a
    reason to gamble on an opponent stepping aside for a bomb of one's own.
    """
    if not other_positions:
        return _escape_from(pos, passable, bits, (), origin)

    pessimistic = passable.copy()
    for p in other_positions:
        pessimistic[p] = False
    safe = _escape_from(pos, pessimistic, bits, (), origin)
    if safe or not allow_fallback:
        return safe
    return _escape_from(pos, passable, bits, other_positions, origin)


def distance_field(targets, passable):
    """Multi-source BFS: distance from every tile to the nearest target.

    Unreachable tiles keep -1. Computing the whole field rather than a single
    path matters for symmetry: picking *one* shortest direction would depend on
    the order neighbours happen to be expanded in, which does not survive a
    rotation of the board.
    """
    dist = np.full(passable.shape, -1, dtype=np.int32)
    frontier = [t for t in targets if passable[t]]
    for t in frontier:
        dist[t] = 0
    d = 0
    while frontier:
        d += 1
        nxt = []
        for (x, y) in frontier:
            for dx, dy in DELTAS:
                q = (x + dx, y + dy)
                if passable[q] and dist[q] < 0:
                    dist[q] = d
                    nxt.append(q)
        frontier = nxt
    return dist


def directions_towards(start, dist):
    """Every first step that lies on a shortest path, plus the distance.

    Returns ``(set_of_direction_indices, distance)``; distance is ``None`` when
    no target is reachable and 0 when the agent already stands on one.
    """
    d0 = int(dist[start])
    if d0 < 0:
        return set(), None
    if d0 == 0:
        return set(), 0
    best = {i for i, (dx, dy) in enumerate(DELTAS)
            if dist[start[0] + dx, start[1] + dy] == d0 - 1}
    return best, d0


def _norm_dist(dist):
    if dist is None:
        return 1.0
    return min(dist, MAX_DIST) / MAX_DIST


def state_to_features(game_state):
    """Return ``(vec, mask)`` for a game state, or ``None`` if there is none."""
    if game_state is None:
        return None

    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    bombs = game_state["bombs"]
    coins = game_state["coins"]
    _, _, bomb_available, own_pos = game_state["self"]
    others = game_state["others"]
    other_positions = {o[3] for o in others}
    bomb_positions = {pos for pos, _ in bombs}

    bits = danger_bitmap(field, bombs, explosion_map)

    # walkable *now*: free tile, no bomb, no other agent
    passable = field == 0
    for (x, y) in bomb_positions:
        passable[x, y] = False
    movable = passable.copy()  # what the environment would let us walk into
    for (x, y) in other_positions:
        movable[x, y] = False

    vec = np.zeros(VEC_SIZE, dtype=np.float32)

    safe_actions = escape_actions(own_pos, passable, bits, other_positions, own_pos)

    coin_targets = set(coins)
    crate_targets = set()
    enemy_targets = set()
    cols, rows = field.shape
    for x in range(1, cols - 1):
        for y in range(1, rows - 1):
            if not passable[x, y]:
                continue
            for dx, dy in DELTAS:
                nb = (x + dx, y + dy)
                if field[nb] == 1:
                    crate_targets.add((x, y))
                elif nb in other_positions:
                    enemy_targets.add((x, y))

    # the agent's own tile is walkable for path purposes even if it sits on a
    # bomb it just dropped: it can still walk out of it
    reachable = movable.copy()
    reachable[own_pos] = True
    coin_dirs, coin_dist = directions_towards(own_pos, distance_field(coin_targets, reachable))
    crate_dirs, crate_dist = directions_towards(own_pos, distance_field(crate_targets, reachable))
    enemy_dirs, enemy_dist = directions_towards(own_pos, distance_field(enemy_targets, reachable))

    ox, oy = own_pos
    for i, (dx, dy) in enumerate(DELTAS):
        nb = (ox + dx, oy + dy)
        base = i * N_DIR_FEATURES
        free = bool(movable[nb])
        vec[base + IDX_FREE] = float(free)
        vec[base + IDX_CRATE] = float(field[nb] == 1)
        vec[base + IDX_DANGER] = danger_scalar(int(bits[nb]))
        vec[base + IDX_ESCAPE] = float(i in safe_actions)
        vec[base + IDX_COIN_DIR] = float(i in coin_dirs)
        vec[base + IDX_CRATE_DIR] = float(i in crate_dirs)
        vec[base + IDX_ENEMY_DIR] = float(i in enemy_dirs)
        if free:
            # a true dead end has exactly one exit: the tile we came from
            exits = sum(1 for ddx, ddy in DELTAS
                        if movable[nb[0] + ddx, nb[1] + ddy])
            vec[base + IDX_DEAD_END] = float(exits <= 1)

    own_blast = blast_coords(ox, oy, field)
    crates_in_blast = sum(1 for p in own_blast if field[p] == 1)
    enemy_in_blast = any(p in other_positions for p in own_blast)

    # would dropping a bomb here still leave an escape route?
    hypothetical = list(bombs) + [((ox, oy), s.BOMB_TIMER)]
    bits_after = danger_bitmap(field, hypothetical, explosion_map)
    passable_after = passable.copy()
    passable_after[ox, oy] = False
    # no fallback here: a bomb we have not dropped yet is never worth betting
    # an opponent will vacate the only corridor
    bomb_here_safe = bool(escape_actions(own_pos, passable_after, bits_after,
                                         other_positions, own_pos,
                                         allow_fallback=False))

    vec[IDX_BOMB_AVAILABLE] = float(bomb_available)
    vec[IDX_DANGER_HERE] = danger_scalar(int(bits[ox, oy]))
    vec[IDX_WAIT_SAFE] = float(4 in safe_actions)
    vec[IDX_BOMB_HERE_SAFE] = float(bomb_here_safe)
    vec[IDX_CRATES_IN_BLAST] = min(crates_in_blast, 4) / 4.0
    vec[IDX_ENEMY_IN_BLAST] = float(enemy_in_blast)
    vec[IDX_COIN_DIST] = _norm_dist(coin_dist)
    vec[IDX_CRATE_DIST] = _norm_dist(crate_dist)
    vec[IDX_ENEMY_DIST] = _norm_dist(enemy_dist)
    vec[IDX_STEP] = min(game_state["step"], s.MAX_STEPS) / s.MAX_STEPS

    # derived from the finished vec, so the mask can never disagree with what
    # action_mask reconstructs for the same state later on
    mask = action_mask(vec)
    return vec, mask


# column of each direction block that a mask reads, as a gather index
_DIR_BLOCKS = np.arange(4) * N_DIR_FEATURES
_FREE_COLS = _DIR_BLOCKS + IDX_FREE
_ESCAPE_COLS = _DIR_BLOCKS + IDX_ESCAPE


def action_mask(vecs):
    """Legal AND safe actions for one vec (42,) or a batch (N, 42): bool[..., 6].

    *Legal* is what the environment would not reject: a move needs its target
    tile free of walls, crates, bombs and other agents, WAIT always passes and
    BOMB needs a bomb in stock. *Safe* is what the escape search believes is
    survivable: the per-direction escape bits, IDX_WAIT_SAFE and
    IDX_BOMB_HERE_SAFE.

    Where nothing legal is also safe the agent is dead either way, and the mask
    degrades to the legal actions alone. An empty mask would make both the
    greedy argmax and the exploration weights meaningless.

    Pure function of the vec, so masks need not be stored in the replay buffer;
    it runs once per gradient step on a whole batch, hence no Python loop.
    """
    v = np.atleast_2d(np.asarray(vecs, dtype=np.float32))
    legal = np.ones((len(v), len(ACTIONS)), dtype=bool)
    legal[:, :4] = v[:, _FREE_COLS] > 0.5
    legal[:, 5] = v[:, IDX_BOMB_AVAILABLE] > 0.5

    safe = np.empty_like(legal)
    safe[:, :4] = v[:, _ESCAPE_COLS] > 0.5
    safe[:, 4] = v[:, IDX_WAIT_SAFE] > 0.5
    safe[:, 5] = v[:, IDX_BOMB_HERE_SAFE] > 0.5

    mask = legal & safe
    empty = ~mask.any(axis=1)
    mask[empty] = legal[empty]
    return mask[0] if np.ndim(vecs) == 1 else mask
