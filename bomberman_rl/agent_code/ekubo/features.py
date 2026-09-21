import os

import numpy as np

import settings as s

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # clockwise, so a rotation shifts 0..3

N_DIR_FEATURES = 8
N_GLOBAL_FEATURES = 10
N_ROBUST_FEATURES = 7
VEC_SIZE = 4 * N_DIR_FEATURES + N_GLOBAL_FEATURES + N_ROBUST_FEATURES  # 49

CROP_RADIUS = s.BOMB_POWER  # 3
CROP_SIZE = 2 * CROP_RADIUS + 1  # 7
CROP_CHANNELS = 6
CROP_SHAPE = (CROP_CHANNELS, CROP_SIZE, CROP_SIZE)

# a new bomb acts like state timer BOMB_TIMER, so this is the last offset to check
HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1  # 7


def _env_int(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# steps an opponent may walk before it bombs; 0 means it bombs where it
# stands, -1 switches the whole robust layer off (how we compared it to v1)
THREAT_STEPS = _env_int("DQN_THREAT_STEPS", 1)

# how far an opponent may walk into our escape route; 0 stops parking only.
# 3 halves the suicides of 2, and 4 is too careful to score (sweep 2026-09-21)
PARK_STEPS = _env_int("DQN_PARK_STEPS", 3)

# off by default: bombs carry no owner, so an opponent whose bomb is about to
# come back looks unarmed until the step it bombs us
THREAT_ARMED_ONLY = bool(_env_int("DQN_THREAT_ARMED_ONLY", 0))

# "union" survives all candidates at once, which two opponents bombing both
# ends of one corridor need. "each" takes them one by one: milder, but slower.
THREAT_MODE = os.environ.get("DQN_THREAT_MODE", "union")

# derived from the knobs, kept in sync by apply_knobs
THREAT_RANGE = PARK_RANGE = ROBUST_HORIZON = 0


def apply_knobs(threat_steps=None, park_steps=None, threat_mode=None,
                armed_only=None):
    """Set the knobs and everything derived from them.

    The sweep and the tests override the env defaults, and the ranges have to
    follow.
    """
    global THREAT_STEPS, PARK_STEPS, THREAT_MODE, THREAT_ARMED_ONLY
    global THREAT_RANGE, PARK_RANGE, ROBUST_HORIZON
    if threat_steps is not None:
        THREAT_STEPS = int(threat_steps)
    if park_steps is not None:
        PARK_STEPS = int(park_steps)
    if threat_mode is not None:
        THREAT_MODE = threat_mode
    if armed_only is not None:
        THREAT_ARMED_ONLY = bool(armed_only)
    if THREAT_MODE not in ("each", "union"):
        THREAT_MODE = "union"
    # 2 * BOMB_POWER is the widest gap a blast can bridge, so anything further
    # away cannot reach us, even if it walks closer first
    THREAT_RANGE = 2 * s.BOMB_POWER + THREAT_STEPS
    PARK_RANGE = 2 * s.BOMB_POWER + max(PARK_STEPS, 0)
    # a late candidate is deadly until BOMB_TIMER + THREAT_STEPS + 2, and
    # parked opponents stay where they are
    ROBUST_HORIZON = HORIZON + max(THREAT_STEPS, PARK_STEPS, 0)


apply_knobs()

# indices into vec; model.pt is trained for this layout, so do not reorder
IDX_FREE = 0          # inside a direction block
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

ROBUST_OFFSET = GLOBAL_OFFSET + N_GLOBAL_FEATURES  # 42
IDX_ROBUST_DIR = ROBUST_OFFSET + 0      # + i for UP, RIGHT, DOWN, LEFT
IDX_WAIT_ROBUST = ROBUST_OFFSET + 4
IDX_BOMB_HERE_ROBUST = ROBUST_OFFSET + 5
IDX_THREAT_NEAR = ROBUST_OFFSET + 6

MAX_DIST = 20.0  # distances are clipped and divided by this


def blast_coords(x, y, field):
    """Tiles a bomb at (x, y) hits. Walls stop it, corners are not turned."""
    coords = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def danger_bitmap(field, bombs, explosion_map):
    """Deadly offsets per tile: bit k set means deadly at offset k."""
    bits = np.zeros(field.shape, dtype=np.int32)
    bits[explosion_map > 0] |= 1 << 1
    for (bx, by), timer in bombs:
        mask = (1 << (timer + 1)) | (1 << (timer + 2))
        for (x, y) in blast_coords(bx, by, field):
            bits[x, y] |= mask
    return bits


def danger_scalar(bits_at_tile):
    """First deadly offset as urgency in (0, 1]. 0 means never deadly."""
    for k in range(1, HORIZON + 1):
        if bits_at_tile & (1 << k):
            return (HORIZON - k) / HORIZON
    return 0.0


def _survives(start, passable, bits, origin, horizon=HORIZON):
    """Can we stay alive from `start` at offset 1?

    `origin` is our tile now. We may stand on it with a bomb below us, but we
    cannot come back once we leave. A tile with no deadly offset left ends the
    search: we can wait there for good.
    """
    if not int(bits[start]) >> 2:
        return True
    frontier = {start}
    for t in range(2, horizon + 1):
        nxt = set()
        for (x, y) in frontier:
            for nx, ny in ((x, y), (x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
                if not (passable[nx, ny] or ((nx, ny) == (x, y) == origin)):
                    continue
                tile_bits = int(bits[nx, ny]) >> t
                if tile_bits & 1:
                    continue
                if not tile_bits:
                    return True
                nxt.add((nx, ny))
        frontier = nxt
        if not frontier:
            return False
    return True


def _escape_from(pos, passable, bits, blocked_at_1, origin):
    """Which of UP/RIGHT/DOWN/LEFT/WAIT survive?

    Opponents belong in `passable` (permanent walls) or in `blocked_at_1`
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
    """Escape set, opponents first as walls.

    If nothing survives we retry with them blocking offset 1 only.
    allow_fallback=False refuses that bet.
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


def threat_candidates(own_pos, passable, others):
    """Set of (tile, delay): where an opponent could bomb, after how many steps.

    We add the danger a candidate would create, not the opponent standing
    there, so its tile stays walkable for us.
    """
    if THREAT_STEPS < 0 or not others:
        return set()

    ox, oy = own_pos
    positions = {o[3] for o in others}
    candidates = set()
    for _name, _score, bombs_left, pos in others:
        if THREAT_ARMED_ONLY and not bombs_left:
            continue
        if abs(pos[0] - ox) + abs(pos[1] - oy) > THREAT_RANGE:
            continue
        blocked = (positions - {pos}) | {own_pos}
        candidates.add((pos, 0))
        frontier = {pos}
        seen = {pos}
        for delay in range(1, THREAT_STEPS + 1):
            nxt = set()
            for (x, y) in frontier:
                for dx, dy in DELTAS:
                    q = (x + dx, y + dy)
                    if q in seen or q in blocked or not passable[q]:
                        continue
                    seen.add(q)
                    nxt.add(q)
                    candidates.add((q, delay))
            frontier = nxt
            if not frontier:
                break
    return candidates


def parking_bits(own_pos, passable, others, shape):
    """(bits, any): where an opponent could block us, from which offset on.

    Same encoding as danger_bitmap, so _survives needs one kind of obstacle
    only: taken and deadly forbid the same thing.
    """
    if THREAT_STEPS < 0 or PARK_STEPS <= 0 or not others:
        return None, False

    ox, oy = own_pos
    positions = {o[3] for o in others}
    bits = np.zeros(shape, dtype=np.int32)
    parked = False
    for _name, _score, _bombs_left, pos in others:
        if abs(pos[0] - ox) + abs(pos[1] - oy) > PARK_RANGE:
            continue
        blocked = (positions - {pos}) | {own_pos}
        frontier = {pos}
        seen = {pos}
        for depth in range(1, PARK_STEPS + 1):
            # every offset from the arrival on, as it can stop and stay
            mask = (1 << (ROBUST_HORIZON + 1)) - (1 << depth)
            nxt = set()
            for (x, y) in frontier:
                for dx, dy in DELTAS:
                    q = (x + dx, y + dy)
                    if q in seen or q in blocked or not passable[q]:
                        continue
                    seen.add(q)
                    nxt.add(q)
                    bits[q] |= mask
                    parked = True
            frontier = nxt
            if not frontier:
                break
    return (bits, True) if parked else (None, False)


def candidate_blasts(candidates, field):
    """(blast tiles, deadly-offset mask) per candidate bomb.

    Sorted so the early exit in robust_escape_actions does not depend on set
    order.
    """
    coords_cache = {}
    blasts = []
    for tile, delay in sorted(candidates):
        coords = coords_cache.get(tile)
        if coords is None:
            coords = coords_cache[tile] = blast_coords(tile[0], tile[1], field)
        mask = (1 << (s.BOMB_TIMER + delay + 1)) | (1 << (s.BOMB_TIMER + delay + 2))
        blasts.append((coords, mask))
    return blasts


def _bits_with(bits, blasts):
    """Copy of `bits` with the deadly offsets of every blast merged in."""
    out = bits.copy()
    for coords, mask in blasts:
        for tile in coords:
            out[tile] |= mask
    return out


def robust_escape_actions(pos, passable, bits, other_positions, origin, blasts,
                          base):
    """Subset of `base` that also survives opponents who walk and bomb.

    `bits` already holds the parking bits, the candidates in `blasts` go on
    top. There is no optimistic fallback: if nothing survives, mask_from_vec
    drops to the merely safe rung.
    """
    if not base:
        return set()

    pessimistic = passable.copy()
    for p in other_positions:
        pessimistic[p] = False

    if THREAT_MODE == "each" and blasts:
        variants = [_bits_with(bits, [blast]) for blast in blasts]
    else:
        variants = [_bits_with(bits, blasts)]

    x, y = pos
    targets = [(x + dx, y + dy) for dx, dy in DELTAS] + [(x, y)]
    robust = set()
    for action in base:
        p1 = targets[action]
        # parking bits reach down to offset 1, candidate blasts never do
        if bits[p1] & (1 << 1):
            continue
        if all(_survives(p1, pessimistic, v, origin, ROBUST_HORIZON)
               for v in variants):
            robust.add(action)
    return robust


def distance_field(targets, passable):
    """Multi-source BFS to the nearest target. Unreachable tiles keep -1."""
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
    """(directions on a shortest path, distance). None if nothing is reachable."""
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
    """(vec, crop, mask) for a game state, or None if there is none."""
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

    # walkable now: free tile, no bomb, no other agent
    passable = field == 0
    for (x, y) in bomb_positions:
        passable[x, y] = False
    movable = passable.copy()  # what the environment lets us walk into
    for (x, y) in other_positions:
        movable[x, y] = False

    vec = np.zeros(VEC_SIZE, dtype=np.float32)

    safe_actions = escape_actions(own_pos, passable, bits, other_positions, own_pos)

    # both stay empty in the solo scenarios, so those cost what they did before
    park, parked = parking_bits(own_pos, passable, others, field.shape)
    blasts = candidate_blasts(threat_candidates(own_pos, passable, others), field)
    threatened = parked or bool(blasts)
    if threatened:
        robust_actions = robust_escape_actions(
            own_pos, passable, (bits | park) if parked else bits,
            other_positions, own_pos, blasts, safe_actions)
    else:
        robust_actions = set(safe_actions)

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

    # our tile stays walkable for pathing, even with our bomb on it
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
            # a dead end has one exit: the tile we came from
            exits = sum(1 for ddx, ddy in DELTAS
                        if movable[nb[0] + ddx, nb[1] + ddy])
            vec[base + IDX_DEAD_END] = float(exits <= 1)
        vec[IDX_ROBUST_DIR + i] = float(i in robust_actions)

    own_blast = blast_coords(ox, oy, field)
    crates_in_blast = sum(1 for p in own_blast if field[p] == 1)
    enemy_in_blast = any(p in other_positions for p in own_blast)

    # Does a bomb here still leave us a way out? Ask about WAIT: BOMB is this
    # step's action, so we still stand here at offset 1.
    hypothetical = list(bombs) + [((ox, oy), s.BOMB_TIMER)]
    bits_after = danger_bitmap(field, hypothetical, explosion_map)
    passable_after = passable.copy()
    passable_after[ox, oy] = False
    bomb_here_safe = 4 in escape_actions(own_pos, passable_after, bits_after,
                                         other_positions, own_pos,
                                         allow_fallback=False)
    bomb_here_robust = bomb_here_safe and (
        not threatened
        or 4 in robust_escape_actions(
            own_pos, passable_after,
            (bits_after | park) if parked else bits_after,
            other_positions, own_pos, blasts, (4,)))

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

    vec[IDX_WAIT_ROBUST] = float(4 in robust_actions)
    vec[IDX_BOMB_HERE_ROBUST] = float(bomb_here_robust)
    vec[IDX_THREAT_NEAR] = float(threatened)

    mask = mask_from_vec(vec)
    crop = _make_crop(field, coins, bombs, bits, other_positions, own_pos)
    return vec, crop, mask


def _make_crop(field, coins, bombs, bits, other_positions, own_pos):
    """6x7x7 view around the agent. Everything off the board is wall."""
    cols, rows = field.shape
    pad = CROP_RADIUS
    planes = np.zeros((CROP_CHANNELS, cols + 2 * pad, rows + 2 * pad), dtype=np.float32)
    planes[0, :, :] = 1.0  # wall everywhere, then carve out the real board
    inner = (slice(pad, pad + cols), slice(pad, pad + rows))
    planes[0][inner] = (field == -1).astype(np.float32)
    planes[1][inner] = (field == 1).astype(np.float32)
    for (x, y) in coins:
        planes[2, x + pad, y + pad] = 1.0
    danger_plane = np.zeros_like(field, dtype=np.float32)
    for x, y in np.argwhere(bits):
        danger_plane[x, y] = danger_scalar(int(bits[x, y]))
    planes[3][inner] = danger_plane
    for (x, y) in other_positions:
        planes[4, x + pad, y + pad] = 1.0
    for (x, y), timer in bombs:
        planes[5, x + pad, y + pad] = (s.BOMB_TIMER - timer) / s.BOMB_TIMER

    cx, cy = own_pos[0] + pad, own_pos[1] + pad
    return planes[:, cx - pad:cx + pad + 1, cy - pad:cy + pad + 1].copy()


# the vec column each mask reads, per direction block
_DIR_BLOCKS = np.arange(4) * N_DIR_FEATURES
_FREE_COLS = _DIR_BLOCKS + IDX_FREE
_ESCAPE_COLS = _DIR_BLOCKS + IDX_ESCAPE
_ROBUST_COLS = ROBUST_OFFSET + np.arange(4)


def _rungs(vecs):
    """(legal, safe, robust) for a batch of vecs.

    legal is what the environment accepts: a free tile for a move, WAIT
    always, BOMB with one in stock. safe survives the bombs on the board,
    robust also survives every bomb an opponent could still drop.
    """
    legal = np.ones((len(vecs), len(ACTIONS)), dtype=bool)
    legal[:, :4] = vecs[:, _FREE_COLS] > 0.5
    legal[:, 5] = vecs[:, IDX_BOMB_AVAILABLE] > 0.5

    safe = np.empty_like(legal)
    safe[:, :4] = vecs[:, _ESCAPE_COLS] > 0.5
    safe[:, 4] = vecs[:, IDX_WAIT_SAFE] > 0.5
    safe[:, 5] = vecs[:, IDX_BOMB_HERE_SAFE] > 0.5

    robust = np.empty_like(legal)
    robust[:, :4] = vecs[:, _ROBUST_COLS] > 0.5
    robust[:, 4] = vecs[:, IDX_WAIT_ROBUST] > 0.5
    robust[:, 5] = vecs[:, IDX_BOMB_HERE_ROBUST] > 0.5
    return legal, safe, robust


def legal_action_mask(vec):
    """bool[6]: what the environment accepts."""
    return _rungs(np.asarray(vec)[None])[0][0]


def safe_action_mask(vec):
    """bool[6]: what survives the bombs on the board."""
    return _rungs(np.asarray(vec)[None])[1][0]


def robust_action_mask(vec):
    """bool[6]: what also survives every bomb an opponent could still drop."""
    return _rungs(np.asarray(vec)[None])[2][0]


def masks_from_vecs(vecs):
    """Action masks for a batch, as a ladder.

    legal & safe & robust first, then legal & safe, then legal alone. We drop
    a rung only when it is empty, as an empty mask would make the argmax and
    the exploration weights meaningless.
    """
    legal, safe, robust = _rungs(np.asarray(vecs))
    safe &= legal
    masks = safe & robust
    empty = ~masks.any(axis=1)
    masks[empty] = safe[empty]
    empty = ~masks.any(axis=1)
    masks[empty] = legal[empty]
    return masks


def mask_from_vec(vec):
    """masks_from_vecs for a single vec."""
    return masks_from_vecs(np.asarray(vec)[None])[0]
