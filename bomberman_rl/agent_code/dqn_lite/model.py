"""Q-network and the D4 symmetry transform.

The board has the symmetry group of the square (4 rotations x mirror), and the
feature vector transforms under it by permuting its four direction blocks. That
makes symmetry augmentation essentially free, which is why the features were
laid out in blocks in the first place.
"""

import numpy as np
import torch
import torch.nn as nn

from .features import N_DIR_FEATURES, VEC_SIZE, ACTIONS

N_ACTIONS = len(ACTIONS)
HIDDEN = 128

# mirroring along the x axis swaps RIGHT and LEFT, leaves UP/DOWN/WAIT/BOMB
_FLIP_PERM = np.array([0, 3, 2, 1, 4, 5])


def action_permutation(k, flip):
    """Where each action index ends up after rotating k times, then flipping."""
    perm = np.arange(N_ACTIONS)
    perm[:4] = (np.arange(4) + k) % 4
    if flip:
        perm = _FLIP_PERM[perm]
    return perm


def _vec_gather_index(k, flip):
    """Index array so that ``new_vec = vec[..., index]``."""
    perm = action_permutation(k, flip)
    inverse = np.empty(4, dtype=int)
    for a in range(4):
        inverse[perm[a]] = a
    index = np.arange(VEC_SIZE)
    for block in range(4):
        src = inverse[block] * N_DIR_FEATURES
        dst = block * N_DIR_FEATURES
        index[dst:dst + N_DIR_FEATURES] = np.arange(src, src + N_DIR_FEATURES)
    return index


# all eight transforms, precomputed
TRANSFORMS = [(k, flip) for flip in (False, True) for k in range(4)]
_VEC_INDEX = {t: _vec_gather_index(*t) for t in TRANSFORMS}
_ACTION_PERM = {t: action_permutation(*t) for t in TRANSFORMS}


def transform_batch(vecs, actions, k, flip):
    """Apply one element of the symmetry group to a batch of feature vectors."""
    if k == 0 and not flip:
        return vecs, actions
    new_vecs = vecs[..., _VEC_INDEX[(k, flip)]]
    new_actions = _ACTION_PERM[(k, flip)][actions] if actions is not None else None
    return new_vecs, new_actions


class QNet(nn.Module):
    """Two-hidden-layer MLP over the 42 features."""

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(VEC_SIZE, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )

    def forward(self, vec):
        return self.head(vec)


def greedy_action(q_values, mask):
    """Best allowed action. ``q_values`` and ``mask`` are 1-D arrays."""
    q = np.asarray(q_values, dtype=np.float64).copy()
    q[~np.asarray(mask, dtype=bool)] = -np.inf
    if not np.isfinite(q).any():  # every action masked, should not happen
        return int(np.argmax(q_values))
    return int(np.argmax(q))
