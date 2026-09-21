import numpy as np
import torch
import torch.nn as nn

from .features import (ACTIONS, CROP_CHANNELS, CROP_SIZE, N_DIR_FEATURES,
                       ROBUST_OFFSET, VEC_SIZE)

N_ACTIONS = len(ACTIONS)
CONV_CHANNELS = 16
CROP_EMBED = 64
HIDDEN = 128

# a mirror along x swaps RIGHT and LEFT and keeps UP, DOWN, WAIT, BOMB
_FLIP_PERM = np.array([0, 3, 2, 1, 4, 5])


def action_permutation(k, flip):
    """Where each action ends up after k rotations and the flip."""
    perm = np.arange(N_ACTIONS)
    perm[:4] = (np.arange(4) + k) % 4
    if flip:
        perm = _FLIP_PERM[perm]
    return perm


def _vec_gather_index(k, flip):
    """Index array with new_vec = vec[..., index]."""
    perm = action_permutation(k, flip)
    inverse = np.empty(4, dtype=int)
    for a in range(4):
        inverse[perm[a]] = a
    index = np.arange(VEC_SIZE)
    for block in range(4):
        src = inverse[block] * N_DIR_FEATURES
        dst = block * N_DIR_FEATURES
        index[dst:dst + N_DIR_FEATURES] = np.arange(src, src + N_DIR_FEATURES)
        # one robust float per direction, permuted the same way
        index[ROBUST_OFFSET + block] = ROBUST_OFFSET + inverse[block]
    return index


TRANSFORMS = [(k, flip) for flip in (False, True) for k in range(4)]
_VEC_INDEX = {t: _vec_gather_index(*t) for t in TRANSFORMS}
_ACTION_PERM = {t: action_permutation(*t) for t in TRANSFORMS}


def transform_batch(vecs, crops, actions, k, flip):
    if k == 0 and not flip:
        return vecs, crops, actions
    new_vecs = vecs[..., _VEC_INDEX[(k, flip)]]
    new_crops = np.rot90(crops, k, axes=(-2, -1))
    if flip:
        new_crops = np.flip(new_crops, axis=-2)
    new_crops = np.ascontiguousarray(new_crops)
    new_actions = _ACTION_PERM[(k, flip)][actions] if actions is not None else None
    return new_vecs, new_crops, new_actions


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CROP_CHANNELS, CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(CONV_CHANNELS * CROP_SIZE * CROP_SIZE, CROP_EMBED),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(VEC_SIZE + CROP_EMBED, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )

    def forward(self, vec, crop):
        return self.head(torch.cat([vec, self.conv(crop)], dim=-1))


def greedy_action(q_values, mask):
    """Best allowed action. Both arguments are 1-D arrays."""
    q = np.asarray(q_values, dtype=np.float64).copy()
    q[~np.asarray(mask, dtype=bool)] = -np.inf
    if not np.isfinite(q).any():  # the ladder should never leave an empty mask
        return int(np.argmax(q_values))
    return int(np.argmax(q))
