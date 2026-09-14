"""
The Q-network of the DQN agent.

The input is a stack of 9 board planes (17 x 17) and a vector with 16 global
values. The output is one Q-value for each of the 6 actions. `to_tensors`
makes that input from the arrays of features.py. It is the only place that
scales the danger plane; callbacks.py and train.py both call it.

The class has two options for the experiments of the report:
  * `dueling`: the head splits into a state value and an action advantage;
  * `strided`: a third convolution with the stride 2 makes the network about
    three times smaller and three times faster.

`dueling` is on by default. The shaped reward puts a large state-only term
into Q. The dueling head holds that term in V(s), so the advantage head keeps
the small differences between the actions. Those differences alone select the
action.
"""

import torch
import torch.nn as nn

from .features import DANGER_LEVELS, DANGER_PLANE, N_ACTIONS, N_GLOBAL, N_PLANES


def to_tensors(planes, glob, device):
    """
    Make the float tensors that the network reads.

    A single observation gets a batch axis. The stored danger plane holds the
    levels 0 ... DANGER_LEVELS; the network wants the range 0 ... 1. The type
    change copies the stored planes, so the division never touches the buffer.
    """
    planes = torch.from_numpy(planes).to(device=device, dtype=torch.float32)
    glob = torch.from_numpy(glob).to(device=device, dtype=torch.float32)
    if planes.ndim == 3:
        planes, glob = planes.unsqueeze(0), glob.unsqueeze(0)
    planes[:, DANGER_PLANE] /= DANGER_LEVELS
    return planes, glob


class DQNNet(nn.Module):
    def __init__(self, dueling: bool = True, strided: bool = False, board: int = 17):
        super().__init__()
        self.dueling = dueling
        # train.py builds the target network with the same options.
        self.options = {"dueling": dueling, "strided": strided, "board": board}

        layers = [
            nn.Conv2d(N_PLANES, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        ]
        size = board
        if strided:
            layers += [nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU()]
            size = (board + 1) // 2
        self.conv = nn.Sequential(*layers)

        self.body = nn.Sequential(nn.Linear(64 * size * size + N_GLOBAL, 256), nn.ReLU())
        if dueling:
            self.value = nn.Linear(256, 1)
            self.advantage = nn.Linear(256, N_ACTIONS)
        else:
            self.head = nn.Linear(256, N_ACTIONS)

    def forward(self, planes, glob):
        """
        Give the Q-values for a batch.

        planes: float tensor (batch, 9, 17, 17); glob: float tensor (batch, 16).
        """
        hidden = self.conv(planes).flatten(1)
        hidden = self.body(torch.cat([hidden, glob], dim=1))
        if not self.dueling:
            return self.head(hidden)
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(dim=1, keepdim=True)
