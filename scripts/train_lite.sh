#!/usr/bin/env bash
# Training for dqn_lite: two stages, one process each. Weights carry over via
# model.pt; the replay buffer does not (classic contains coins, crates and
# opponents, so nothing from stage 1 is missing from stage 2's experience).
# Stage 1 only seeds navigation quickly, stage 2 is the tournament setting.
#
#   scripts/train_lite.sh              # both stages
#   ROUNDS=800 scripts/train_lite.sh   # shorter stage 2
#   DQN_DEVICE=cuda scripts/train_lite.sh
#   scripts/train_lite.sh reset        # rm model.pt best.pt train_log.csv
set -euo pipefail
cd "$(dirname "$0")/../bomberman_rl"
PY="${PY:-python}"
A=agent_code/dqn_lite
[ "${1:-}" = reset ] && { rm -f $A/model.pt $A/best.pt $A/train_log.csv; exit 0; }
"$PY" main.py play --no-gui --train 1 --n-rounds 150 --scenario coin-heaven --agents dqn_lite
"$PY" main.py play --no-gui --train 1 --n-rounds "${ROUNDS:-1500}" --scenario classic \
      --agents dqn_lite rule_based_agent rule_based_agent rule_based_agent
echo "done; progress is in $A/train_log.csv"
