#!/usr/bin/env bash
# Curriculum training for dqn_agent_v2.
#
# Every stage keeps the network weights *and* the replay buffer (replay.npz
# next to model.pt). The framework fixes one scenario per process, so mixing
# scenarios is done here at the shell level: stage 3 runs short blocks of
# different scenarios back to back, and the persistent buffer keeps all of them
# in the gradient updates. All blocks of a stage share DQN_STAGE, so epsilon
# decays once over the whole stage instead of restarting per block.
#
#   scripts/train_curriculum.sh           # run stages 1, 2, 3 in order
#   scripts/train_curriculum.sh 3         # run only stage 3
#   DQN_CYCLES=4 scripts/train_curriculum.sh 3    # shorter stage 3
#   scripts/train_curriculum.sh reset     # delete all training artefacts
#
#   DQN_AGENT_DIR=ekubo DQN_DEVICE=cuda scripts/train_curriculum.sh 4
#                                         # warm-start fine-tune of the copy
#
# | stage | scenario(s)                                  | rounds            |
# |-------|----------------------------------------------|-------------------|
# | 1     | coin-heaven, solo                            | 300               |
# | 2     | loot-crate, solo                             | 400               |
# | 3     | mixed cycles of 5 x 25 rounds:               | 125 x DQN_CYCLES  |
# |       |   coin-heaven solo                           | (default 8 -> 1000)|
# |       |   loot-crate solo                            |                   |
# |       |   classic + peaceful_agent coin_collector    |                   |
# |       |   classic + 3x rule_based_agent              |                   |
# |       |   classic + 3x rule_based_agent              |                   |
# | 4     | warm-start fine-tune, mixed, rule-based      | 150 x DQN_CYCLES  |
# |       | heavy; cycles of 6 x 25 rounds:              | (default 5 -> 750)|
# |       |   coin-heaven solo                           |                   |
# |       |   loot-crate solo                            |                   |
# |       |   classic + peaceful_agent coin_collector    |                   |
# |       |   3 x classic + 3x rule_based_agent          |                   |
# | 6     | self-play vs a frozen snapshot (opt-in)      | 1500              |
#
# rule_based_agent appears twice per stage-3 cycle so the tournament setting
# dominates; stage 4 triples it, because it fine-tunes an agent that already
# plays the solo scenarios well and only loses rounds against rule_based_agent.
#
# Timing: with safe exploration the agent rarely dies early, so rounds run to
# about the full 400 steps, roughly 2 s per round on the GPU. Budget ~10 min
# for stage 1, ~15 min for stage 2, ~35 min for a default stage 3 and ~25 min
# for a default stage 4.
#
# Set DQN_DEVICE=cuda to train on the GPU, PY to pick the interpreter, and
# DQN_AGENT_DIR to train a copy of the agent instead of agent_code/dqn_agent_v2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/bomberman_rl"
PY="${PY:-python}"
AGENT="${DQN_AGENT_DIR:-dqn_agent_v2}"
AGENT_PATH="$ROOT/bomberman_rl/agent_code/$AGENT"

ARTEFACTS=(model.pt best.pt train_state.pkl train_log.csv replay.npz)

reset_agent () {
  echo "removing training artefacts from $AGENT_PATH:"
  for f in "${ARTEFACTS[@]}"; do
    if [ -e "$AGENT_PATH/$f" ]; then
      echo "  $f"
    else
      echo "  $f (not present)"
    fi
  done
  for f in "${ARTEFACTS[@]}"; do
    rm -f "$AGENT_PATH/$f"
  done
  echo "reset done; the next run starts from scratch."
}

run_block () {
  local stage="$1" rounds="$2" scenario="$3"; shift 3
  echo "=== stage $stage: $rounds rounds, scenario $scenario, opponents: ${*:-none} ==="
  DQN_STAGE="$stage" "$PY" main.py play --no-gui --train 1 \
    --n-rounds "$rounds" --scenario "$scenario" \
    --agents "$AGENT" "$@"
}

stage_1 () { run_block 1 300 coin-heaven; }   # navigation, dense coins
stage_2 () { run_block 2 400 loot-crate; }    # bomb crates and escape

stage_3 () {                                  # everything at once, mixed
  local cycles="${DQN_CYCLES:-8}" c
  for ((c = 1; c <= cycles; c++)); do
    echo "--- stage 3, cycle $c/$cycles ---"
    run_block 3 25 coin-heaven
    run_block 3 25 loot-crate
    run_block 3 25 classic peaceful_agent coin_collector_agent
    run_block 3 25 classic rule_based_agent rule_based_agent rule_based_agent
    run_block 3 25 classic rule_based_agent rule_based_agent rule_based_agent
  done
}

stage_4 () {                                  # warm start, rule-based heavy
  local cycles="${DQN_CYCLES:-5}" c b
  for ((c = 1; c <= cycles; c++)); do
    echo "--- stage 4, cycle $c/$cycles ---"
    run_block 4 25 coin-heaven               # keep the solo skills alive
    run_block 4 25 loot-crate
    run_block 4 25 classic peaceful_agent coin_collector_agent
    for ((b = 1; b <= 3; b++)); do
      run_block 4 25 classic rule_based_agent rule_based_agent rule_based_agent
    done
  done
}

stage_6 () { run_block 6 1500 classic rule_based_agent rule_based_agent dqn_agent_frozen; }

if [ "${1:-}" = "reset" ]; then
  reset_agent
  exit 0
fi

stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then stages=(1 2 3); fi

for stage in "${stages[@]}"; do
  if [ "$stage" = "6" ]; then
    # self-play opponent: a frozen snapshot that never trains
    rm -rf agent_code/dqn_agent_frozen
    mkdir -p agent_code/dqn_agent_frozen
    cp "agent_code/$AGENT"/{callbacks.py,features.py,model.py,model.pt} \
       agent_code/dqn_agent_frozen/
    sed -i 's/from \.features/from .features/' agent_code/dqn_agent_frozen/callbacks.py
  fi
  "stage_$stage"
done

echo "done; progress is in agent_code/$AGENT/train_log.csv"
