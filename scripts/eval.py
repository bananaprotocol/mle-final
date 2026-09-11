#!/usr/bin/env python3
"""Eval harness: run our agent against the baseline ladder and aggregate stats.

Usage (from repo root):
    uv run python scripts/eval.py --agent q_learning_agent --n-rounds 10 --seed 42
    uv run python scripts/eval.py --agent q_learning_agent --train 1

Win metric: our score > sum of the 3 opponent seats' scores.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAME = ROOT / "bomberman_rl"
EVAL_DIR = ROOT / "results" / "eval"
LADDER = ["rule_based_agent", "coin_collector_agent", "peaceful_agent", "random_agent"]
STATS = ["rounds", "score", "coins", "kills", "suicides"]


def stat(s, key):
    # keys only appear in the JSON if the event occurred at least once
    return s.get(key, 0)


def match(agent, opp, n_rounds, seed, train):
    match_name = f"{agent}_vs_{opp}"
    out_path = EVAL_DIR / f"{match_name}.json"
    cmd = [
        sys.executable, "main.py", "play",
        "--agents", agent, opp, opp, opp,
        "--n-rounds", str(n_rounds),
        "--no-gui", "--save-stats", str(out_path),
        "--match-name", match_name,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if train > 0:
        cmd += ["--train", str(train)]

    print(f"\n=== {match_name} (seats: {agent} vs 3x {opp}) ===")
    subprocess.run(cmd, cwd=FRAME, check=True)

    with open(out_path) as f:
        by_agent = json.load(f)["by_agent"]
    us = by_agent[agent]
    # duplicate seats are suffixed: opp, opp_0, opp_1, opp_2 (see environment.setup_agents)
    opps = [k for k in by_agent if k != agent]
    opp_agg = {k: sum(stat(by_agent[seat], k) for seat in opps) for k in STATS}
    return us, opp_agg, max(stat(by_agent[seat], "score") for seat in opps)


def row(name, s):
    return (f"{name:<22} {stat(s, 'rounds'):>6} {stat(s, 'score'):>8} "
            f"{stat(s, 'coins'):>5} {stat(s, 'kills'):>5} {stat(s, 'suicides'):>8}")


def main():
    parser = argparse.ArgumentParser(description="Eval ladder harness")
    parser.add_argument("--agent", default="q_learning_agent",
                        help="our agent name (agent_code/<name>/)")
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="train mode for the first N agents (seats, lineup order)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = EVAL_DIR / f"{args.agent}_{timestamp}_summary.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'agent':<22} {'rounds':>6} {'score':>8} {'coins':>5} "
          f"{'kills':>5} {'suic':>8}")

    matches = []
    for opp in LADDER:
        if opp == args.agent:
            print(f"skipping {opp} (same as our agent)")
            continue
        us, opp_agg, opp_best = match(args.agent, opp, args.n_rounds,
                                      args.seed, args.train)
        matches.append({
            "opponent": opp,
            "us": {k: stat(us, k) for k in STATS},
            "opp": opp_agg,  # summed over the 3 opponent seats
            "opp_best_score": opp_best,
            "win": stat(us, "score") > opp_agg["score"],
            "draw": stat(us, "score") == opp_agg["score"],
        })
        print(row(f"us        (score {stat(us, 'score')})", us))
        print(row(f"vs {opp:<13} (sum {opp_agg['score']})", opp_agg))
        outcome = "WIN" if matches[-1]["win"] else ("DRAW" if matches[-1]["draw"] else "LOSS")
        print(f"-> {outcome} ({stat(us, 'score')} vs {opp_agg['score']} sum, "
              f"best seat {opp_best})\n")

    with open(summary_path, "w") as f:
        json.dump({
            "agent": args.agent,
            "n_rounds": args.n_rounds,
            "seed": args.seed,
            "train": args.train,
            "timestamp": timestamp,
            "matches": matches,
        }, f, indent=2)

    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
