#!/usr/bin/env python3
"""Eval harness: run our agent against four task scenarios and aggregate stats.

Usage (from repo root):
    # normal run
    uv run python scripts/eval.py --agent q_learning_agent --seed 42

    # smoke test
    uv run python scripts/eval.py --agent q_learning_agent --n-rounds 2

    # train run
    uv run python scripts/eval.py --agent q_learning_agent --train 1

Win metric (only for non-solo tasks): our score > sum of the 3 opponent seats' scores.
Solo tasks report coins, crates, and suicides.
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
TASKS = {
    "coin-heaven": ("coin-heaven", [], 100),
    "loot-crate": ("loot-crate", [], 100),
    "classic-mix": (
        "classic",
        ["peaceful_agent", "coin_collector_agent", "coin_collector_agent"],
        100,
    ),
    "classic-rb": ("classic", ["rule_based_agent"] * 3, 200),
}
STATS = ["rounds", "score", "coins", "kills", "suicides", "crates"]


def stat(s, key):
    # keys only appear in the JSON if the event occurred at least once
    return s.get(key, 0)


def match(agent, task, scenario, opp_list, n_rounds, seed, train):
    match_name = f"{agent}_vs_{task}"
    out_path = EVAL_DIR / f"{match_name}.json"
    log_dir = EVAL_DIR / "logs" / match_name
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", "play", "--scenario", scenario, "--agents", agent]

    for opp in opp_list:
        cmd.append(opp)

    cmd += [
        "--n-rounds",
        str(n_rounds),
        "--no-gui",
        "--save-stats",
        str(out_path),
        "--match-name",
        match_name,
        "--log-dir",
        str(log_dir),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if train > 0:
        cmd += ["--train", str(train)]

    print(f"\n=== {match_name} (seats: {agent} vs {opp_list}) ===")
    subprocess.run(cmd, cwd=FRAME, check=True)

    with open(out_path) as f:
        by_agent = json.load(f)["by_agent"]
    us = by_agent[agent]
    # duplicate seats are suffixed: opp, opp_0, opp_1, opp_2 (see environment.setup_agents)
    opps = [k for k in by_agent if k != agent]
    opp_agg = {k: sum(stat(by_agent[seat], k) for seat in opps) for k in STATS}
    return (
        us,
        opp_agg,
        max((stat(by_agent[seat], "score") for seat in opps), default=None),
    )


def row(name, s):
    return (
        f"{name:<22} {stat(s, 'rounds'):>6} {stat(s, 'score'):>8} "
        f"{stat(s, 'coins'):>5} {stat(s, 'kills'):>5} {stat(s, 'suicides'):>8}"
    )


def main():
    parser = argparse.ArgumentParser(description="Eval ladder harness")
    parser.add_argument(
        "--agent",
        default="q_learning_agent",
        help="our agent name (agent_code/<name>/)",
    )
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--train",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="train mode for the first N agents (seats, lineup order)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = EVAL_DIR / f"{args.agent}_{timestamp}_summary.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"{'agent':<22} {'rounds':>6} {'score':>8} {'coins':>5} "
        f"{'kills':>5} {'suic':>8}"
    )

    matches = []
    for task_name, task in TASKS.items():
        scenario = task[0]
        opp_list = task[1]
        n_rounds = task[2] if args.n_rounds is None else args.n_rounds
        us, opp_agg, opp_best = match(
            args.agent,
            task_name,
            scenario,
            opp_list,
            n_rounds,
            args.seed,
            args.train,
        )
        solo = not opp_list
        matches.append(
            {
                "opponent": None if solo else opp_list,
                "us": {k: stat(us, k) for k in STATS},
                "opp": None if solo else opp_agg,  # summed over the 3 opponent seats
                "opp_best_score": opp_best,
                "win": None if solo else stat(us, "score") > opp_agg["score"],
                "draw": None if solo else stat(us, "score") == opp_agg["score"],
                "task": task_name,
                "scenario": scenario,
                "n_rounds": n_rounds,
            }
        )
        print(row(f"us        (score {stat(us, 'score')})", us))

        if not solo:
            print(row(f"vs {task_name:<13} (sum {opp_agg['score']})", opp_agg))

            outcome = (
                "WIN"
                if matches[-1]["win"]
                else ("DRAW" if matches[-1]["draw"] else "LOSS")
            )
            print(
                f"-> {outcome} ({stat(us, 'score')} vs {opp_agg['score']} sum, "
                f"best seat {opp_best})\n"
            )
        else:
            print(f"solo: {stat(us, 'coins')} coins, {stat(us, 'suicides')} suicides")

    with open(summary_path, "w") as f:
        json.dump(
            {
                "agent": args.agent,
                "seed": args.seed,
                "train": args.train,
                "timestamp": timestamp,
                "matches": matches,
            },
            f,
            indent=2,
        )

    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
