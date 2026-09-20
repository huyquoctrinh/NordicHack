"""Time-budgeted architecture and parameter search.

Every ranked episode uses the requested full horizon. Search saves source-level
candidate wrappers, raw evaluation JSON, logs, status, and a final report.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

from search_engine.space import (
    architecture_seeds,
    compound_variants,
    effective_key,
    mutate_one,
    objective,
)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parse_seeds(value):
    return [int(item) for item in value.split(",") if item]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--cap", type=float, default=3000.0)
    parser.add_argument("--development-seeds", type=parse_seeds, default=[1, 2])
    parser.add_argument("--validation-seeds", type=parse_seeds, default=[3, 4, 5, 6])
    parser.add_argument("--holdout-seeds", type=parse_seeds,
                        default=list(range(1001, 1013)))
    parser.add_argument("--development-repeats", type=int, default=2)
    parser.add_argument("--latency-limit-ms", type=float, default=10.0)
    parser.add_argument("--target-mean", type=float, default=2500.0)
    parser.add_argument("--target-min", type=float, default=2000.0)
    parser.add_argument("--rng-seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    if args.hours <= 0 or args.cap <= 0 or args.development_repeats < 1:
        parser.error("hours, cap, and development-repeats must be positive")

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner_log = output / "runner.log"

    def emit(message):
        print(message, flush=True)
        with runner_log.open("a") as stream:
            stream.write(message + "\n")

    started = time.time()
    deadline = started + args.hours * 3600
    search_end = started + args.hours * 3600 * .60
    validation_end = started + args.hours * 3600 * .78
    evaluator = Path(__file__).with_name("evaluate.py")
    rng = random.Random(args.rng_seed)
    history, durations, seen = [], [], set()
    initial = list(architecture_seeds())
    compound_queue = []
    status = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        "cap": args.cap,
        "target_mean": args.target_mean,
        "target_min": args.target_min,
        "latency_limit_ms": args.latency_limit_ms,
        "development_seeds": args.development_seeds,
        "development_repeats": args.development_repeats,
        "accepted": False,
    }
    atomic_json(output / "status.json", status)

    def create_candidate(config):
        index = len(history)
        path = output / f"candidate_{index:04d}.py"
        path.write_text(
            "from search_engine.candidate import build_policy\n"
            f"HivemindPolicy = build_policy({config!r})\n"
        )
        entry = {
            "index": index,
            "config": config,
            "policy": str(path),
            "runs": [],
            "errors": [],
        }
        history.append(entry)
        return entry

    def run_trial(entry, seed, phase, phase_deadline):
        average = (statistics.mean(durations) if durations
                   else min(60.0, max(5.0, args.cap * .2)))
        remaining = min(deadline, phase_deadline) - time.time()
        if remaining < max(5.0, average * 1.15):
            return False
        repeat = sum(
            run["seed"] == seed and run["phase"] == phase
            for run in entry["runs"]
        )
        report = output / f"trial_{entry['index']:04d}_{phase}_{seed}_{repeat}.json"
        log = report.with_suffix(".log")
        command = [
            sys.executable, str(evaluator), "--simulator", str(args.simulator.resolve()),
            "--policy", entry["policy"], "--seeds", str(seed),
            "--cap", str(args.cap), "--trace-every", "1000", "--out", str(report),
        ]
        begin = time.time()
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=remaining,
            )
            log.write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError(completed.stdout[-2000:])
            run = json.loads(report.read_text())["runs"][0]
            run.update(phase=phase, repeat=repeat)
            entry["runs"].append(run)
            durations.append(time.time() - begin)
            emit(
                f"candidate={entry['index']} phase={phase} seed={seed} "
                f"repeat={repeat} score={run['score']:.1f}"
            )
        except Exception as error:
            entry["errors"].append({"seed": seed, "phase": phase, "error": repr(error)})
        atomic_json(output / "history.json", history)
        status.update(
            candidates=len(history),
            completed_episodes=sum(len(item["runs"]) for item in history),
            latest_candidate=entry["index"],
            latest_phase=phase,
        )
        atomic_json(output / "status.json", status)
        return True

    def complete(entry, phase, seeds, repeats=1):
        return all(
            sum(run["phase"] == phase and run["seed"] == seed for run in entry["runs"])
            >= repeats
            for seed in seeds
        )

    def ranking(phase, seeds, repeats=1):
        eligible = [
            item for item in history if complete(item, phase, seeds, repeats)
        ]
        return sorted(
            eligible,
            key=lambda item: objective(
                [run for run in item["runs"] if run["phase"] == phase],
                args.latency_limit_ms,
            ),
            reverse=True,
        )

    try:
        while time.time() < search_end:
            if initial:
                config = initial.pop(0)
            elif compound_queue:
                config = compound_queue.pop(0)
            else:
                leaders = ranking(
                    "development", args.development_seeds,
                    args.development_repeats,
                )[:4]
                if not leaders:
                    break
                if not any(item.get("compound_expanded") for item in leaders):
                    leader = leaders[0]
                    leader["compound_expanded"] = True
                    compound_queue.extend(compound_variants(leader["config"]))
                    continue
                config = mutate_one(rng.choice(leaders)["config"], rng)
            key = effective_key(config)
            if key in seen:
                continue
            seen.add(key)
            entry = create_candidate(config)
            exhausted = False
            for _ in range(args.development_repeats):
                for seed in args.development_seeds:
                    if not run_trial(entry, seed, "development", search_end):
                        exhausted = True
                        break
                if exhausted:
                    break
            if exhausted:
                break

        leaders = ranking(
            "development", args.development_seeds, args.development_repeats
        )[:3]
        for seed in args.validation_seeds:
            for entry in leaders:
                if not run_trial(entry, seed, "validation", validation_end):
                    break
        validated = ranking("validation", args.validation_seeds)
        finalists = validated or leaders
        if not finalists:
            raise RuntimeError("no candidate completed development")
        winner = finalists[0]
        shutil.copyfile(winner["policy"], output / "best_policy.py")
        for seed in args.holdout_seeds:
            if not run_trial(winner, seed, "holdout", deadline):
                break
        holdout = [run for run in winner["runs"] if run["phase"] == "holdout"]
        accepted = (
            bool(validated)
            and complete(winner, "holdout", args.holdout_seeds)
            and statistics.mean(run["score"] for run in holdout) >= args.target_mean
            and min(run["score"] for run in holdout) >= args.target_min
            and max(run["policy_p95_ms"] for run in holdout) < args.latency_limit_ms
        )
        status.update(
            status="complete",
            accepted=accepted,
            winner=winner,
            validation_complete=bool(validated),
            holdout_complete=complete(winner, "holdout", args.holdout_seeds),
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(output / "final_report.json", status)
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        atomic_json(output / "status.json", status)


if __name__ == "__main__":
    main()
