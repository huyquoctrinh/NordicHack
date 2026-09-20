"""Offline, paired-seed evaluation for the survival policy.

The simulator is intentionally external to the deployable API directory. Pass
its checkout explicitly; this script never modifies or starts the live service.
"""
import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location("evaluated_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.HivemindPolicy


def evaluate(simulator: Path, policy_path: Path, seeds, cap: float, trace_every: int):
    sys.path.insert(0, str(simulator))
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    policy_type = load_policy(policy_path)
    records = []
    for seed in seeds:
        sim = SimulationCore(seed=seed)
        policy = policy_type(seed=7)
        actions, latencies, trace = [], [], []
        deaths = {"starved": 0, "eaten": 0, "predation_score_loss": 0.0}
        original_kill = sim.env.kill_agent

        def record_kill(agent):
            if agent.energy > 0:
                deaths["eaten"] += 1
                deaths["predation_score_loss"] += agent.energy / 100.0
            else:
                deaths["starved"] += 1
            original_kill(agent)

        sim.env.kill_agent = record_kill
        peak = 0
        tick = 0
        started = time.perf_counter()
        while True:
            state = sim.step(actions)
            peak = max(peak, state["num_agents"])
            tick += 1
            if not state["num_agents"] or state["sim_time"] > cap:
                break
            statuses = [item for item in state["observations"] if item is not None]
            begin = time.perf_counter()
            if hasattr(policy, "decide_dicts"):
                decided = policy.decide_dicts(statuses, state["sim_time"])
            else:
                decided = [a.model_dump() for a in policy.decide(statuses, state["sim_time"])]
            latencies.append((time.perf_counter() - begin) * 1000.0)
            assert [a["agent_id"] for a in decided] == [s["agent_id"] for s in statuses]
            assert all(math.isfinite(a[k]) for a in decided
                       for k in ("move_distance", "move_direction", "turn_angle"))
            actions = [(a["agent_id"], ActionRequest(**a)) for a in decided]
            if trace_every and tick % trace_every == 0:
                agents = sim.env.agents
                trace.append({
                    "time": round(state["sim_time"], 1),
                    "score": round(state["score"], 2),
                    "population": len(agents),
                    "mean_energy": round(statistics.mean(a.energy for a in agents), 2),
                    "max_walk": round(max(a.speed for a in agents), 3),
                    "max_sprint": round(max(a.sprint_speed for a in agents), 3),
                    "max_capacity": round(max(a.max_energy for a in agents), 3),
                    "trees": len(sim.env.trees), "fruits": len(sim.env.fruits),
                    "predators": len(sim.env.predators),
                    "policy": dict(getattr(policy, "telemetry", {})),
                })
        ordered = sorted(latencies)
        records.append({
            "seed": seed, "score": state["score"],
            "survival_seconds": state["sim_time"], "peak_population": peak,
            **deaths, "wall_seconds": time.perf_counter() - started,
            "policy_mean_ms": statistics.mean(latencies) if latencies else 0.0,
            "policy_p95_ms": ordered[int((len(ordered) - 1) * .95)] if ordered else 0.0,
            "policy_max_ms": max(latencies, default=0.0),
            "modes": policy.mode_counts, "telemetry": getattr(policy, "telemetry", {}),
            "trace": trace,
        })
        print(f"seed={seed} score={state['score']:.1f} survived={state['sim_time']:.1f} "
              f"peak={peak} p95={records[-1]['policy_p95_ms']:.2f}ms", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--policy", type=Path,
                        default=ROOT / "src/survival_agent/policies/joint_planner.py")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cap", type=float, default=3000.0)
    parser.add_argument("--trace-every", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = evaluate(args.simulator.resolve(), args.policy.resolve(), args.seeds,
                    args.cap, args.trace_every)
    scores = [run["score"] for run in runs]
    result = {
        "policy": str(args.policy.resolve()),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "simulator": str(args.simulator.resolve()), "cap": args.cap,
        "seeds": args.seeds, "runs": runs,
        "mean_score": statistics.mean(scores), "min_score": min(scores),
        "accepted": (statistics.mean(scores) >= 2500.0 and min(scores) >= 2000.0
                     and max(run["policy_p95_ms"] for run in runs) < 10.0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"mean={result['mean_score']:.1f} min={result['min_score']:.1f} "
          f"accepted={result['accepted']}")


if __name__ == "__main__":
    main()
