"""Search space and robust ranking functions."""

import json
import random
import statistics


DEFAULT_CONFIG = {
    "planner_enabled": True,
    "horizon": 3,
    "food_energy_threshold": .65,
    "close_fallback_radius": 75.0,
    "hard_separation": 35.0,
    "soft_separation": 90.0,
    "risk_weight": .045,
    "improvement_margin": .25,
    "max_fruits": 4,
    "communication": "none",
    "assignment": False,
    "breeding": True,
    "breeding_mode": "legacy",
    "population": 16,
    "population_floor": 8,
    "population_half_life": 1800.0,
    "birth_slots": 2,
}


def objective(runs, latency_limit_ms=10.0):
    """Reward both average and worst-case score; reject slow policies."""
    if not runs or any(run["policy_p95_ms"] >= latency_limit_ms for run in runs):
        return -1e9
    scores = [run["score"] for run in runs]
    return .5 * statistics.mean(scores) + .5 * min(scores)


def effective_key(config):
    """Remove settings that cannot affect the selected architecture."""
    active = dict(config)
    if not active["planner_enabled"]:
        for name in (
            "horizon", "food_energy_threshold", "close_fallback_radius",
            "hard_separation", "soft_separation", "risk_weight",
            "improvement_margin", "max_fruits",
        ):
            active.pop(name, None)
    if active["communication"] == "none" and not active["assignment"]:
        active.pop("assignment", None)
    if not active["breeding"]:
        for name in (
            "breeding_mode", "population", "population_floor",
            "population_half_life", "birth_slots",
        ):
            active.pop(name, None)
    elif active["breeding_mode"] == "legacy":
        active.pop("population_floor", None)
    return json.dumps(active, sort_keys=True)


def architecture_seeds():
    """Matched ablations that identify which large modules earn their cost."""
    base = dict(DEFAULT_CONFIG)
    yield {**base, "planner_enabled": False}
    yield base
    yield {**base, "breeding_mode": "bounded"}
    yield {**base, "communication": "component"}
    yield {**base, "assignment": True}


def compound_variants(config):
    """Scale horizon, risk radius and action breadth together.

    This follows EfficientNet's useful search principle: first choose a sound
    architecture, then scale coupled dimensions in a balanced way rather than
    maximizing one dimension independently.
    """
    for scale in (.8, 1.0, 1.25):
        yield {
            **config,
            "horizon": max(2, min(5, round(config["horizon"] * scale))),
            "soft_separation": max(65.0, min(130.0, config["soft_separation"] * scale)),
            "max_fruits": max(2, min(7, round(config["max_fruits"] * scale))),
            "risk_weight": max(.02, min(.09, config["risk_weight"] * scale)),
        }


MUTATIONS = {
    "food_energy_threshold": [.5, .6, .65, .75, .85],
    "close_fallback_radius": [60.0, 75.0, 90.0, 110.0],
    "hard_separation": [25.0, 35.0, 45.0, 55.0],
    "improvement_margin": [0.0, .25, .5, 1.0],
    "population": [10, 12, 16, 20],
    "birth_slots": [1, 2, 3],
    "population_half_life": [600.0, 1200.0, 1800.0, 3600.0],
}


def mutate_one(config, rng: random.Random):
    """Change exactly one active parameter for interpretable comparisons."""
    available = dict(MUTATIONS)
    if not config["planner_enabled"]:
        for name in (
            "food_energy_threshold", "close_fallback_radius",
            "hard_separation", "improvement_margin",
        ):
            available.pop(name)
    key = rng.choice(list(available))
    choices = [value for value in available[key] if value != config[key]]
    return {**config, key: rng.choice(choices)}
