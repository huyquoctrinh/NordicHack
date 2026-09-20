import random
import unittest

from search_engine.candidate import build_policy
from search_engine.space import (
    DEFAULT_CONFIG,
    architecture_seeds,
    compound_variants,
    effective_key,
    mutate_one,
    objective,
)


class SearchTests(unittest.TestCase):
    def test_latency_constraint_and_robust_objective(self):
        self.assertEqual(objective([{"score": 2600, "policy_p95_ms": 11}]), -1e9)
        self.assertEqual(
            objective([
                {"score": 2600, "policy_p95_ms": 1},
                {"score": 2000, "policy_p95_ms": 1},
            ]),
            2150,
        )

    def test_architectures_are_unique(self):
        keys = [effective_key(config) for config in architecture_seeds()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_compound_scaling_changes_coupled_dimensions(self):
        variants = list(compound_variants(DEFAULT_CONFIG))
        low, _, high = variants
        self.assertLessEqual(low["horizon"], high["horizon"])
        self.assertLess(low["soft_separation"], high["soft_separation"])
        self.assertLessEqual(low["max_fruits"], high["max_fruits"])

    def test_mutation_changes_one_setting(self):
        for seed in range(30):
            result = mutate_one(DEFAULT_CONFIG, random.Random(seed))
            self.assertEqual(
                sum(result[key] != DEFAULT_CONFIG[key] for key in DEFAULT_CONFIG),
                1,
            )

    def test_candidate_builder_applies_planner_and_colony_settings(self):
        policy_type = build_policy({
            **DEFAULT_CONFIG,
            "horizon": 5,
            "population": 12,
            "breeding_mode": "bounded",
        })
        policy = policy_type()
        self.assertEqual(policy.HORIZON, 5)
        self.assertEqual(policy.config["population"], 12)
        self.assertEqual(policy.config["breeding_mode"], "bounded")
