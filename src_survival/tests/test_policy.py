import math
import unittest

from survival_agent.policies.joint_planner import (
    HivemindPolicy,
    IncumbentPolicy,
    PredatorTracker,
    path_near_edge,
)


def state(observations, agent_id=1, energy=80.0):
    return {
        "agent_id": agent_id,
        "energy": energy,
        "max_energy": 500.0,
        "speed": 10.0,
        "sprint_speed": 20.0,
        "age": 10.0,
        "biome": "grassland",
        "hearing_radius": 50.0,
        "vision_angle": math.pi / 3,
        "vision_range": 200.0,
        "observations": observations,
    }


def predator(distance=180.0, angle=0.0):
    return {"type": "Predator", "distance": distance, "angle": angle}


class PlannerTests(unittest.TestCase):
    def test_own_motion_is_removed_from_track_velocity(self):
        policy = HivemindPolicy()
        tracker = PredatorTracker(policy)
        tracker.observe([predator(100.0)])
        action = policy.decide([state([])], 1.0)[0]
        action.move_distance = 10.0
        action.move_direction = 0.0
        action.turn_angle = math.pi / 2
        tracker.advance_frame(action, state([]))
        track = tracker.observe([predator(90.0, -math.pi / 2)])[0]
        self.assertAlmostEqual(track["vx"], 0.0, places=7)
        self.assertAlmostEqual(track["vy"], 0.0, places=7)

    def test_close_threat_keeps_incumbent_action(self):
        status = state([
            predator(60.0),
            {"type": "Fruit", "distance": 20.0, "angle": 1.0},
        ])
        self.assertEqual(
            HivemindPolicy().decide([status], 1.0),
            IncumbentPolicy().decide([status], 1.0),
        )

    def test_safe_food_action_updates_energy_prediction(self):
        status = state([
            predator(210.0),
            {"type": "Fruit", "distance": 10.0, "angle": math.pi},
        ])
        policy = HivemindPolicy()
        action = policy.decide([status], 1.0)[0]
        self.assertGreater(action.move_distance, 0.0)
        self.assertAlmostEqual(abs(action.move_direction), math.pi)
        expected = (
            80.0 - action.move_distance * .05
            - abs(action.turn_angle) / (2 * math.pi) - .1
        )
        self.assertAlmostEqual(policy.memory[1].predicted, expected)

    def test_visible_wall_path_is_rejected(self):
        self.assertTrue(path_near_edge(30.0, 0.0, ((15.0, -20.0), (15.0, 20.0))))
        self.assertFalse(path_near_edge(0.0, 30.0, ((15.0, -20.0), (15.0, 40.0))))

    def test_no_threat_matches_incumbent(self):
        status = state([{"type": "Fruit", "distance": 20.0, "angle": .5}])
        self.assertEqual(
            HivemindPolicy().decide([status], 1.0),
            IncumbentPolicy().decide([status], 1.0),
        )
