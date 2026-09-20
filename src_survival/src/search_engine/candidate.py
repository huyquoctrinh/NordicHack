"""Build importable policies from serializable search configurations."""

from survival_agent.policies.joint_planner import HivemindPolicy as JointPlanner


PLANNER_ATTRIBUTES = {
    "planner_enabled": "PLANNER_ENABLED",
    "horizon": "HORIZON",
    "food_energy_threshold": "FOOD_ENERGY_THRESHOLD",
    "close_fallback_radius": "CLOSE_FALLBACK_RADIUS",
    "hard_separation": "HARD_SEPARATION",
    "soft_separation": "SOFT_SEPARATION",
    "risk_weight": "RISK_WEIGHT",
    "improvement_margin": "IMPROVEMENT_MARGIN",
    "max_fruits": "MAX_FRUITS",
}

COLONY_KEYS = {
    "communication", "assignment", "breeding", "breeding_mode",
    "population", "population_floor", "population_half_life", "birth_slots",
}


def build_policy(config):
    planner = {
        PLANNER_ATTRIBUTES[key]: value
        for key, value in config.items()
        if key in PLANNER_ATTRIBUTES
    }
    colony = {key: value for key, value in config.items() if key in COLONY_KEYS}

    class CandidatePolicy(JointPlanner):
        def __init__(self, seed=7):
            super().__init__(seed=seed, config=colony)

    for name, value in planner.items():
        setattr(CandidatePolicy, name, value)
    CandidatePolicy.__name__ = "HivemindPolicy"
    return CandidatePolicy
