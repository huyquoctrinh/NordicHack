import asyncio
import json
import unittest

from survival_agent import api
from survival_agent.dto import ObservationResponse, StepResponse


class ApiTests(unittest.TestCase):
    def test_health(self):
        self.assertEqual(api.health()["candidate"], "joint-planner-v1")

    def test_predict_shape(self):
        state = ObservationResponse(
            agent_id=1,
            energy=100,
            biome="grassland",
            age=.1,
            speed=10,
            sprint_speed=20,
            hearing_radius=50,
            vision_angle=1.047,
            vision_range=200,
            max_energy=500,
            observations=[],
        )
        response = asyncio.run(api.predict(StepResponse(
            game_status="test",
            score=0,
            sim_time=.1,
            n_agents=1,
            agent_status=[state],
        )))
        body = json.loads(response.body)
        self.assertEqual(body["actions"][0]["agent_id"], 1)
