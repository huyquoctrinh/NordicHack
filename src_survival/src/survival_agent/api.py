"""FastAPI entry point for the packaged joint planner."""

import json
import os

from fastapi import Body, FastAPI, Response

from survival_agent.dto import StepResponse
from survival_agent.episode import starts_new_episode
from survival_agent.policies.joint_planner import HivemindPolicy


app = FastAPI(title="Survival Simulator Joint Planner", version="1.0.0")
policy = HivemindPolicy(seed=int(os.environ.get("SURVIVAL_POLICY_SEED", "7")))
episode = 0
last_sim_time = None


@app.post("/predict")
async def predict(step: StepResponse = Body(...)):
    global episode, last_sim_time
    request = step.model_dump()
    if starts_new_episode(step.sim_time, request["agent_status"], last_sim_time):
        episode += 1
        policy.reset()
        policy.last_sim_time = -1.0
    last_sim_time = step.sim_time
    actions = policy.decide(request["agent_status"], step.sim_time)
    result = {"actions": [action.model_dump() for action in actions]}
    return Response(
        json.dumps(result, separators=(",", ":"), allow_nan=False),
        media_type="application/json",
    )


@app.get("/")
def health():
    return {
        "message": "Agent endpoint running!",
        "candidate": "joint-planner-v1",
        "episode_reset": "initial-frame-v1",
    }
