# Survival Agent

This repository packages the complete Survival Simulator solution: the joint
food-and-safety policy, HTTP API, offline evaluator, and staged search/tuning
engine.

The deployed policy uses an economical survival controller as its fallback. When
food and predators are visible together, a three-step planner evaluates staying,
walking toward food, retreating, lateral movement, and an affordable sprint. It
tracks predator motion in each agent's changing local frame, charges movement and
turning energy, rejects visible wall crossings, and keeps a conservative minimum
separation. Close encounters always use the proven fallback escape behavior.

Recorded development and validation results are in [`docs/evaluation.json`](docs/evaluation.json).
The packaged planner previously averaged 1,058 over six local seeds with a minimum
of 949. Those measurements are evidence, not a guaranteed remote score or a claim
that the 2,500 target has been achieved.

## Repository layout

```text
src/
  survival_agent/
    api.py                  FastAPI service
    dto.py                  request/response models
    episode.py              preflight-safe episode reset detection
    policies/
      baseline.py           economical survival controller
      coordination.py       breeding and communication layer
      joint_planner.py      predator tracker and joint action planner
  search_engine/
    evaluate.py             full-game offline evaluator
    space.py                architectures, scaling, mutations, objective
    candidate.py            configuration-to-policy builder
    tune.py                 staged time-budgeted search
    README.md               search design and usage
scripts/
  run_api.sh                local API launcher
  tune.sh                   five-hour search wrapper
deploy/
  survival-agent.service    systemd example for port 9082
tests/                      policy, API, and search tests
```

## Installation

Python 3.10 or newer is required. The offline evaluator also needs a checkout of
the Survival Simulator. The HTTP API does not import simulator internals or need
the optional scientific packages.

```bash
cd ./src_survival
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
```

If this environment will also run evaluation or tuning, install the optional
simulator-side dependencies:

```bash
.venv/bin/pip install -e ".[search]"
```

For this machine, the existing environment can also be used for quick checks:

```bash
PYTHONPATH=./src_survival/src \
  ./survival-agent-2/.venv/bin/python -m unittest discover -s tests -v
```

## Run the API

The default port is 9082.

```bash
cd ./src_survival
SURVIVAL_PORT=9082 \
.venv/bin/survival-api
```

Equivalent module command:

```bash
SURVIVAL_PORT=9082 .venv/bin/python -m survival_agent
```

Endpoints:

- `GET /` returns health and policy version.
- `POST /predict` accepts a simulator `StepResponse` and returns
  `{"actions": [...]}`.

Run one simulation at a time per process. The policy intentionally keeps memory
between ticks. Initial frames at simulation time 0 or 0.1 reset all memory and
random state, preventing a preflight request from contaminating a real game.

Quick health check:

```bash
curl -fsS http://127.0.0.1:9082/
```

Install the supplied service after changing `User`, paths, and port if needed:

```bash
sudo install -m 0644 deploy/survival-agent.service \
  /etc/systemd/system/survival-agent.service
sudo systemctl daemon-reload
sudo systemctl enable --now survival-agent.service
systemctl status survival-agent.service --no-pager
```

To run independent endpoints, create one service per port. Each process then has
independent policy state.

## Evaluate one policy

Use full game horizons when comparing policies. Short screens cannot establish
whether the colony survives long enough to approach the target.

```bash
cd ./src_survival
.venv/bin/survival-evaluate \
  --simulator ./survival-simulator-3 \
  --policy src/survival_agent/policies/joint_planner.py \
  --seeds 1 2 7 8 \
  --cap 3000 \
  --trace-every 1000 \
  --out runs/manual-evaluation.json
```

The evaluator reports score, survival time, starvation/predation indicators,
population, action modes, and policy latency. The simulator uses object sets in
some interactions, so repeat important comparisons; fixed seeds alone do not
guarantee byte-for-byte deterministic results across isolated processes.

## Run tuning

The convenient five-hour command is:

```bash
cd ./src_survival
./scripts/tune.sh ./survival-simulator-3 5 runs/search-5h
```

Direct command with explicit seed groups:

```bash
.venv/bin/survival-tune \
  --simulator ./survival-simulator-3 \
  --hours 5 \
  --out runs/search-5h \
  --development-seeds 1,2 \
  --validation-seeds 3,4,5,6 \
  --holdout-seeds 1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012
```

Watch progress with:

```bash
tail -f runs/search-5h/runner.log
cat runs/search-5h/status.json
```

If launching manually in the background, redirect output yourself:

```bash
nohup ./scripts/tune.sh ./survival-simulator-3 5 runs/search-5h \
  > runs/search-5h.runner.log 2>&1 &
```

The search never deploys a candidate. Review `final_report.json`, the raw trial
JSON, and `best_policy.py`, then validate independently before changing a service.
See [`src/search_engine/README.md`](src/search_engine/README.md) for the algorithm,
configuration, output schema, and smoke-test command.

## Tests

```bash
cd src_survival/
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover coordinate transforms, predator association, frame correction,
close-threat fallback, wall rejection, episode reset, search equivalence, robust
ranking, and API request behavior.
