# Search engine

`search_engine` performs a staged, time-budgeted search over controller structure
and parameters. It borrows the sequencing idea from EfficientNet-style compound
scaling: establish a useful architecture first, then scale interacting dimensions
together, and finally make interpretable one-parameter mutations.

This is controller search, not neural-network training. Every score comes from a
complete simulator episode up to the configured cap.

## Stages

1. **Architecture ablation** compares the fallback, joint planner, bounded
   breeding, communication, and assignment variants.
2. **Compound scaling** expands the leading architecture across planner horizon,
   predator safety radius, fruit candidate breadth, and risk weight together.
3. **Local mutation** changes one active parameter at a time. Configurations that
   differ only in disabled settings share the same effective key and are skipped.
4. **Development ranking** repeats every development seed. The objective is
   `0.5 × mean score + 0.5 × minimum score`, subject to the p95 latency limit.
5. **Validation promotion** evaluates the best three candidates on distinct
   seeds. Development scores do not select among candidates after validation is
   complete.
6. **Holdout** evaluates the selected winner once on untouched seeds. Holdout
   never feeds back into proposals. Acceptance requires complete holdout, target
   mean, target minimum, and latency constraints.

The default acceptance thresholds are mean ≥ 2,500, minimum ≥ 2,000, and policy
p95 < 10 ms. A high score on one seed cannot satisfy acceptance.

## Search parameters

Planner parameters:

- `planner_enabled`: enable joint food/safety action comparison.
- `horizon`: number of ticks used for local prediction.
- `food_energy_threshold`: energy fraction below which planning for food begins.
- `close_fallback_radius`: distance below which the incumbent escape action wins.
- `hard_separation` and `soft_separation`: conservative action safety gates.
- `risk_weight`: penalty for predicted predator proximity.
- `improvement_margin`: minimum score advantage required to override fallback.
- `max_fruits`: number of visible fruit targets evaluated per decision.

Colony parameters:

- `communication` and `assignment`: connected-frame sharing and fruit allocation.
- `breeding_mode`: legacy replacement or bounded overlap.
- `population`, `population_floor`, `population_half_life`, and `birth_slots`.

The defaults reproduce the packaged joint planner. `candidate.py` converts any
serializable configuration into a policy class without modifying source files.

## Full search

Install the repository in editable mode, then run:

```bash
survival-tune \
  --simulator /home/van/survival-simulator-3 \
  --hours 5 \
  --cap 3000 \
  --out runs/search-5h
```

Useful options:

```text
--development-seeds 1,2
--development-repeats 2
--validation-seeds 3,4,5,6
--holdout-seeds 1001,1002,...,1012
--latency-limit-ms 10
--target-mean 2500
--target-min 2000
--rng-seed 20260920
```

## Smoke test

This checks orchestration quickly; its short cap must not be used to select a
production policy:

```bash
survival-tune \
  --simulator /home/van/survival-simulator-3 \
  --hours 0.03 \
  --cap 20 \
  --development-seeds 1 \
  --development-repeats 1 \
  --validation-seeds 2 \
  --holdout-seeds 3 \
  --out runs/smoke
```

## Output files

- `status.json`: atomic progress snapshot, deadline, counts, and final status.
- `history.json`: every configuration, run, and captured error.
- `candidate_NNNN.py`: importable policy wrapper for an exact configuration.
- `trial_*.json`: evaluator output, including latency and behavioral traces.
- `trial_*.log`: subprocess output for troubleshooting.
- `best_policy.py`: selected wrapper after development/validation.
- `final_report.json`: winner, holdout completeness, and acceptance decision.

Interrupted work remains reviewable. An incomplete holdout can never be accepted.
The search engine does not install services, modify running APIs, or copy a winner
into the runtime policy automatically.

## Why results vary

The simulator uses unordered object sets for some neighborhood and collision
operations. A fixed RNG seed therefore may not reproduce exactly across isolated
processes. Development repeats reduce selection based on luck; validation and
holdout measure generalization. More repeats and more fresh seeds improve evidence
but consume more of the fixed time budget.
