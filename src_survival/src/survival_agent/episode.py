"""Detect fresh simulator frames even after a /predict preflight probe."""

def starts_new_episode(sim_time, statuses, last_sim_time):
    if last_sim_time is None or sim_time < last_sim_time - 1e-9:
        return True
    # The evaluator's game starts at t=0.1, with every initial agent aged 0.1.
    # A preflight at t=0 can otherwise contaminate that first game. This also
    # makes retries of an initial frame reproduce the same initialized action.
    return (0. <= sim_time <= .1 + 1e-9
            and all(s['age'] <= sim_time + 1e-9 for s in statuses))
