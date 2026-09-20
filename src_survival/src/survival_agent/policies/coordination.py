"""Searchable colony coordination layered on a frozen deployed-baseline derivative.

Only the observations received by the API are used. Relative sightings align
connected agents' frames for this tick; there is no access to simulator state.
"""
import math

from survival_agent.policies import baseline as _base

DEFAULT_CONFIG = dict(communication='component', assignment=True, breeding=True,
                      share_range=350., max_targets=12, population=16,
                      birth_slots=2, urgency=100., breeding_mode='legacy',
                      population_floor=8, population_half_life=1800.)


def rotate(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def aligned_components(statuses, communication):
    """Return groups of id -> (x, y, heading) in a common temporary frame."""
    graph = {s['agent_id']: [] for s in statuses}
    if communication != 'none':
        for status in statuses:
            bid = status['agent_id']
            for obs in status['observations']:
                aid = obs.get('id')
                if obs.get('type') != 'Agent' or aid not in graph or 'rel_dir' not in obs:
                    continue
                d, theta, rho = obs['distance'], obs['angle'], obs['rel_dir']
                graph[bid].append((aid, d*math.cos(theta), d*math.sin(theta),
                                   _base._wrap(theta + math.pi - rho)))
                graph[aid].append((bid, d*math.cos(rho), d*math.sin(rho),
                                   _base._wrap(rho + math.pi - theta)))
    visited, groups = set(), []
    for root in graph:
        if root in visited:
            continue
        poses = {root: (0., 0., 0.)}
        queue = [root]
        visited.add(root)
        for current in queue:
            x, y, heading = poses[current]
            for other, dx, dy, angle in graph[current]:
                if other in visited:
                    continue
                dx, dy = rotate(dx, dy, heading)
                poses[other] = (x+dx, y+dy, _base._wrap(heading+angle))
                visited.add(other)
                queue.append(other)
        groups.append(poses)
    return groups


class HivemindPolicy(_base.HivemindPolicy):
    def __init__(self, seed=7, config=None):
        super().__init__(seed)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.assignments = {}
        self.current_id = None
        self.current_energy = 0.
        self.telemetry = {}

    def reset(self):
        super().reset()
        self.assignments.clear()
        self.telemetry.clear()

    def decide(self, statuses, sim_time=0.):
        if sim_time < self.last_sim_time - 1e-9:
            self.reset()
        by_id = {s['agent_id']: s for s in statuses}
        enriched = {i: {**s, 'observations': list(s['observations'])} for i, s in by_id.items()}
        self.assignments = {}
        messages = 0
        for poses in aligned_components(statuses, self.config['communication']):
            pools = {'Fruit': [], 'Predator': []}
            for aid, (x, y, heading) in poses.items():
                for obs in by_id[aid]['observations']:
                    kind = obs.get('type')
                    if kind not in pools:
                        continue
                    dx, dy = rotate(obs['distance']*math.cos(obs['angle']),
                                    obs['distance']*math.sin(obs['angle']), heading)
                    px, py = x+dx, y+dy
                    if not any(math.hypot(px-q[0], py-q[1]) < 4. for q in pools[kind]):
                        pools[kind].append((px, py, aid))
            bids = []
            for aid, (x, y, heading) in poses.items():
                state = by_id[aid]
                candidates = []
                for kind, points in pools.items():
                    for index, (px, py, observer) in enumerate(points):
                        distance = math.hypot(px-x, py-y)
                        if distance > self.config['share_range']:
                            continue
                        obs = dict(type=kind, distance=distance,
                                   angle=_base._wrap(math.atan2(py-y, px-x)-heading))
                        if kind == 'Fruit':
                            candidates.append((distance, index, obs))
                            if state['energy'] < state['max_energy'] * _base.FORAGE_THRESHOLD:
                                hunger = max(0., 1. - state['energy']/state['max_energy'])
                                bids.append((distance-self.config['urgency']*hunger, aid, index, obs))
                        elif observer != aid:
                            enriched[aid]['observations'].append(obs)
                            messages += 1
                candidates.sort(key=lambda item: item[0])
                if self.config['communication'] != 'none':
                    enriched[aid]['observations'] = [o for o in enriched[aid]['observations']
                                                      if o.get('type') != 'Fruit']
                    enriched[aid]['observations'].extend(o for _, _, o in candidates[:self.config['max_targets']])
            allocated = set()
            for _, aid, index, obs in sorted(bids, key=lambda item: (item[0], item[1], item[2])):
                if aid not in self.assignments and index not in allocated:
                    self.assignments[aid] = obs
                    allocated.add(index)

        actions = super().decide([enriched[s['agent_id']] for s in statuses], sim_time)
        if self.config['breeding']:
            bounded = self.config['breeding_mode'] == 'bounded'
            target = max(self.config['population_floor'] if bounded else 4,
                         round(self.config['population'] * .5 **
                               (sim_time / self.config['population_half_life'])))
            # Replace aging agents only while the non-aging cohort is short.
            # Also cap overlap so an entire aging cohort cannot double the colony.
            young = sum(not self.memory[s['agent_id']].old for s in statuses)
            slots = self.config['birth_slots']
            if bounded:
                slots = min(slots, max(0, target-young),
                            max(0, target+max(2, target//4)-len(statuses)))
            candidates = []
            for action in actions:
                if not action.spawn_agent:
                    continue
                state = by_id[action.agent_id]
                memory = self.memory[action.agent_id]
                if len(statuses) < target or memory.old:
                    # Speed helps escape cheaply; reserve supports a viable heir.
                    rank = state['speed'] + .003*state['energy'] + (2. if memory.old else 0.)
                    candidates.append((rank, action.agent_id))
            permitted = {aid for _, aid in sorted(candidates, reverse=True)[:slots]}
            for action in actions:
                if action.spawn_agent and action.agent_id not in permitted:
                    action.spawn_agent = False
                    state = by_id[action.agent_id]
                    self._remember_prediction(self.memory[action.agent_id], state['energy'],
                        state['max_energy'], state['speed'], state['sprint_speed'],
                        action.move_distance, action.turn_angle, False, state['age'])
            self.telemetry.update(birth_target=target, young_population=young,
                                  available_birth_slots=slots)
        self.telemetry.update(population=len(statuses), shared_threats=messages,
                              assignments=len(self.assignments),
                              births_requested=sum(a.spawn_agent for a in actions))
        return actions

    def _decide_one(self, status, population):
        self.current_id = status['agent_id']
        self.current_energy = status['energy']
        return super()._decide_one(status, population)

    def _pick_fruit(self, fruits, others):
        if self.config['assignment'] and self.current_energy >= 45.:
            return self.assignments.get(self.current_id)
        return super()._pick_fruit(fruits, others)
