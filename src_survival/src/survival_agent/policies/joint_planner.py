"""Short-horizon food/safety planner layered on the immutable hosted candidate.

Only API observations are used. Predictor errors widen uncertainty; close-contact
escape and situations without reliable planning information retain the incumbent.
"""
import math

from survival_agent.policies import baseline as _base
from survival_agent.policies.coordination import HivemindPolicy as CoordinationPolicy


INCUMBENT_CONFIG = {
    'communication': 'none',
    'assignment': False,
    'breeding': True,
    'share_range': 350.0,
    'max_targets': 12,
    'population': 16,
    'birth_slots': 2,
    'urgency': 100.0,
    'breeding_mode': 'legacy',
    'population_floor': 8,
    'population_half_life': 1800.0,
}


class IncumbentPolicy(CoordinationPolicy):
    """The strongest pre-planner controller used as the safe fallback."""

    def __init__(self, seed=7, config=None):
        super().__init__(seed=seed, config={**INCUMBENT_CONFIG, **(config or {})})


def rotate(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c*x-s*y, s*x+c*y


def displacement(action, state):
    distance = min(action.move_distance, state['sprint_speed'])
    if state['energy'] < state['max_energy']*.2:
        distance = min(distance, state['speed'])
    distance *= {'swamp':.5, 'river':.3, 'desert':.8}.get(state.get('biome'), 1.)
    return distance*math.cos(action.move_direction), distance*math.sin(action.move_direction)


def path_near_edge(dx, dy, edge, clearance=6.):
    (x1,y1),(x2,y2)=edge
    def distance(px,py,ax,ay,bx,by):
        ux,uy=bx-ax,by-ay
        t=max(0.,min(1.,((px-ax)*ux+(py-ay)*uy)/max(1e-12,ux*ux+uy*uy)))
        return math.hypot(px-ax-t*ux,py-ay-t*uy)
    cross=lambda ax,ay,bx,by:ax*by-ay*bx
    ex,ey=x2-x1,y2-y1
    denom=cross(dx,dy,ex,ey)
    if abs(denom)>1e-9:
        t=cross(x1,y1,ex,ey)/denom
        u=cross(x1,y1,dx,dy)/denom
        if 0<=t<=1 and 0<=u<=1:return True
    return min(distance(0.,0.,x1,y1,x2,y2),distance(dx,dy,x1,y1,x2,y2),
               distance(x1,y1,0.,0.,dx,dy),distance(x2,y2,0.,0.,dx,dy))<clearance


class PredatorTracker:
    """One observer's unlabeled tracks, expressed in its current local frame."""
    def __init__(self, owner):
        self.owner = owner
        self.tracks = []
        self.errors = []

    def observe(self, observations):
        points = [(o['distance']*math.cos(o['angle']), o['distance']*math.sin(o['angle']))
                  for o in observations if o.get('type') == 'Predator']
        # Globally greedy one-to-one association avoids using one old track twice.
        pairs = sorted((math.hypot(x-t['x'],y-t['y']),i,j)
                       for i,(x,y) in enumerate(points) for j,t in enumerate(self.tracks))
        matched, used = {}, set()
        for distance,i,j in pairs:
            if (i in matched or j in used
                    or distance > self.owner.TRACK_GATE
                    + self.tracks[j]['missing'] * self.owner.THREAT_SPEED):
                continue
            matched[i] = j
            used.add(j)
        updated = []
        for i,(x,y) in enumerate(points):
            if i in matched:
                old = self.tracks[matched[i]]
                gap = old['missing']+1
                vx,vy = (x-old['x'])/gap, (y-old['y'])/gap
                error = math.hypot(vx-old['vx'],vy-old['vy'])
                self.errors.append(error)
                self.errors = self.errors[-128:]
                updated.append(dict(x=x,y=y,vx=vx,vy=vy,missing=0,
                    samples=old['samples']+1,
                    uncertainty=min(self.owner.MAX_UNCERTAINTY,
                                    self.owner.BASE_UNCERTAINTY + error)))
            else:
                updated.append(dict(x=x,y=y,vx=0.,vy=0.,missing=0,samples=1,
                                    uncertainty=self.owner.INITIAL_UNCERTAINTY))
        for j,old in enumerate(self.tracks):
            if j not in used and old['missing'] < self.owner.TRACK_MEMORY:
                updated.append({**old,'missing':old['missing']+1,
                                'uncertainty':old['uncertainty'] + self.owner.THREAT_SPEED})
        self.tracks = updated
        return self.tracks

    def advance_frame(self, action, state):
        dx,dy=displacement(action,state)
        for track in self.tracks:
            track['x'],track['y']=rotate(track['x']-dx,track['y']-dy,-action.turn_angle)
            track['vx'],track['vy']=rotate(track['vx'],track['vy'],-action.turn_angle)


class HivemindPolicy(IncumbentPolicy):
    PLANNER_ENABLED = True
    HORIZON = 3
    FOOD_ENERGY_THRESHOLD = .65
    CLOSE_FALLBACK_RADIUS = 75.
    MAX_FRUITS = 4
    THREAT_SPEED = 15.
    TRACK_GATE = 24.
    TRACK_MEMORY = 3
    BASE_UNCERTAINTY = 2.
    INITIAL_UNCERTAINTY = 15.
    MAX_UNCERTAINTY = 20.
    HARD_SEPARATION = 35.
    SOFT_SEPARATION = 90.
    RISK_WEIGHT = .045
    FRUIT_PROGRESS_WEIGHT = .10
    FRUIT_REWARD = 20.
    IMPROVEMENT_MARGIN = .25

    def __init__(self, seed=7, config=None):
        super().__init__(seed, config=config)
        self.trackers = {}
        self.planner_counts = dict(planned=0,overrides=0,close_fallbacks=0)

    def reset(self):
        super().reset()
        self.trackers.clear()
        self.planner_counts = dict(planned=0,overrides=0,close_fallbacks=0)

    def decide(self, statuses, sim_time=0.):
        actions=super().decide(statuses,sim_time)
        alive={s['agent_id'] for s in statuses}
        self.trackers={aid:t for aid,t in self.trackers.items() if aid in alive}
        self.telemetry.update(self.planner_counts)
        return actions

    def _score(self, action, state, tracks, fruits):
        dx,dy=displacement(action,state)
        move_cost=_base._move_cost(action.move_distance,state['speed'],state['sprint_speed'],state['energy'],state['max_energy'])
        cost=self.HORIZON*(move_cost+.1)+_base._turn_cost(action.turn_angle)
        # Reject paths crossing or grazing a visible wall. Local geometry is
        # uncertain beyond the observation, so do not invent unseen obstacles.
        for obs in state['observations']:
            if obs.get('type') != 'Edge':
                continue
            if path_near_edge(dx*self.HORIZON,dy*self.HORIZON,obs['coords']):
                return -float('inf'), -float('inf')
        minimum=float('inf')
        for step in range(1,self.HORIZON+1):
            ax,ay=dx*step,dy*step
            for t in tracks:
                # The reachable-distance bound assumes an immediate turn toward
                # us; motion prediction is an additional, never weaker check.
                missing=t['missing']
                gap=(math.hypot(t['x']-ax,t['y']-ay)
                     - self.THREAT_SPEED * (step + missing))
                px=t['x']+t['vx']*(step+missing)
                py=t['y']+t['vy']*(step+missing)
                predicted=math.hypot(px-ax,py-ay)-t['uncertainty']
                minimum=min(minimum,gap,predicted)
        food=0.
        hunger=max(.2,1.-state['energy']/state['max_energy'])
        for f in fruits:
            x,y=f['distance']*math.cos(f['angle']),f['distance']*math.sin(f['angle'])
            final=math.hypot(x-dx*self.HORIZON,y-dy*self.HORIZON)
            value=self.FRUIT_PROGRESS_WEIGHT*(f['distance']-final)
            for step in range(1,self.HORIZON+1):
                if math.hypot(x-dx*step,y-dy*step)<=9.:
                    value+=self.FRUIT_REWARD/step
                    break
            food=max(food,value*hunger)
        risk=self.RISK_WEIGHT*max(0.,150.-minimum)
        return food-cost-risk, minimum

    def _decide_one(self, state, population):
        aid=state['agent_id']
        tracker=self.trackers.setdefault(aid, PredatorTracker(self))
        tracks=tracker.observe(state['observations'])
        fallback=super()._decide_one(state,population)
        fruits=sorted([o for o in state['observations'] if o.get('type')=='Fruit'],
                      key=lambda o:o['distance'])[:self.MAX_FRUITS]
        visible=[t for t in tracks if not t['missing']]
        action=fallback
        if (self.PLANNER_ENABLED and visible and fruits
                and state['energy'] < state['max_energy'] * self.FOOD_ENERGY_THRESHOLD):
            nearest=min(visible,key=lambda t:math.hypot(t['x'],t['y']))
            distance=math.hypot(nearest['x'],nearest['y'])
            if distance < self.CLOSE_FALLBACK_RADIUS:
                self.planner_counts['close_fallbacks']+=1
            else:
                self.planner_counts['planned']+=1
                angle=math.atan2(nearest['y'],nearest['x'])
                score,base_gap=self._score(fallback,state,tracks,fruits)
                directions=[f['angle'] for f in fruits]+[angle+math.pi,angle+math.pi/2,angle-math.pi/2]
                edges=[o for o in state['observations'] if o.get('type')=='Edge']
                candidates=[(0.,0.)]+[(state['speed'],d) for d in directions]
                # Exact fruit approach avoids paying for overshoot.
                candidates += [(min(state['speed'],max(0.,f['distance']-3.6)),f['angle']) for f in fruits]
                if state['energy']>state['max_energy']*.2:
                    candidates += [(min(state['sprint_speed'],16.5),angle+math.pi)]
                for move,direction in candidates:
                    direction=self._avoid_walls(direction,move,edges)
                    candidate=_base.ActionRequest(agent_id=aid,move_distance=move,
                        move_direction=_base._wrap(direction),turn_angle=angle,spawn_agent=False)
                    value,gap=self._score(candidate,state,tracks,fruits)
                    # Hard safety floor plus a relative bound against the incumbent.
                    if (gap >= max(self.HARD_SEPARATION,
                                   min(self.SOFT_SEPARATION, base_gap))
                            and value > score + self.IMPROVEMENT_MARGIN):
                        action,score=candidate,value
                if action is not fallback:
                    self.planner_counts['overrides']+=1
                    memory=self.memory[aid]
                    # Undo the incumbent's frame transform, then apply our action.
                    odx,ody=displacement(fallback,state)
                    ndx,ndy=displacement(action,state)
                    for name in ('tree','threat'):
                        point=getattr(memory,name)
                        if point is None:continue
                        x,y=rotate(point[0]*math.cos(point[1]),point[0]*math.sin(point[1]),fallback.turn_angle)
                        x,y=rotate(x+odx-ndx,y+ody-ndy,-action.turn_angle)
                        setattr(memory,name,(math.hypot(x,y),math.atan2(y,x)))
                    self._remember_prediction(memory,state['energy'],state['max_energy'],
                        state['speed'],state['sprint_speed'],action.move_distance,
                        action.turn_angle,action.spawn_agent,state['age'])
        tracker.advance_frame(action,state)
        return action
