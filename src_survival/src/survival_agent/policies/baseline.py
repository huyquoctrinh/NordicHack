import math
import random
from typing import Dict, List, Optional, Tuple

from survival_agent.dto import ActionRequest

"""
Hivemind policy for the survival simulator.

Everything below is derived from the simulator source rather than the README,
because the two disagree in one important place:

* ``move_direction`` is RELATIVE to the agent's current heading. The
  environment computes ``direction = entity.direction + move_direction``
  (Environment.update_entity_position), so movement is decoupled from facing.
  An agent can walk away from a predator while still looking straight at it.
* A predator only commits to a chase when ``abs(agent_looking_dir) > pi/2`` or
  when it is closer than ``hearing_radius * 1.5 == 90`` (Predator.step). That
  ``agent_looking_dir`` is the same quantity the agent sees as its own bearing
  to the predator, so keeping a predator in the forward half-plane at range
  makes it circle instead of attack.
* Score is ``+dt`` every tick regardless of population, ``+fruit.energy/1000``
  per fruit and ``-agent.energy/100`` per agent eaten. Survival time dominates,
  so the species must never go extinct, and agents should not die rich.
* Agents die of old age: past ``max_age`` (60-120s, not observable) they lose an
  extra ``0.01 * age`` energy per tick, which is ~10x the living cost. Long runs
  are therefore only possible through continuous reproduction.
"""

TWO_PI = 2.0 * math.pi
DT = 0.1  # simulation seconds per tick

# --- Predator constants, read off src/elements/predator.py ---
PRED_CHASE_RADIUS = 90.0   # predator.hearing_radius * 1.5: inside this it always charges
PRED_SAFE_RADIUS = 150.0   # keep this much air between us and the nearest predator
PRED_ALERT_RADIUS = 230.0  # start reacting this early
PRED_CLOSING_SPEED = 15.0  # predator.sprint_speed, used to dead-reckon an unseen chaser
THREAT_MEMORY_TICKS = 25   # keep fleeing this long after losing sight of a predator

# --- Energy costs, read off Environment.update_entity_position ---
WALK_COST = 0.05
SPRINT_COST = 0.5
LIVING_COST = 0.1          # per tick, every biome has energy_drain_rate 1.0

# --- Reproduction thresholds ---
# The environment refuses to sprint an agent whose energy is below
# max_energy / 5 (update_entity_position). An agent that cannot sprint cannot
# outrun a predator, so that band is treated as a hard reserve: reproduction is
# only ever paid for out of energy above it.
SPRINT_RESERVE_FRACTION = 0.2
SPAWN_COST = 100.0
SPAWN_FLOOR = 104.0        # env requires energy > 100 at spawn time
SPAWN_MARGIN = 25.0        # keep this much clear of the sprint reserve after spawning
MIN_POP = 12               # below this, rebuild as fast as energy allows
TARGET_POP = 26            # above this, only the genuinely rich add another mouth

# --- Foraging ---
FORAGE_THRESHOLD = 0.55    # collect fruit only below this fraction of max_energy
EAT_RADIUS = 9.0           # agent.size + fruit.radius, both 5 at spawn
ENABLE_CAMP = True         # False makes agents roam continuously instead of settling
CAMP_RADIUS = 14.0         # how close to sit to a tree we are camping on
TREE_CHASE_RANGE = 320.0   # ignore trees further away than this
IDLE_SCAN_TURN = 0.13      # radians per tick while camping, to sweep the vision cone
WALL_REPULSE_DIST = 34.0
CROWD_RADIUS = 70.0        # another agent this close to a tree is competition for it
CROWD_PENALTY = 70.0       # distance-equivalent cost of each rival already camped there
CAMP_PATIENCE = 220        # ticks of fruitless camping before abandoning a tree
ROAM_TICKS = 160           # ticks spent relocating before settling again

# --- Old-age detection ---
OLD_AGE_MIN = 55.0         # max_age is uniform on [60, 120]
OLD_AGE_SLACK = 0.22       # unexplained drain above this means the penalty kicked in

MEMORY_TTL = 90            # ticks a remembered tree stays trusted


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % TWO_PI - math.pi


def _move_cost(distance: float, speed: float, sprint_speed: float,
               energy: float, max_energy: float) -> float:
    """Mirror of the environment's movement charge, used to predict our own energy."""
    distance = max(0.0, min(distance, sprint_speed))
    if energy < max_energy / 5 and distance > speed:
        distance = speed
    if distance <= speed:
        return distance * WALK_COST
    return speed * WALK_COST + (distance - speed) * SPRINT_COST


def _turn_cost(angle: float) -> float:
    """Mirror of Environment.update_entity_direction."""
    return min(math.pi, abs(angle)) / TWO_PI


class _Memory:
    """Per-agent scratch state carried between ticks."""

    __slots__ = ("tree", "tree_ttl", "old", "prev_energy", "predicted", "wander_ttl",
                 "threat", "threat_ttl", "hungry_ticks")

    def __init__(self) -> None:
        self.tree: Optional[Tuple[float, float]] = None  # (distance, bearing) in the agent's frame
        self.tree_ttl: int = 0
        self.old: bool = False
        self.prev_energy: Optional[float] = None
        self.predicted: Optional[float] = None
        self.wander_ttl: int = 0
        self.threat: Optional[Tuple[float, float]] = None  # (distance, bearing)
        self.threat_ttl: int = 0
        self.hungry_ticks: int = 0


class HivemindPolicy:
    """
    Controls the whole species. ``decide`` takes the full agent_status list so
    decisions that depend on the colony as a whole -- chiefly how hard to
    reproduce -- can use the real population count.
    """

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.memory: Dict[int, _Memory] = {}
        self.last_sim_time = -1.0
        self.mode_counts: Dict[str, int] = {}

    def reset(self) -> None:
        """Drop all per-agent state. Called when a fresh simulation is detected."""
        self.memory.clear()
        self.rng.seed(self.seed)
        self.mode_counts.clear()

    def decide(self, agent_status: List[dict], sim_time: float = 0.0) -> List[ActionRequest]:
        """
        Args:
            agent_status (list): One observation dict per living agent.
            sim_time (float): Simulated seconds elapsed.

        Returns:
            list[ActionRequest]: One action per agent, in the same order.
        """
        # The evaluation runs three simulations back to back against one server,
        # and agent ids restart at 0 each time, so stale memory must be dropped.
        if sim_time < self.last_sim_time - 1e-9:
            self.reset()
        self.last_sim_time = sim_time

        alive = {status["agent_id"] for status in agent_status}
        for agent_id in [k for k in self.memory if k not in alive]:
            del self.memory[agent_id]

        population = len(agent_status)
        return [self._decide_one(status, population) for status in agent_status]

    # ------------------- per-agent decision -------------------

    def _decide_one(self, status: dict, population: int) -> ActionRequest:
        agent_id = status["agent_id"]
        energy = status["energy"]
        max_energy = status["max_energy"]
        speed = status["speed"]
        sprint_speed = status["sprint_speed"]
        age = status["age"]
        biome = status.get("biome", "")

        memory = self.memory.get(agent_id)
        if memory is None:
            memory = _Memory()
            self.memory[agent_id] = memory

        self._detect_old_age(memory, energy, age)
        self._track_hunger(memory, energy)

        fruits, predators, others, trees, edges = self._split(status["observations"])
        self._age_memory(memory, trees)

        # 1. Staying alive beats everything else.
        threat = self._track_threat(memory, predators)
        if threat is not None:
            self._count("evade")
            move_distance, move_direction, turn_angle = self._evade(
                threat, speed, sprint_speed, energy, max_energy
            )
            # Pay for only enough speed to open the gap. In slow terrain the
            # maximum command may still be necessary; cheap walking is enough
            # for a fast lineage on normal ground.
            if threat["distance"] < PRED_CHASE_RADIUS:
                penalty = {"swamp": .5, "desert": .8, "river": .3}.get(biome, 1.0)
                required = (PRED_CLOSING_SPEED + 1.5) / penalty
                if speed >= required:
                    move_distance = min(speed, sprint_speed)
                elif energy > max_energy * SPRINT_RESERVE_FRACTION and threat.get("visible", True):
                    move_distance = min(sprint_speed, required)
            spawn = False  # never make a 75-energy newborn next to a predator
        else:
            move_distance, move_direction, turn_angle = self._forage(
                memory, fruits, others, trees, speed, biome, energy, max_energy
            )
            spawn = self._want_spawn(memory, energy, max_energy, population)

        # 2. Steer clear of walls; collisions still charge full price for the move.
        move_direction = self._avoid_walls(move_direction, move_distance, edges)

        # Remember positions in the NEXT observation frame. Movement precedes
        # rotation, and biomes scale displacement but not the energy charge.
        displacement = min(move_distance, sprint_speed)
        if energy < max_energy * SPRINT_RESERVE_FRACTION:
            displacement = min(displacement, speed)
        displacement *= {"swamp": .5, "desert": .8, "river": .3}.get(biome, 1.0)
        if memory.tree is not None:
            memory.tree = self._transform_point(memory.tree, displacement,
                                                 move_direction, turn_angle)
        if threat is not None:
            memory.threat = self._transform_point(
                (threat["distance"], threat["angle"]), displacement,
                move_direction, turn_angle)

        self._remember_prediction(
            memory, energy, max_energy, speed, sprint_speed,
            move_distance, turn_angle, spawn, age,
        )

        return ActionRequest(
            agent_id=agent_id,
            move_distance=move_distance,
            move_direction=move_direction,
            turn_angle=turn_angle,
            spawn_agent=spawn,
        )

    @staticmethod
    def _transform_point(point, distance, direction, turn):
        radius, bearing = point
        x = radius * math.cos(bearing) - distance * math.cos(direction)
        y = radius * math.sin(bearing) - distance * math.sin(direction)
        return math.hypot(x, y), _wrap(math.atan2(y, x) - turn)

    # ------------------- behaviours -------------------

    def _track_threat(self, memory: _Memory, predators: List[dict]) -> Optional[dict]:
        """
        Nearest predator, remembered for a short while after it leaves perception.

        This matters because a predator's preferred approach is exactly the one
        an agent cannot see: the vision cone is only 60 degrees wide and hearing
        reaches 50 units, so anything chasing from behind is invisible until it
        is almost touching. Forgetting a predator the instant it slips out of the
        cone makes an agent turn straight back into it.
        """
        if predators:
            nearest = min(predators, key=lambda o: o["distance"])
            if nearest["distance"] < PRED_ALERT_RADIUS:
                memory.threat = (nearest["distance"], nearest["angle"])
                memory.threat_ttl = THREAT_MEMORY_TICKS
                return {"distance": nearest["distance"], "angle": nearest["angle"],
                        "visible": True}

        if memory.threat_ttl > 0 and memory.threat is not None:
            memory.threat_ttl -= 1
            distance, angle = memory.threat
            return {"distance": distance, "angle": angle, "visible": False}

        memory.threat = None
        return None

    def _evade(self, threat: dict, speed: float, sprint_speed: float,
               energy: float, max_energy: float):
        """
        Back away from a predator while turning to face it.

        Facing it flips Predator.step out of its chase branch into the pivot
        branch, but only while the gap stays above 90 units -- and the pivot
        still closes ground -- so the retreat matters as much as the stare.
        """
        distance, angle = threat["distance"], threat["angle"]

        away = _wrap(angle + math.pi)  # directly away, in our own heading frame
        turn_angle = _wrap(angle)      # point our nose at it for the next tick

        # The environment silently downgrades a sprint to a walk below 20% energy,
        # so only ask for one when it will actually be granted. A sprint also
        # costs ten times a walk per unit, so it is only ever spent on a predator
        # we can actually see -- never on a remembered one, whose position is a
        # guess that drifts closer every tick it goes unconfirmed.
        can_sprint = (energy > max_energy * SPRINT_RESERVE_FRACTION
                      and threat.get("visible", True))

        if distance < PRED_CHASE_RADIUS:
            # Committed chase. Agent sprint speed is 20 against the predator's 15,
            # so running is the only thing that actually breaks contact.
            move_distance = sprint_speed if can_sprint else speed
        elif distance < PRED_SAFE_RADIUS:
            move_distance = speed
        else:
            # Far enough that facing it is the whole defence: it will circle
            # rather than close. Standing still costs nothing but the turn, and
            # energy in the tank is what keeps the sprint option available.
            move_distance = 0.0

        return move_distance, away, turn_angle

    @staticmethod
    def _dead_reckon_threat(memory: _Memory, threat: dict,
                            move_distance: float, turn_angle: float) -> None:
        """
        Carry the threat's bearing forward through our own move and turn.

        An unconfirmed predator is assumed to hold station rather than to keep
        closing: crediting it with its full sprint every tick manufactures a
        phantom that crosses the panic threshold on its own and triggers a real,
        expensive sprint away from nothing.
        """
        closing = PRED_CLOSING_SPEED if threat.get("visible", True) else 0.0
        distance = threat["distance"] + move_distance - closing
        memory.threat = (max(distance, 0.0), _wrap(threat["angle"] - turn_angle))

    def _forage(self, memory: _Memory, fruits: List[dict], others: List[dict],
                trees: List[dict], speed: float, biome: str,
                energy: float, max_energy: float):
        """Eat, else camp a tree, else explore."""
        # Fruit spawns at 20 energy and ripens at 2/s to a cap of 60, rotting only
        # at 50s. Eating it the instant it appears throws away two thirds of a
        # tree's output, so a well-fed agent leaves it on the branch and collects
        # a ripe batch once its own reserves have run down.
        hungry = energy < max_energy * FORAGE_THRESHOLD

        # "This camp is barren" has to mean no fruit is appearing, not that we
        # chose to let what is there ripen -- otherwise a well-fed agent walks
        # away from the one tree that is actually feeding it.
        if fruits:
            memory.hungry_ticks = 0
        else:
            memory.hungry_ticks += 1
            if memory.hungry_ticks > CAMP_PATIENCE + ROAM_TICKS:
                memory.hungry_ticks = 0  # settle wherever the roam has taken us
                memory.tree = None
                memory.tree_ttl = 0

        target = self._pick_fruit(fruits, others) if hungry else None
        if target is not None:
            self._count("fruit")
            distance, angle = target["distance"], target["angle"]
            # Full walking speed or the exact gap, whichever is smaller. Energy per
            # unit distance is flat, so dawdling only adds living cost.
            move_distance = min(speed, max(distance - EAT_RADIUS * 0.4, 0.0))
            return move_distance, _wrap(angle), _wrap(angle) * 0.35

        # Rivers grow nothing and cost 70% of movement, so never settle in one.
        # A camp that has produced nothing for a while is a dying or barren tree,
        # so give up on it rather than starving politely next to it.
        if ENABLE_CAMP and biome != "river" and memory.hungry_ticks < CAMP_PATIENCE:
            tree = self._pick_tree(memory, trees, others)
            if tree is not None:
                distance, angle = tree
                if distance <= CAMP_RADIUS:
                    # Sit on the tree. Fruit spawns within 3x the trunk radius, i.e.
                    # at most 60 units, and hearing covers 50 of that without moving.
                    self._count("camp")
                    return 0.0, 0.0, IDLE_SCAN_TURN
                self._count("to_tree")
                return min(speed, distance), _wrap(angle), _wrap(angle) * 0.4

        self._count("explore")
        return self._explore(memory, others, speed)

    def _explore(self, memory: _Memory, others: List[dict], speed: float):
        """
        Walk a straight line, changing course occasionally, biased away from the
        rest of the colony so that new ground actually gets covered.
        """
        turn_angle = 0.0
        if memory.wander_ttl <= 0:
            turn_angle = self.rng.uniform(-0.9, 0.9)
            if others:
                # Head away from the local crowd: whatever they are standing on
                # has already been picked over.
                cx = sum(o["distance"] * math.cos(o["angle"]) for o in others) / len(others)
                cy = sum(o["distance"] * math.sin(o["angle"]) for o in others) / len(others)
                turn_angle = _wrap(math.atan2(cy, cx) + math.pi) * 0.7
            memory.wander_ttl = self.rng.randint(40, 90)
        memory.wander_ttl -= 1
        return speed, 0.0, turn_angle

    def _pick_fruit(self, fruits: List[dict], others: List[dict]) -> Optional[dict]:
        """
        Nearest fruit that no other agent is obviously about to take. Two agents
        walking onto the same berry means one of them paid for nothing.
        """
        if not fruits:
            return None

        rivals = [
            (o["distance"] * math.cos(o["angle"]), o["distance"] * math.sin(o["angle"]))
            for o in others
        ]

        best = None
        best_distance = float("inf")
        for fruit in fruits:
            distance = fruit["distance"]
            if distance >= best_distance:
                continue
            if rivals:
                fx = distance * math.cos(fruit["angle"])
                fy = distance * math.sin(fruit["angle"])
                if any(math.hypot(fx - rx, fy - ry) < distance - 6.0 for rx, ry in rivals):
                    continue  # somebody else is closer to it than we are
            best, best_distance = fruit, distance

        # If every fruit is contested, still go for the closest rather than idle.
        if best is None:
            best = min(fruits, key=lambda f: f["distance"])
        return best

    def _pick_tree(self, memory: _Memory, trees: List[dict],
                   others: List[dict]) -> Optional[Tuple[float, float]]:
        """
        Pick a tree to camp on, preferring empty ones.

        A tree yields roughly one fruit every ten seconds, about 4 energy/s,
        against a 1 energy/s living cost. That comfortably feeds one agent and
        starves five. Children spawn 10-30 units from their parent, so without an
        explicit push the whole colony inherits one tree and slowly starves on it
        while most of the map's trees go unvisited.
        """
        if trees:
            nearest = min(trees, key=lambda t: t["distance"])
            if nearest["distance"] <= CAMP_RADIUS and memory.hungry_ticks < CAMP_PATIENCE:
                # Already settled and still being fed. Re-shopping every tick makes
                # a newborn walk away from its parent's working tree on 75 energy,
                # and often pushes the parent off it in the same breath.
                memory.tree = (nearest["distance"], nearest["angle"])
                memory.tree_ttl = MEMORY_TTL
                return memory.tree

            rivals = [
                (o["distance"] * math.cos(o["angle"]), o["distance"] * math.sin(o["angle"]))
                for o in others
            ]
            best = None
            best_cost = float("inf")
            for tree in trees:
                distance = tree["distance"]
                if distance > TREE_CHASE_RANGE:
                    continue
                tx = distance * math.cos(tree["angle"])
                ty = distance * math.sin(tree["angle"])
                crowd = sum(1 for rx, ry in rivals if math.hypot(tx - rx, ty - ry) < CROWD_RADIUS)
                cost = distance + crowd * CROWD_PENALTY
                if cost < best_cost:
                    best, best_cost = tree, cost
            if best is not None:
                memory.tree = (best["distance"], best["angle"])
                memory.tree_ttl = MEMORY_TTL
                return memory.tree
        if memory.tree is not None and memory.tree_ttl > 0:
            return memory.tree
        return None

    def _avoid_walls(self, move_direction: float, move_distance: float,
                     edges: List[dict]) -> float:
        """Bend the movement direction away from any edge we are about to hug."""
        if not edges or move_distance <= 0.0:
            return move_direction

        closest = None
        closest_distance = float("inf")
        for edge in edges:
            point = _closest_point_on_segment(edge["coords"])
            distance = math.hypot(point[0], point[1])
            if distance < closest_distance:
                closest, closest_distance = point, distance

        if closest is None or closest_distance > WALL_REPULSE_DIST:
            return move_direction

        wall_angle = math.atan2(closest[1], closest[0])
        if abs(_wrap(wall_angle - move_direction)) > math.pi / 2:
            return move_direction  # already heading away from it

        # Push perpendicular, harder the closer we are.
        strength = 1.0 - closest_distance / WALL_REPULSE_DIST
        sign = -1.0 if _wrap(wall_angle - move_direction) >= 0 else 1.0
        return _wrap(move_direction + sign * strength * math.pi / 2)

    # ------------------- reproduction -------------------

    def _want_spawn(self, memory: _Memory, energy: float, max_energy: float,
                    population: int) -> bool:
        """
        Reproduction is the only answer to old age, and it doubles as insurance:
        energy carried by an agent that gets eaten becomes a score penalty, while
        energy converted into a child does not.

        The hard constraint is that spawning must never drop the parent into the
        band where the environment forbids sprinting, because an agent that
        cannot sprint cannot escape a predator.
        """
        if energy <= SPAWN_FLOOR:
            return False

        if memory.old:
            # Dying anyway, and the reserve buys nothing: energy left in the
            # corpse is simply deleted, so convert all of it into children.
            return True

        # Holding the sprint reserve keeps an individual alive, but a colony that
        # only ever breeds from surplus stops replacing itself and dies of old age
        # instead. When numbers are thin the species outranks the individual.
        reserve = max_energy * SPRINT_RESERVE_FRACTION
        if population < MIN_POP:
            return energy > SPAWN_COST + 60.0

        if energy - SPAWN_COST <= reserve:
            return False
        if population < TARGET_POP:
            return energy - SPAWN_COST > reserve + SPAWN_MARGIN
        return energy > max_energy * 0.75

    # ------------------- bookkeeping -------------------

    def _detect_old_age(self, memory: _Memory, energy: float, age: float) -> None:
        """
        ``max_age`` is not observable, so infer it. Past it the agent loses an
        extra ``0.01 * age`` per tick, which is far larger than the noise in our
        own cost prediction, so an unexplained drop is a reliable signal.
        """
        if memory.old or age < OLD_AGE_MIN:
            memory.prev_energy = energy
            return
        if memory.predicted is not None and memory.prev_energy is not None:
            # A rise means fruit was eaten this tick, which masks the penalty.
            if energy < memory.prev_energy and energy < memory.predicted - OLD_AGE_SLACK:
                memory.old = True
        memory.prev_energy = energy

    def _remember_prediction(self, memory: _Memory, energy: float, max_energy: float,
                             speed: float, sprint_speed: float, move_distance: float,
                             turn_angle: float, spawn: bool, age: float) -> None:
        """Predict next tick's energy assuming we are NOT yet old."""
        remaining = energy
        remaining -= _move_cost(move_distance, speed, sprint_speed, energy, max_energy)
        remaining -= _turn_cost(turn_angle)
        if spawn and remaining > 100.0:
            remaining -= 100.0
        memory.predicted = remaining - LIVING_COST

    def _track_hunger(self, memory: _Memory, energy: float) -> None:
        """
        Count ticks since the last meal, using the energy prediction to spot one.

        A camp that stops producing is either a dying tree or one in a barren
        biome; either way the answer is to leave, roam for a while and settle
        somewhere new rather than wait it out.
        """
        if memory.predicted is not None and energy > memory.predicted + 1.0:
            memory.hungry_ticks = 0
            memory.tree_ttl = MEMORY_TTL  # productive, worth staying on

    def _age_memory(self, memory: _Memory, trees: List[dict]) -> None:
        if memory.tree_ttl > 0:
            memory.tree_ttl -= 1
        if memory.tree_ttl == 0:
            memory.tree = None

    def _count(self, mode: str) -> None:
        """Behaviour telemetry, read by the offline tuning harness."""
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1

    @staticmethod
    def _split(observations: List[dict]):
        """Bucket one agent's observations by type."""
        fruits, predators, agents, trees, edges = [], [], [], [], []
        for observation in observations:
            kind = observation.get("type")
            if kind == "Fruit":
                fruits.append(observation)
            elif kind == "Predator":
                predators.append(observation)
            elif kind == "Agent":
                agents.append(observation)
            elif kind == "Tree":
                trees.append(observation)
            elif kind == "Edge":
                edges.append(observation)
        return fruits, predators, agents, trees, edges


def _closest_point_on_segment(coords) -> Tuple[float, float]:
    """Closest point to the agent (the origin of this frame) on an edge segment."""
    (x1, y1), (x2, y2) = coords
    dx, dy = x2 - x1, y2 - y1
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return x1, y1
    t = -(x1 * dx + y1 * dy) / denominator
    t = max(0.0, min(1.0, t))
    return x1 + t * dx, y1 + t * dy
