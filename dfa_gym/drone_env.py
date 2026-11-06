""" DroneEnv: continuous point-mass agents flying in a bounded 3D box.

Action is a per-agent 3D velocity command (default) or, if
`use_displacement_action=True`, a direct 3D position-delta ("distance
change") command. No goal is defined here; wrap or subclass for that.

There is no geofence, as on the real drone: positions and velocities are never
clipped. Instead, the episode ends (for every agent) on the step any agent leaves
[low, high], and each agent that left gets `out_of_bounds_reward` (0 by default).
The labeled regions sit `region_margin` inside the box walls (in x and y), leaving
room to turn without leaving the box or touching a color, and neighboring regions
are `region_gap` apart.

If `use_velocity_filter=True`, velocity commands instead go through the deployed
drone's velocity filter: the action is normalized to [-1, 1] and scaled by
`max_speed` into a target velocity, which is held for the whole policy step `dt`,
split into ceil(dt / filter_dt) substeps, in each of which the acceleration
follows a critically-damped second-order law toward the target (vector-norm
limited to `max_filter_accel`) and is integrated into velocity and position.
Deployment equivalents: dt = POLICY_DT, filter_dt = FILTER_DT,
velocity_filter_tau = TAU (dfa_velocity_filter_tau_s), max_filter_accel =
MAX_ACCEL (dfa_max_filter_acceleration_mps2), max_speed = VELOCITY_SCALE
(dfa_velocity_scale).
"""
import jax
import math
import chex
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple, Dict
from dfa_gym import spaces
from dfa_gym.env import MultiAgentEnv, State


@struct.dataclass
class DroneEnvState(State):
    positions: jax.Array
    velocities: jax.Array
    accelerations: jax.Array  # Velocity filter state; stays zero when the filter is off.
    prev_positions: jax.Array  # Positions one step earlier, so label_f only fires on entering a region.
    time: int

class DroneEnv(MultiAgentEnv):

    def __init__(
        self,
        n_agents: int = 1,
        x_low: float = -1.0,
        x_high: float = 1.0,
        y_low: float = -1.0,
        y_high: float = 1.0,
        z_low: float = -1.0,
        z_high: float = 1.0,
        max_speed: float = 1.0,
        dt: float = 0.1,
        use_displacement_action: bool = False,
        use_velocity_filter: bool = False,
        velocity_filter_tau: float = 0.1,
        max_filter_accel: float = 3.0,
        filter_dt: float = 0.005,
        max_steps_in_episode: int = 200,
        region_margin: float = 0.25,
        region_gap: float = 0.1,
        out_of_bounds_reward: float = 0.0,
    ) -> None:
        assert not (use_displacement_action and use_velocity_filter), "the velocity filter only applies to velocity actions"
        assert region_margin >= 0 and region_gap >= 0, "region_margin and region_gap must be non-negative"
        super().__init__(num_agents=n_agents)
        self.n_agents = n_agents
        self.low = jnp.array([x_low, y_low, z_low], dtype=jnp.float32)
        self.high = jnp.array([x_high, y_high, z_high], dtype=jnp.float32)
        self.max_speed = max_speed
        self.dt = dt
        self.use_displacement_action = use_displacement_action
        self.use_velocity_filter = use_velocity_filter
        self.velocity_filter_tau = velocity_filter_tau
        self.max_filter_accel = max_filter_accel
        self.filter_dt = filter_dt
        self.n_substeps = math.ceil(dt / filter_dt)
        self.h = dt / self.n_substeps
        self.max_steps_in_episode = max_steps_in_episode
        self.region_margin = region_margin
        self.region_gap = region_gap
        self.out_of_bounds_reward = out_of_bounds_reward
        # A max-magnitude velocity command covers max_speed*dt of ground per step,
        # so cap displacement actions at that same per-step reach.
        self.max_action = max_speed if not use_displacement_action else max_speed * dt
        if use_velocity_filter:
            # As in deployment, filtered velocity actions are normalized and scaled by max_speed.
            self.max_action = 1.0

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        regions = self.label_regions()
        assert all(params[2] > 0 for _, kind, params in regions if kind == "circle"), \
            "region_margin/region_gap leave no room for the label regions; shrink them or enlarge the box"
        self.n_tokens = len(set(token for token, _, _ in regions))

        self.action_spaces = {
            agent: spaces.Box(low=-self.max_action, high=self.max_action, shape=(3,), dtype=jnp.float32)
            for agent in self.agents
        }
        # Nothing is clipped, so the last observation of an episode that ends by leaving
        # the box can lie up to one step's reach (max_speed * dt per axis) past a wall.
        reach = self.max_speed * self.dt
        self.observation_spaces = {
            agent: spaces.Box(
                low=jnp.concatenate([self.low - reach, -self.max_speed * jnp.ones(3, dtype=jnp.float32)]),
                high=jnp.concatenate([self.high + reach, self.max_speed * jnp.ones(3, dtype=jnp.float32)]),
                shape=(6,),
                dtype=jnp.float32,
            )
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], DroneEnvState]:
        positions = jnp.zeros((self.n_agents, 3), dtype=jnp.float32)
        velocities = jnp.zeros((self.n_agents, 3), dtype=jnp.float32)
        accelerations = jnp.zeros((self.n_agents, 3), dtype=jnp.float32)
        state = DroneEnvState(
            positions=positions, velocities=velocities, accelerations=accelerations, prev_positions=positions, time=0
        )
        return self.get_obs(state=state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: DroneEnvState,
        actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], DroneEnvState, Dict[str, float], Dict[str, bool], Dict]:
        action = jnp.stack([actions[agent] for agent in self.agents])
        action = jnp.clip(action, -self.max_action, self.max_action)
        if self.use_velocity_filter:
            positions, velocities, accelerations = self.filter_velocity(
                state.positions, state.velocities, state.accelerations, self.max_speed * action
            )
        else:
            delta = action if self.use_displacement_action else action * self.dt
            positions = state.positions + delta
            velocities = delta / self.dt
            accelerations = state.accelerations

        new_state = DroneEnvState(
            positions=positions, velocities=velocities, accelerations=accelerations,
            prev_positions=state.positions, time=state.time + 1
        )
        # Strict, so a drone exactly on a wall is still inside. Any agent leaving ends the
        # whole episode: done agents keep stepping until __all__, and nothing would stop
        # an out-of-box drone from drifting on.
        out_of_bounds = jnp.any((positions < self.low) | (positions > self.high), axis=-1)
        done = (new_state.time >= self.max_steps_in_episode) | jnp.any(out_of_bounds)

        rewards = {
            agent: jnp.where(out_of_bounds[i], self.out_of_bounds_reward, 0.0).astype(jnp.float32)
            for i, agent in enumerate(self.agents)
        }
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        obs = self.get_obs(state=new_state)
        info = {}

        return obs, new_state, rewards, dones, info

    def filter_velocity(
        self,
        positions: jax.Array,
        velocities: jax.Array,
        accelerations: jax.Array,
        target_velocities: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Advances (positions, velocities, accelerations) by one policy step `dt` of the
        deployed velocity filter, holding `target_velocities` constant throughout.

        Mirrors the deployment's `simulate_step` substep for substep (no geofence).
        """
        tau = self.velocity_filter_tau

        def substep(_, carry):
            p, v, a = carry
            jerk = (target_velocities - v) / tau**2 - 2.0 * a / tau
            a_next = a + self.h * jerk
            # Vector-norm acceleration limit, not a per-axis limit.
            magnitude = jnp.linalg.norm(a_next, axis=-1, keepdims=True)
            a_next = a_next * jnp.where(magnitude > self.max_filter_accel, self.max_filter_accel / magnitude, 1.0)
            v_next = v + self.h * a_next
            p_next = p + 0.5 * self.h * (v + v_next)
            return p_next, v_next, a_next

        return jax.lax.fori_loop(0, self.n_substeps, substep, (positions, velocities, accelerations))

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: DroneEnvState
    ) -> Dict[str, chex.Array]:
        return {
            agent: jnp.concatenate([state.positions[i], state.velocities[i]])
            for i, agent in enumerate(self.agents)
        }

    def label_regions(self):
        """Geometry of the regions labeled by `label_f`; also used to draw them.

        Every region is a 1.0-tall (z) prism. In x/y, the regions form a ring of cells
        along the walls of the inner box [low + region_margin, high - region_margin],
        five per edge (corner, corridor, edge midpoint, corridor, corner), each cell 2r
        wide with r = 0.1 * the inner box's shorter side (corridors stretch along a longer
        side). Each region is its cell shrunk by region_gap / 2 on every side, so
        neighboring regions are region_gap apart. The band between the walls and the
        inner box is unlabeled room to turn in.

        Returns a list of (token, kind, params) tuples in label-priority order (an
        earlier entry wins any overlap): kind "circle" -> params (cx, cy, r, z_lo, z_hi);
        kind "rect" -> params (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi).
        0: corners, 1: edge midpoints, 2-9: the corner<->midpoint corridors, each with
        its own token -- corner k's corridor along the vertical (left/right) edge is 2 + 2k
        and along the horizontal (top/bottom) edge is 3 + 2k, with corners k ordered
        (x_low, y_low), (x_low, y_high), (x_high, y_low), (x_high, y_high).
        """
        x_lo, y_lo = self.low[0] + self.region_margin, self.low[1] + self.region_margin
        x_hi, y_hi = self.high[0] - self.region_margin, self.high[1] - self.region_margin
        x_mid = 0.5 * (x_lo + x_hi)
        y_mid = 0.5 * (y_lo + y_hi)
        r = 0.1 * jnp.minimum(x_hi - x_lo, y_hi - y_lo)
        s = 0.5 * self.region_gap

        z_mid = 0.5 * (self.low[2] + self.high[2])
        z_lo, z_hi = z_mid - 0.75, z_mid + 0.25

        # Corner/edge-midpoint circles are centered one cell half-width in from the inner box walls.
        x_lo_in, x_hi_in = x_lo + r, x_hi - r
        y_lo_in, y_hi_in = y_lo + r, y_hi - r

        corners = [(x_lo_in, y_lo_in), (x_lo_in, y_hi_in), (x_hi_in, y_lo_in), (x_hi_in, y_hi_in)]
        edge_mids = [(x_mid, y_lo_in), (x_mid, y_hi_in), (x_lo_in, y_mid), (x_hi_in, y_mid)]
        # The vertical-edge then horizontal-edge corridor cell of each corner, in `corners` order.
        corner_cells = [
            (x_lo, x_lo + 2 * r, y_lo + 2 * r, y_mid - r),
            (x_lo + 2 * r, x_mid - r, y_lo, y_lo + 2 * r),
            (x_lo, x_lo + 2 * r, y_mid + r, y_hi - 2 * r),
            (x_lo + 2 * r, x_mid - r, y_hi - 2 * r, y_hi),
            (x_hi - 2 * r, x_hi, y_lo + 2 * r, y_mid - r),
            (x_mid + r, x_hi - 2 * r, y_lo, y_lo + 2 * r),
            (x_hi - 2 * r, x_hi, y_mid + r, y_hi - 2 * r),
            (x_mid + r, x_hi - 2 * r, y_hi - 2 * r, y_hi),
        ]
        corner_rects = [(a + s, b - s, c + s, d - s) for a, b, c, d in corner_cells]

        regions = [(0, "circle", (cx, cy, r - s, z_lo, z_hi)) for cx, cy in corners]
        regions += [(1, "circle", (mx, my, r - s, z_lo, z_hi)) for mx, my in edge_mids]
        regions += [(2 + i, "rect", bounds + (z_lo, z_hi)) for i, bounds in enumerate(corner_rects)]
        return regions

    @partial(jax.jit, static_argnums=(0,))
    def label_f(
        self,
        state: DroneEnvState
    ) -> Dict[str, int]:
        # Labels the agent's full (x, y, z) position -- each region is a 1.0-tall prism -- before
        # and after the last step: a token fires only on the step its region is entered, not on
        # every step the agent stays inside it.
        positions = jnp.stack([state.prev_positions, state.positions])
        x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
        labels = jnp.full(x.shape, -1, dtype=jnp.int32)
        for token, kind, params in reversed(self.label_regions()):
            if kind == "circle":
                cx, cy, r, z_lo, z_hi = params
                mask = ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2) & (z >= z_lo) & (z <= z_hi)
            else:
                x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = params
                mask = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi) & (z >= z_lo) & (z <= z_hi)
            labels = jnp.where(mask, token, labels)
        labels = jnp.where(labels[1] != labels[0], labels[1], -1)

        return {agent: labels[i] for i, agent in enumerate(self.agents)}
