""" DroneEnv: continuous point-mass agents confined to a bounded 3D box.

Action is a per-agent 3D velocity command (default) or, if
`use_displacement_action=True`, a direct 3D position-delta ("distance
change") command. No reward/goal is defined here; wrap or subclass for that.

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
    ) -> None:
        assert not (use_displacement_action and use_velocity_filter), "the velocity filter only applies to velocity actions"
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
        # A max-magnitude velocity command covers max_speed*dt of ground per step,
        # so cap displacement actions at that same per-step reach.
        self.max_action = max_speed if not use_displacement_action else max_speed * dt
        if use_velocity_filter:
            # As in deployment, filtered velocity actions are normalized and scaled by max_speed.
            self.max_action = 1.0

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.n_tokens = len(set(token for token, _, _ in self.label_regions()))

        self.action_spaces = {
            agent: spaces.Box(low=-self.max_action, high=self.max_action, shape=(3,), dtype=jnp.float32)
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Box(
                low=jnp.concatenate([self.low, -self.max_speed * jnp.ones(3, dtype=jnp.float32)]),
                high=jnp.concatenate([self.high, self.max_speed * jnp.ones(3, dtype=jnp.float32)]),
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
            positions = jnp.clip(state.positions + delta, self.low, self.high)
            # Achieved velocity, not the commanded one: zero along any axis pinned by the geofence.
            velocities = (positions - state.positions) / self.dt
            accelerations = state.accelerations

        new_state = DroneEnvState(
            positions=positions, velocities=velocities, accelerations=accelerations,
            prev_positions=state.positions, time=state.time + 1
        )
        done = new_state.time >= self.max_steps_in_episode

        rewards = {agent: jnp.zeros((), dtype=jnp.float32) for agent in self.agents}
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

        Mirrors the deployment's `simulate_step` substep for substep, plus a geofence
        the deployment doesn't have: an axis at (or pushed past) a wall while moving
        into it is pinned at the wall and its velocity and acceleration are zeroed.
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

            # An axis at (or pushed past) a wall while moving into it is pinned there.
            pinned = ((p_next <= self.low) & (v_next < 0)) | ((p_next >= self.high) & (v_next > 0))
            p_next = jnp.clip(p_next, self.low, self.high)
            return p_next, jnp.where(pinned, 0.0, v_next), jnp.where(pinned, 0.0, a_next)

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

        Every region is a 1.0-tall (z) prism, inset so it lies fully inside
        [low, high]. Returns a list of (token, kind, params) tuples in label-priority
        order (an earlier entry wins any overlap): kind "circle" -> params
        (cx, cy, r, z_lo, z_hi); kind "rect" -> params (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi).
        0: corners, 1: edge midpoints, 2-9: the corner<->midpoint corridors, each with
        its own token -- corner k's corridor along the vertical (left/right) edge is 2 + 2k
        and along the horizontal (top/bottom) edge is 3 + 2k, with corners k ordered
        (x_low, y_low), (x_low, y_high), (x_high, y_low), (x_high, y_high).
        """
        x_low, y_low = self.low[0], self.low[1]
        x_high, y_high = self.high[0], self.high[1]
        x_mid = 0.5 * (x_low + x_high)
        y_mid = 0.5 * (y_low + y_high)
        r = 0.1 * jnp.minimum(x_high - x_low, y_high - y_low)

        z_mid = 0.5 * (self.low[2] + self.high[2])
        z_lo, z_hi = z_mid - 0.75, z_mid + 0.25

        # Corner/edge-midpoint circles are inset by r so they sit fully inside [low, high]
        # instead of being centered on the boundary itself.
        x_lo_in, x_hi_in = x_low + r, x_high - r
        y_lo_in, y_hi_in = y_low + r, y_high - r

        corners = [(x_lo_in, y_lo_in), (x_lo_in, y_hi_in), (x_hi_in, y_lo_in), (x_hi_in, y_hi_in)]
        edge_mids = [(x_mid, y_lo_in), (x_mid, y_hi_in), (x_lo_in, y_mid), (x_hi_in, y_mid)]
        # The vertical-edge then horizontal-edge corridor of each corner, in `corners` order.
        corner_rects = [
            (x_low, x_low + 2 * r, y_low + 2 * r, y_mid - r),
            (x_low + 2 * r, x_mid - r, y_low, y_low + 2 * r),
            (x_low, x_low + 2 * r, y_mid + r, y_high - 2 * r),
            (x_low + 2 * r, x_mid - r, y_high - 2 * r, y_high),
            (x_high - 2 * r, x_high, y_low + 2 * r, y_mid - r),
            (x_mid + r, x_high - 2 * r, y_low, y_low + 2 * r),
            (x_high - 2 * r, x_high, y_mid + r, y_high - 2 * r),
            (x_mid + r, x_high - 2 * r, y_high - 2 * r, y_high),
        ]

        regions = [(0, "circle", (cx, cy, r, z_lo, z_hi)) for cx, cy in corners]
        regions += [(1, "circle", (mx, my, r, z_lo, z_hi)) for mx, my in edge_mids]
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
