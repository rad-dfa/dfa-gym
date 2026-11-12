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

Observation noise is off by default. If `obs_noise_std` > 0 (a single std, or a
(position_std, velocity_std) pair), every observation, at reset and at every step,
gets fresh zero-mean Gaussian noise truncated to +-3 std, drawn independently per
agent. The noise only changes what the policy sees: the dynamics, `label_f`, and
the out-of-bounds check all use the true state.

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

If, on top of the velocity filter, `use_tracking_controller=True`, the filter no
longer moves the drone directly: its (position, velocity, acceleration) is the
reference that the stack's controller tracks, which emulates controller tracking
error (the filter smooths the command; tracking adds the controller's lag). In each
filter substep h, after the reference advances, the controller requests the
per-axis acceleration
    request = ref_a + tracking_kp * (ref_p - p) + tracking_kv * (ref_v - v),
a first-order motor delay moves the drone's acceleration toward it,
    a += (1 - exp(-h / motor_tau)) * (request - a),
and the point mass integrates it (v += h * a, trapezoidal p). The defaults are the
stack's controller position gain (KP), velocity gain (KV) and motor delay (TAU), in
(x, y, z) order. The reference lives in `ref_positions`/`ref_velocities`/
`ref_accelerations` (zero unless this mode is on) and starts at the drone, while
`positions`/`velocities`/`accelerations` are the drone's own, so observations,
`label_f` and the out-of-bounds check all see the tracked drone. Its velocity can
overshoot the reference's, by at most `vel_gain` per axis (see
`tracking_velocity_gain`), which widens the observation space to match.
"""
import jax
import math
import chex
import numpy as np
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
    accelerations: jax.Array  # Velocity filter state (tracked drone's with the tracking controller); stays zero when the filter is off.
    prev_positions: jax.Array  # Positions one step earlier, so label_f only fires on entering a region.
    time: int
    # Velocity filter state that the tracking controller follows; stays zero unless use_tracking_controller.
    ref_positions: jax.Array
    ref_velocities: jax.Array
    ref_accelerations: jax.Array

class DroneEnv(MultiAgentEnv):

    # Observation noise is a standard normal truncated to this many stds, so noisy
    # observations stay within the (widened) declared observation space.
    OBS_NOISE_MAX_SIGMAS = 3.0

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
        use_tracking_controller: bool = False,
        tracking_kp: Tuple[float, float, float] = (7.1, 7.1, 11.9),
        tracking_kv: Tuple[float, float, float] = (2.4, 2.4, 6.7),
        motor_tau: Tuple[float, float, float] = (0.080019, 0.080019, 0.058938),
        max_steps_in_episode: int = 200,
        region_margin: float = 0.25,
        region_gap: float = 0.1,
        out_of_bounds_reward: float = 0.0,
        obs_noise_std: float | Tuple[float, float] = 0.0,
    ) -> None:
        assert not (use_displacement_action and use_velocity_filter), "the velocity filter only applies to velocity actions"
        assert use_velocity_filter or not use_tracking_controller, "the tracking controller tracks the velocity filter; set use_velocity_filter=True"
        assert all(len(x) == 3 for x in (tracking_kp, tracking_kv, motor_tau)), "tracking_kp, tracking_kv and motor_tau are per-axis (x, y, z)"
        assert all(t > 0 for t in motor_tau), "motor_tau must be positive"
        assert region_margin >= 0 and region_gap >= 0, "region_margin and region_gap must be non-negative"
        if isinstance(obs_noise_std, (int, float)):
            pos_noise_std = vel_noise_std = float(obs_noise_std)
        else:
            pos_noise_std, vel_noise_std = (float(std) for std in obs_noise_std)
        assert pos_noise_std >= 0 and vel_noise_std >= 0, "obs_noise_std must be non-negative"
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
        self.use_tracking_controller = use_tracking_controller
        self.tracking_kp = tuple(float(k) for k in tracking_kp)
        self.tracking_kv = tuple(float(k) for k in tracking_kv)
        self.motor_tau = tuple(float(t) for t in motor_tau)
        self.kp = jnp.array(self.tracking_kp, dtype=jnp.float32)
        self.kv = jnp.array(self.tracking_kv, dtype=jnp.float32)
        # Fraction of the gap to the requested acceleration the motors close in one substep.
        self.motor_alpha = jnp.array([1.0 - math.exp(-self.h / t) for t in self.motor_tau], dtype=jnp.float32)
        # Per-axis bound on |velocity| / max_speed: the tracked drone can overshoot its reference.
        self.vel_gain = self.tracking_velocity_gain() if use_tracking_controller else np.ones(3)
        self.max_steps_in_episode = max_steps_in_episode
        self.region_margin = region_margin
        self.region_gap = region_gap
        self.out_of_bounds_reward = out_of_bounds_reward
        self.obs_noise_std = obs_noise_std
        self.obs_noise_scale = jnp.array([pos_noise_std] * 3 + [vel_noise_std] * 3, dtype=jnp.float32)
        self.use_obs_noise = pos_noise_std > 0 or vel_noise_std > 0
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
        # the box can lie up to one step's reach (max_speed * dt per axis, times vel_gain
        # with the tracking controller) past a wall. Observation noise widens both bounds
        # by its maximum magnitude.
        max_velocity = self.max_speed * self.vel_gain  # float64, cast to float32 only once below
        reach = max_velocity * self.dt
        pos_slack = jnp.asarray(reach + self.OBS_NOISE_MAX_SIGMAS * pos_noise_std, dtype=jnp.float32)
        vel_bound = jnp.asarray(max_velocity + self.OBS_NOISE_MAX_SIGMAS * vel_noise_std, dtype=jnp.float32)
        self.observation_spaces = {
            agent: spaces.Box(
                low=jnp.concatenate([self.low - pos_slack, -vel_bound]),
                high=jnp.concatenate([self.high + pos_slack, vel_bound]),
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
        # The tracking controller's reference starts at the drone.
        state = DroneEnvState(
            positions=positions, velocities=velocities, accelerations=accelerations, prev_positions=positions, time=0,
            ref_positions=positions, ref_velocities=velocities, ref_accelerations=accelerations,
        )
        return self.add_obs_noise(key, self.get_obs(state=state)), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: DroneEnvState,
        actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], DroneEnvState, Dict[str, float], Dict[str, bool], Dict]:
        action = jnp.stack([actions[agent] for agent in self.agents])
        action = jnp.clip(action, -self.max_action, self.max_action)
        ref_positions, ref_velocities, ref_accelerations = state.ref_positions, state.ref_velocities, state.ref_accelerations
        if self.use_tracking_controller:
            (ref_positions, ref_velocities, ref_accelerations), (positions, velocities, accelerations) = self.track_reference(
                (ref_positions, ref_velocities, ref_accelerations),
                (state.positions, state.velocities, state.accelerations),
                self.max_speed * action,
            )
        elif self.use_velocity_filter:
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
            prev_positions=state.positions, time=state.time + 1,
            ref_positions=ref_positions, ref_velocities=ref_velocities, ref_accelerations=ref_accelerations,
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

        obs = self.add_obs_noise(key, self.get_obs(state=new_state))
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
        def substep(_, carry):
            return self.filter_substep(*carry, target_velocities)

        return jax.lax.fori_loop(0, self.n_substeps, substep, (positions, velocities, accelerations))

    def filter_substep(
        self,
        p: jax.Array,
        v: jax.Array,
        a: jax.Array,
        target_velocities: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Advances (p, v, a) by one substep `h` of the deployed velocity filter."""
        tau = self.velocity_filter_tau
        jerk = (target_velocities - v) / tau**2 - 2.0 * a / tau
        a_next = a + self.h * jerk
        # Vector-norm acceleration limit, not a per-axis limit.
        magnitude = jnp.linalg.norm(a_next, axis=-1, keepdims=True)
        a_next = a_next * jnp.where(magnitude > self.max_filter_accel, self.max_filter_accel / magnitude, 1.0)
        v_next = v + self.h * a_next
        p_next = p + 0.5 * self.h * (v + v_next)
        return p_next, v_next, a_next

    def track_reference(
        self,
        reference: Tuple[jax.Array, jax.Array, jax.Array],
        drone: Tuple[jax.Array, jax.Array, jax.Array],
        target_velocities: jax.Array
    ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:
        """Advances the velocity filter's (positions, velocities, accelerations) `reference`
        and the point-mass `drone` tracking it by one policy step `dt`, holding
        `target_velocities` constant throughout (see the module docstring).

        Each substep first advances the reference, then the drone toward the advanced
        reference, as in the collaborator's point-mass model of the stack.
        """
        def substep(_, carry):
            (ref_p, ref_v, ref_a), (p, v, a) = carry
            ref_p, ref_v, ref_a = self.filter_substep(ref_p, ref_v, ref_a, target_velocities)
            request = ref_a + self.kp * (ref_p - p) + self.kv * (ref_v - v)
            a = a + self.motor_alpha * (request - a)
            v_next = v + self.h * a
            p_next = p + 0.5 * self.h * (v + v_next)
            return (ref_p, ref_v, ref_a), (p_next, v_next, a)

        return jax.lax.fori_loop(0, self.n_substeps, substep, (reference, drone))

    def tracking_velocity_gain(self, tol: float = 1e-12, max_seconds: float = 600.0) -> np.ndarray:
        """Per-axis worst case of |drone velocity| / max_t |reference velocity| under the
        tracking controller, starting from the drone at its reference.

        Per axis, the substep map from the reference velocity sequence to the drone's
        velocity is linear and time-invariant (the reference's acceleration is its
        velocity's difference over h and its position the trapezoidal integral, exactly
        as in `filter_substep`), so the worst case is the L1 norm of its impulse
        response. Computed in float64; asserts the response decays, i.e. the gains are
        stable at this h.
        """
        h = self.h
        kp, kv = np.array(self.tracking_kp), np.array(self.tracking_kv)
        alpha = 1.0 - np.exp(-h / np.array(self.motor_tau))
        # Unit impulse in the reference velocity at the first substep.
        ref_v_prev, ref_v = np.zeros(3), np.ones(3)
        ref_p, p, v, a = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        gain = np.zeros(3)
        for _ in range(int(max_seconds / h)):
            ref_a = (ref_v - ref_v_prev) / h
            ref_p = ref_p + 0.5 * h * (ref_v_prev + ref_v)
            a = a + alpha * ((ref_a + kp * (ref_p - p) + kv * (ref_v - v)) - a)
            v_next = v + h * a
            p = p + 0.5 * h * (v + v_next)
            v = v_next
            gain += np.abs(v)
            ref_v_prev, ref_v = ref_v, np.zeros(3)
            if np.all(np.abs(np.concatenate([v, a, p - ref_p])) < tol):
                return gain
            if not np.all(gain < 1e6):  # diverging
                break
        raise AssertionError(
            f"tracking controller (tracking_kp={self.tracking_kp}, tracking_kv={self.tracking_kv}, "
            f"motor_tau={self.motor_tau}) is unstable or too slow at substep h={h}"
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: DroneEnvState
    ) -> Dict[str, chex.Array]:
        return {
            agent: jnp.concatenate([state.positions[i], state.velocities[i]])
            for i, agent in enumerate(self.agents)
        }

    def add_obs_noise(
        self,
        key: chex.PRNGKey,
        obs: Dict[str, chex.Array]
    ) -> Dict[str, chex.Array]:
        """Adds fresh per-agent observation noise (see the module docstring) to `obs`;
        returns `obs` untouched, drawing no randomness, when noise is disabled."""
        if not self.use_obs_noise:
            return obs
        keys = jax.random.split(key, self.n_agents)
        bound = self.OBS_NOISE_MAX_SIGMAS
        return {
            agent: obs[agent] + self.obs_noise_scale * jax.random.truncated_normal(k, -bound, bound, (6,), jnp.float32)
            for agent, k in zip(self.agents, keys)
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
