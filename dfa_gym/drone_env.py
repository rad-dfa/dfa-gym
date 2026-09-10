""" DroneEnv: continuous point-mass agents confined to a bounded 3D box.

Action is a per-agent 3D velocity command (default) or, if
`use_displacement_action=True`, a direct 3D position-delta ("distance
change") command. No reward/goal is defined here; wrap or subclass for that.
"""
import jax
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
        max_steps_in_episode: int = 200,
    ) -> None:
        super().__init__(num_agents=n_agents)
        self.n_agents = n_agents
        self.low = jnp.array([x_low, y_low, z_low], dtype=jnp.float32)
        self.high = jnp.array([x_high, y_high, z_high], dtype=jnp.float32)
        self.max_speed = max_speed
        self.dt = dt
        self.use_displacement_action = use_displacement_action
        self.max_steps_in_episode = max_steps_in_episode
        # A max-magnitude velocity command covers max_speed*dt of ground per step,
        # so cap displacement actions at that same per-step reach.
        self.max_action = max_speed if not use_displacement_action else max_speed * dt

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
        state = DroneEnvState(positions=positions, velocities=velocities, time=0)
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
        delta = action if self.use_displacement_action else action * self.dt
        positions = jnp.clip(state.positions + delta, self.low, self.high)
        # Achieved velocity, not the commanded one: zero along any axis pinned by the geofence.
        velocities = (positions - state.positions) / self.dt

        new_state = DroneEnvState(positions=positions, velocities=velocities, time=state.time + 1)
        done = new_state.time >= self.max_steps_in_episode

        rewards = {agent: jnp.zeros((), dtype=jnp.float32) for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        obs = self.get_obs(state=new_state)
        info = {}

        return obs, new_state, rewards, dones, info

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
        0: center, 1: corners, 2: edge midpoints, 3: corner<->midpoint corridors on
        the vertical (left/right) edges, 4: same but on the horizontal (top/bottom) edges.
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
        vert_edges = [
            (x_low, x_low + 2 * r, y_low + 2 * r, y_mid - r),
            (x_low, x_low + 2 * r, y_mid + r, y_high - 2 * r),
            (x_high - 2 * r, x_high, y_low + 2 * r, y_mid - r),
            (x_high - 2 * r, x_high, y_mid + r, y_high - 2 * r),
        ]
        horiz_edges = [
            (x_low + 2 * r, x_mid - r, y_low, y_low + 2 * r),
            (x_mid + r, x_high - 2 * r, y_low, y_low + 2 * r),
            (x_low + 2 * r, x_mid - r, y_high - 2 * r, y_high),
            (x_mid + r, x_high - 2 * r, y_high - 2 * r, y_high),
        ]

        regions = [(0, "circle", (x_mid, y_mid, r * 2, z_hi, z_hi + 0.25))]
        regions += [(1, "circle", (cx, cy, r, z_lo, z_hi)) for cx, cy in corners]
        regions += [(2, "circle", (mx, my, r, z_lo, z_hi)) for mx, my in edge_mids]
        regions += [(3, "rect", bounds + (z_lo, z_hi)) for bounds in vert_edges]
        regions += [(4, "rect", bounds + (z_lo, z_hi)) for bounds in horiz_edges]
        return regions

    @partial(jax.jit, static_argnums=(0,))
    def label_f(
        self,
        state: DroneEnvState
    ) -> Dict[str, int]:
        # Labels the agent's full (x, y, z) position -- each region is a 1.0-tall prism.
        x, y, z = state.positions[:, 0], state.positions[:, 1], state.positions[:, 2]
        labels = jnp.full(x.shape, -1, dtype=jnp.int32)
        for token, kind, params in reversed(self.label_regions()):
            if kind == "circle":
                cx, cy, r, z_lo, z_hi = params
                mask = ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2) & (z >= z_lo) & (z <= z_hi)
            else:
                x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = params
                mask = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi) & (z >= z_lo) & (z <= z_hi)
            labels = jnp.where(mask, token, labels)

        return {agent: labels[i] for i, agent in enumerate(self.agents)}
