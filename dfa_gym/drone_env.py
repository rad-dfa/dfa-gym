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
        positions = jax.random.uniform(key, shape=(self.n_agents, 3), minval=self.low, maxval=self.high)
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
