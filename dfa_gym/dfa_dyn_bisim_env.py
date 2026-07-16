import jax
import dfax
import chex
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple, Dict
from dfa_gym import spaces
from dfa_gym.env import MultiAgentEnv, State
from dfax.samplers import DFASampler, RADSampler


@struct.dataclass
class DFADynBisimState(State):
    dfa_l: dfax.DFAx
    dfa_r: dfax.DFAx
    evt2sym: chex.Array
    time: int

class DFADynBisimEnv(MultiAgentEnv):

    def __init__(
        self,
        n_events: int = 10,
        sampler: DFASampler = RADSampler(),
        max_steps_in_episode: int = 100,
        binary_reward: bool = True,
    ) -> None:
        super().__init__(num_agents=1)
        self.n_agents = self.num_agents
        self.sampler = sampler
        self.n_events = n_events
        self.max_steps_in_episode = max_steps_in_episode
        self.binary_reward = binary_reward

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.action_spaces = {
            agent: spaces.Discrete(self.n_events)
            for agent in self.agents
        }
        max_dfa_size = self.sampler.max_size
        n_tokens = self.sampler.n_tokens
        n_events = self.n_events
        self.observation_spaces = {
            agent: spaces.Dict({
                "graph_l": spaces.Dict({
                    "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 4), dtype=jnp.uint16),
                    "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens + 8), dtype=jnp.uint16),
                    "edge_index": spaces.Box(low=0, high=max_dfa_size, shape=(2, max_dfa_size*max_dfa_size), dtype=jnp.uint16),
                    "current_state": spaces.Box(low=0, high=max_dfa_size, shape=(1,), dtype=jnp.uint16),
                    "n_states": spaces.Box(low=0, high=max_dfa_size, shape=(max_dfa_size,), dtype=jnp.uint16)
                }),
                "graph_r": spaces.Dict({
                    "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size, 4), dtype=jnp.uint16),
                    "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*max_dfa_size, n_tokens + 8), dtype=jnp.uint16),
                    "edge_index": spaces.Box(low=0, high=max_dfa_size, shape=(2, max_dfa_size*max_dfa_size), dtype=jnp.uint16),
                    "current_state": spaces.Box(low=0, high=max_dfa_size, shape=(1,), dtype=jnp.uint16),
                    "n_states": spaces.Box(low=0, high=max_dfa_size, shape=(max_dfa_size,), dtype=jnp.uint16)
                }),
                "evt2sym": spaces.Box(low=0, high=1, shape=(n_tokens, n_events), dtype=jnp.float32)
            })
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], DFADynBisimState]:

        def cond_fn(carry):
            _, dfa_l, dfa_r, word = carry
            return jnp.logical_or(dfa_l == dfa_r, jnp.all(word == -1))

        def body_fn(carry):
            key, _, _, _ = carry
            key, kl, kr, kx = jax.random.split(key, 4)
            dfa_l = self.sampler.sample(kl)
            dfa_r = self.sampler.sample(kr)
            dfa_xor = dfa_l ^ dfa_r
            word = dfa_xor.find_word(kx)
            return (key, dfa_l, dfa_r, word)

        init_carry = body_fn((key, None, None, None))
        key, dfa_l, dfa_r, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)

        key, subkey = jax.random.split(key)
        symbols = jax.random.randint(subkey, shape=(self.n_events,), minval=0, maxval=self.sampler.n_tokens)
        evt2sym = jax.nn.one_hot(symbols, self.sampler.n_tokens, dtype=jnp.float32).T

        state = DFADynBisimState(dfa_l=dfa_l, dfa_r=dfa_r, evt2sym=evt2sym, time=0)
        obs = self.get_obs(state=state)

        return {self.agents[0]: obs}, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: DFADynBisimState,
        action: int
    ) -> Tuple[Dict[str, chex.Array], DFADynBisimState, Dict[str, float], Dict[str, bool], Dict]:

        event = action[self.agents[0]]
        symbol = jnp.argmax(state.evt2sym[:, event])

        dfa_l = state.dfa_l.advance(symbol).minimize()
        dfa_r = state.dfa_r.advance(symbol).minimize()

        reward_l = dfa_l.reward(binary=self.binary_reward)
        reward_r = dfa_r.reward(binary=self.binary_reward)
        reward = reward_l - reward_r

        new_state = DFADynBisimState(
            dfa_l=dfa_l,
            dfa_r=dfa_r,
            evt2sym=state.evt2sym,
            time=state.time+1
        )

        done = jnp.logical_or(jnp.logical_or(reward_l != 0, reward_r != 0), new_state.time >= self.max_steps_in_episode)

        obs = self.get_obs(state=new_state)
        info = {}

        return {self.agents[0]: obs}, new_state, {self.agents[0]: reward}, {self.agents[0]: done, "__all__": done}, info

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: DFADynBisimState
    ) -> Dict[str, chex.Array]:
        return {
            "graph_l": state.dfa_l.to_graph(),
            "graph_r": state.dfa_r.to_graph(),
            "evt2sym": state.evt2sym
        }
