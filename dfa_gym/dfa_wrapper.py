import jax
import dfax
import chex
import jax.numpy as jnp
from flax import struct
from dfa_gym import spaces
from functools import partial
from typing import Tuple, Dict, Callable
from dfax.utils import list2batch, batch2graph
from dfa_gym.env import MultiAgentEnv, State
from dfax.samplers import DFASampler, RADSampler


@struct.dataclass
class DFAWrapperState(State):
    dfas: Dict[str, dfax.DFAx]
    init_dfas: Dict[str, dfax.DFAx]
    env_obs: chex.Array
    env_state: State

class DFAWrapper(MultiAgentEnv):

    def __init__(
        self,
        env: MultiAgentEnv,
        gamma: float | None = None,
        sampler: DFASampler = RADSampler(),
        binary_reward: bool = True,
        progress: bool = True,
        embedder: Callable[[dfax.DFAx], jnp.ndarray] | None = None,
        embedding_dim: int | None = None,
        combine_embed: bool = False,
    ) -> None:
        super().__init__(num_agents=env.num_agents)
        self.env = env
        self.gamma = gamma
        self.sampler = sampler
        self.binary_reward = binary_reward
        self.progress = progress
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.combine_embed = combine_embed

        assert (self.embedder is None) == (self.embedding_dim is None)
        assert self.sampler.n_tokens == self.env.n_tokens

        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        self.action_spaces = {
            agent: self.env.action_space(agent)
            for agent in self.agents
        }
        max_dfa_size = self.sampler.max_size
        n_tokens = self.sampler.n_tokens

        if self.embedder is None:
            self.observation_spaces = {
                agent: spaces.Dict({
                    "_id": spaces.Discrete(self.num_agents),
                    "obs": self.env.observation_space(agent),
                    "dfa": spaces.Dict({
                        "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*self.num_agents, 4), dtype=jnp.float32),
                        "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*self.num_agents*max_dfa_size*self.num_agents, n_tokens + 8), dtype=jnp.float32),
                        "edge_index": spaces.Box(low=0, high=max_dfa_size*self.num_agents, shape=(2, max_dfa_size*self.num_agents*max_dfa_size*self.num_agents), dtype=jnp.int32),
                        "current_state": spaces.Box(low=0, high=max_dfa_size*self.num_agents, shape=(self.num_agents,), dtype=jnp.int32),
                        "n_states": spaces.Box(low=0, high=max_dfa_size*self.num_agents, shape=(max_dfa_size*self.num_agents,), dtype=jnp.int32)
                    }),
                })
                for agent in self.agents
            }
        else:
            if self.combine_embed:
                self.observation_spaces = {
                    agent: spaces.Dict({
                        "_id": spaces.Discrete(self.num_agents),
                        "obs": self.env.observation_space(agent),
                        "dfa": spaces.Dict({
                            "node_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*self.num_agents, 4), dtype=jnp.float32),
                            "edge_features": spaces.Box(low=0, high=1, shape=(max_dfa_size*self.num_agents*max_dfa_size*self.num_agents, n_tokens + 8), dtype=jnp.float32),
                            "edge_index": spaces.Box(low=0, high=max_dfa_size*self.num_agents, shape=(2, max_dfa_size*self.num_agents*max_dfa_size*self.num_agents), dtype=jnp.int32),
                            "current_state": spaces.Box(low=0, high=max_dfa_size*self.num_agents, shape=(self.num_agents,), dtype=jnp.int32),
                            "n_states": spaces.Box(low=0, high=max_dfa_size*self.num_agents, shape=(max_dfa_size*self.num_agents,), dtype=jnp.int32)
                        }),
                        "emb": spaces.Box(low=-1, high=1, shape=(self.num_agents, self.embedding_dim), dtype=jnp.float32)
                    })
                    for agent in self.agents
                }
            else:
                self.observation_spaces = {
                    agent: spaces.Dict({
                        "_id": spaces.Discrete(self.num_agents),
                        "obs": self.env.observation_space(agent),
                        "emb": spaces.Box(low=-1, high=1, shape=(self.num_agents, self.embedding_dim), dtype=jnp.float32)
                    })
                    for agent in self.agents
                }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], DFAWrapperState]:
        keys = jax.random.split(key, 4 + self.num_agents)

        env_obs, env_state = self.env.reset(keys[1])

        n_trivial = jax.random.choice(keys[2], self.num_agents)
        mask = jax.random.permutation(keys[3], jnp.arange(self.num_agents) < n_trivial)

        def sample_dfa(dfa_key, sample_trivial):
            return jax.tree_util.tree_map(
                lambda t, s: jnp.where(sample_trivial, t, s),
                self.sampler.trivial(True),
                self.sampler.sample(dfa_key)
            )

        dfas_tree = jax.vmap(sample_dfa)(keys[4:], mask)

        dfas = {
            agent: jax.tree_util.tree_map(lambda x: x[i], dfas_tree)
            for i, agent in enumerate(self.agents)
        }

        state = DFAWrapperState(
            dfas=dfas,
            init_dfas={agent: dfas[agent] for agent in self.agents},
            env_obs=env_obs,
            env_state=env_state
        )
        obs = self.get_obs(state=state)

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: DFAWrapperState,
        action: int,
    ) -> Tuple[Dict[str, chex.Array], DFAWrapperState, Dict[str, float], Dict[str, bool], Dict]:

        env_obs, env_state, env_rewards, env_dones, env_info = self.env.step_env(key, state.env_state, action)

        symbols = self.env.label_f(env_state)

        dfas = {
            agent: state.dfas[agent].advance(symbols[agent]).minimize()
            for agent in self.agents
        }

        dones = {
            agent: jnp.logical_or(env_dones[agent], dfas[agent].n_states <= 1)
            for agent in self.agents
        }
        _dones = jnp.array([dones[agent] for agent in self.agents])
        dones.update({"__all__": jnp.all(_dones)})

        dfa_rewards_min = jnp.min(jnp.array([dfas[agent].reward(binary=self.binary_reward) for agent in self.agents]))
        rewards = {
            agent: jax.lax.cond(
                dones["__all__"],
                lambda _: env_rewards[agent] + dfa_rewards_min,
                lambda _: env_rewards[agent],
                operand=None
            )
            for agent in self.agents
        }

        if self.gamma is not None:
            rewards = {
                agent: rewards[agent] + self.gamma * dfas[agent].reward(binary=self.binary_reward) - state.dfas[agent].reward(binary=self.binary_reward)
                for agent in self.agents
            }

        infos = {}

        state = DFAWrapperState(
            dfas=dfas,
            init_dfas=state.init_dfas,
            env_obs=env_obs,
            env_state=env_state
        )

        obs = self.get_obs(state=state)

        return obs, state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: DFAWrapperState
    ) -> Dict[str, chex.Array]:

        dfas = None
        if (self.embedder is None) or (self.combine_embed):
            if self.progress:
                dfas = batch2graph(
                    list2batch(
                        [state.dfas[agent].to_graph() for agent in self.agents]
                    )
                )
            else:
                dfas = batch2graph(
                    list2batch(
                        [state.init_dfas[agent].to_graph() for agent in self.agents]
                    )
                )

        embs = None
        if self.embedder is not None:
            if self.progress:
                embs = jnp.array([self.embedder(state.dfas[agent]) for agent in self.agents])
            else:
                embs = jnp.array([self.embedder(state.init_dfas[agent]) for agent in self.agents])

        assert (dfas is not None) or (embs is not None)

        def make_entry(i, agent):
            entry = {
                "_id": i,
                "obs": state.env_obs[agent],
            }
            if dfas is not None:
                entry["dfa"] = dfas
            if embs is not None:
                entry["emb"] = embs
            return entry

        return {agent: make_entry(i, agent) for i, agent in enumerate(self.agents)}

    def render(self, state: DFAWrapperState):
        out = ""
        for agent in self.agents:
            out += "****\n"
            out += f"{agent}'s DFA:\n"
            if self.progress:
                out += f"{state.dfas[agent]}\n"
            else:
                out += f"{state.init_dfas[agent]}\n"
        self.env.render(state.env_state)
        print(out)

