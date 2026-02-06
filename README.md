# dfa-gym

This repo implements (Multi-Agent) Reinforcement Learning environments in JAX for solving objectives given as Deteministic Finite Automata (DFAs). There are three environments:

1. `TokenEnv` is a fully observable grid environment with tokens in cells. The grid can be created randomly or from a specific layout. It can be instantiated in both single- and multi-agent settings.
2. `DFAWrapper` is an environment wrapper assigning tasks represented as Deterministic Finite Automata (DFAs) to the agents in the wrapped environment. DFAs are repsented as [`DFAx`](https://github.com/rad-dfa/dfax) objects.
3. `DFABisimEnv` is an environment for solving DFA bisimulation games to learn RAD Embeddings, provably correct latent DFA representation, as described in [this paper](https://arxiv.org/pdf/2503.05042).


## Installation

Install using pip.

```
pip install dfa-gym
```

## TokenEnv

Create a grid world with token and agent positions assigned randomly.

```python
from dfa_gym import TokenEnv

env = TokenEnv(
        n_agents=1, # Single agent
        n_tokens=10, # 10 different token types
        n_token_repeat=2, # Each token repeated twice
        grid_shape=(7, 7), # Shape of the grid
        fixed_map_seed=None, # If not None, then samples the same map using the given seed
        max_steps_in_episode=100, # Episode length is 100
    )
```

Create a grid world from a given layout.

```python
layout = """
    [ 0 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    [   ][   ][ a ][   ][#,a][ 0 ][   ][ 2 ][ # ]
    [ A ][   ][ a ][   ][#,a][   ][ 8 ][   ][ # ]
    [   ][   ][ a ][   ][#,a][ 6 ][   ][ 4 ][ # ]
    [ 1 ][   ][   ][ 3 ][ # ][ # ][ # ][ # ][ # ]
    [   ][   ][ b ][   ][#,b][ 1 ][   ][ 3 ][ # ]
    [ B ][   ][ b ][   ][#,b][   ][ 9 ][   ][ # ]
    [   ][   ][ b ][   ][#,b][ 7 ][   ][ 5 ][ # ]
    [ 2 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    """
    env = TokenEnv(
        layout=layout, # Set layout, where each [] indicates a cell, uppercase letters are
                       # agents, # are walls, and lower case letters are buttons when alone
                       # and doors when paired with a wall. For example, [#,a] is a door
                       # that is open if an agent is on a [ a ] cell and closed otherwise.
    )
```


## DFAWrapper

Wrap a `TokenEnv` instance using `DFAWrapper `.

```python
from dfa_gym import DFAWrapper
from dfax.samplers import ReachSampler

env = DFAWrapper(
    env=TokenEnv(layout=layout),
    sampler=ReachSampler()
)
```

## DFABisimEnv

Create DFA bisimulation game.

```python
from dfa_gym import DFABisimEnv
from dfax.samplers import RADSampler

env = DFABisimEnv(sampler=RADSampler())
```

