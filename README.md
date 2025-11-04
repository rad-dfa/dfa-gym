# dfa-gym

This repo implements a Gymnasium environment `DFAEnv` for solving Deteministic Finite Automata (DFA) and a Gymnasium environment wrapper `DFAWrapper` for wrapping environments with DFA tasks.

## Installation

```
pip install dfa-gym
```

## Usage

```
import gymnasium as gym
from dfa_gym import DFAEnv, DFAWrapper

if __name__ == "__main__":
    # Test DFAEnv
    dfa_env = gym.make("DFAEnv-v0")
    obs, info = dfa_env.reset()
    for _ in range(1000):
        action = dfa_env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = dfa_env.step(action)
        if done:
            break
    dfa_env.close()
    # Test DFAWrapper
    env_cls = "CartPole-v1"
    wrapped_env = DFAWrapper(env_cls)
    observation, info = wrapped_env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        action = wrapped_env.action_space.sample()
        observation, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated
    wrapped_env.close()
```
