import jax
import jax.numpy as jnp
from dfa_gym import DFABisimEnv, DFADynBisimEnv

def test(env, n=100, seed=42):

    print(f"Running tests for {env.name}.")
    key = jax.random.PRNGKey(seed)

    for i in range(n):

        key, subkey = jax.random.split(key)
        _, state = env.reset(key=subkey)
        done = False
        steps = 0

        while not done:
            key, action_key, step_key = jax.random.split(key, 3)
            actions = {agent: env.action_space(agent).sample(action_key) for agent in env.agents}

            _, new_state, rewards, dones, _ = env.step_env(step_key, state, actions)

            action = actions[env.agents[0]]
            if hasattr(state, "evt2sym"):
                action = jnp.argmax(state.evt2sym[:, action])

            expected_l = state.dfa_l.advance(action).minimize()
            expected_r = state.dfa_r.advance(action).minimize()
            assert expected_l == new_state.dfa_l
            assert expected_r == new_state.dfa_r

            reward_l = expected_l.reward(binary=env.binary_reward)
            reward_r = expected_r.reward(binary=env.binary_reward)
            assert reward_l - reward_r == rewards[env.agents[0]]

            expected_done = reward_l != 0 or reward_r != 0 or new_state.time >= env.max_steps_in_episode
            assert expected_done == dones["__all__"]

            if hasattr(state, "evt2sym"):
                assert jnp.array_equal(new_state.evt2sym, state.evt2sym)

            state = new_state
            done = dones["__all__"]
            steps += 1
            assert steps <= env.max_steps_in_episode

        print(f"Test completed for {i + 1} samples.", end="\r")

    print(f"Test completed for {n} samples.")


if __name__ == '__main__':

    test(DFABisimEnv())
    test(DFADynBisimEnv())
