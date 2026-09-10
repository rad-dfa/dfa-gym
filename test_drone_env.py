import argparse
import jax
import jax.numpy as jnp
from dfa_gym import DroneEnv, visualize_drone_state, animate_drone_trace

def test(env, n=100, seed=42):

    print(f"Running tests for {env.name} (n_agents={env.n_agents}, use_displacement_action={env.use_displacement_action}).")
    key = jax.random.PRNGKey(seed)

    for i in range(n):

        key, subkey = jax.random.split(key)
        _, state = env.reset(key=subkey)
        assert jnp.all(state.positions >= env.low) and jnp.all(state.positions <= env.high)
        assert jnp.all(state.velocities == 0.0)
        done = False
        steps = 0

        while not done:
            key, action_key, step_key = jax.random.split(key, 3)
            # sample well outside the valid action range to also exercise clipping
            action_keys = jax.random.split(action_key, env.n_agents)
            actions = {
                agent: jax.random.uniform(k, (3,), minval=-10.0, maxval=10.0)
                for agent, k in zip(env.agents, action_keys)
            }

            obs, new_state, rewards, dones, _ = env.step_env(step_key, state, actions)

            clipped = jnp.clip(jnp.stack([actions[a] for a in env.agents]), -env.max_action, env.max_action)
            delta = clipped if env.use_displacement_action else clipped * env.dt
            expected_positions = jnp.clip(state.positions + delta, env.low, env.high)
            expected_velocities = (expected_positions - state.positions) / env.dt
            # atol on velocities is looser: dividing by dt amplifies float32 rounding
            # differences between this eager recomputation and the jitted step_env.
            assert jnp.allclose(new_state.positions, expected_positions)
            assert jnp.allclose(new_state.velocities, expected_velocities, atol=1e-4)
            assert jnp.all(expected_positions >= env.low) and jnp.all(expected_positions <= env.high)
            assert jnp.all(jnp.abs(expected_velocities) <= env.max_speed + 1e-6)

            for i, agent in enumerate(env.agents):
                assert jnp.allclose(obs[agent], jnp.concatenate([expected_positions[i], expected_velocities[i]]), atol=1e-4)

            expected_done = new_state.time >= env.max_steps_in_episode
            for agent in env.agents:
                assert rewards[agent] == 0.0
                assert dones[agent] == expected_done
            assert dones["__all__"] == expected_done

            state = new_state
            done = dones["__all__"]
            steps += 1
            assert steps <= env.max_steps_in_episode

        print(f"Test completed for {i + 1} samples.", end="\r")

    print(f"Test completed for {n} samples.")


def rollout(env, seed=0):
    """Runs one episode with a random policy and returns the list of visited states."""
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    _, state = env.reset(subkey)
    trace = [state]

    done = False
    while not done:
        key, action_key, step_key = jax.random.split(key, 3)
        action_keys = jax.random.split(action_key, env.n_agents)
        actions = {agent: env.action_space(agent).sample(k) for agent, k in zip(env.agents, action_keys)}
        _, state, _, dones, _ = env.step_env(step_key, state, actions)
        trace.append(state)
        done = dones["__all__"]

    return trace


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize-init-state", action="store_true")
    parser.add_argument("--generate-gif", action="store_true")
    args = parser.parse_args()

    test(DroneEnv(n_agents=1, use_displacement_action=False))
    test(DroneEnv(n_agents=3, use_displacement_action=False))
    test(DroneEnv(n_agents=1, use_displacement_action=True))
    test(DroneEnv(n_agents=3, use_displacement_action=True, x_low=-2.0, x_high=2.0, y_low=-1.5, y_high=1.5, z_low=0.0, z_high=2.0))

    if args.visualize_init_state:
        env = DroneEnv(n_agents=1)
        _, state = env.reset(jax.random.PRNGKey(0))
        visualize_drone_state(env, state)

    if args.generate_gif:
        env = DroneEnv(n_agents=1, max_steps_in_episode=100)
        trace = rollout(env)
        animate_drone_trace(env, trace, save_path="drone_trace.gif")
        print("Saved gif to drone_trace.gif")
