import argparse
import jax
import jax.numpy as jnp
from dfa_gym import DroneEnv, DFAWrapper, visualize_drone_state, animate_drone_trace
from dfax.samplers import RADSampler
from dfax.utils import list2batch, batch2graph

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


def test_dfa_wrapper(env, n=5, seed=42):
    """Tests DFAWrapper(env=DroneEnv(...)): verifies the augmented obs dict and the
    DFA/reward/done wiring, treating DroneEnv.step_env/label_f and DFAx's own
    advance/minimize/reward/to_graph as ground truth (mirrors test.py's style).
    """

    def assert_dfa_leaf_shapes(dfa_obs, agent):
        max_dfa_size = env.sampler.max_size
        n_syms = env.n_syms
        num_agents = env.num_agents
        dfa_space = env.observation_space(agent).spaces["dfa"].spaces

        assert dfa_obs["node_features"].shape == (max_dfa_size * num_agents, 4)
        assert dfa_obs["node_features"].shape == dfa_space["node_features"].shape
        assert dfa_obs["current_state"].shape == (num_agents,)
        assert dfa_obs["current_state"].shape == dfa_space["current_state"].shape
        assert dfa_obs["n_states"].shape == (max_dfa_size * num_agents,)
        assert dfa_obs["n_states"].shape == dfa_space["n_states"].shape
        # dfa_wrapper.py declares edge_features/edge_index shapes as quadratic in
        # num_agents, but batch2graph/list2batch actually concatenate per-agent
        # graphs, giving a shape linear in num_agents. The two formulas coincide
        # only when num_agents == 1, so these two leaves are checked against the
        # true (linear) formula rather than the declared observation_space.
        assert dfa_obs["edge_features"].shape == (max_dfa_size * max_dfa_size * num_agents, n_syms + 8)
        assert dfa_obs["edge_index"].shape == (2, max_dfa_size * max_dfa_size * num_agents)

    def assert_obs_matches(obs, state):
        assert set(obs.keys()) == set(env.agents)
        expected_dfa = batch2graph(list2batch([state.dfas[agent].to_graph() for agent in env.agents]))
        for i, agent in enumerate(env.agents):
            assert set(obs[agent].keys()) == {"_id", "obs", "dfa"}
            assert obs[agent]["_id"] == i
            assert jnp.array_equal(obs[agent]["obs"], state.env_obs[agent])
            for leaf in ("node_features", "edge_features", "edge_index", "current_state", "n_states"):
                assert jnp.array_equal(obs[agent]["dfa"][leaf], expected_dfa[leaf])
            assert_dfa_leaf_shapes(obs[agent]["dfa"], agent)

    print(f"Running tests for DFAWrapper(env={env.env.name}, n_agents={env.num_agents}).")
    key = jax.random.PRNGKey(seed)

    for i in range(n):

        key, subkey = jax.random.split(key)
        obs, state = env.reset(key=subkey)
        assert_obs_matches(obs, state)
        done = False
        steps = 0

        while not done:
            key, action_key, step_key = jax.random.split(key, 3)
            action_keys = jax.random.split(action_key, env.num_agents)
            actions = {
                agent: env.action_space(agent).sample(k)
                for agent, k in zip(env.agents, action_keys)
            }

            obs, new_state, rewards, dones, _ = env.step_env(step_key, state, actions)

            # Independently recompute the expected transition, reusing the same
            # step_key DFAWrapper itself passes to the wrapped env's step_env.
            expected_env_obs, expected_env_state, expected_env_rewards, expected_env_dones, _ = \
                env.env.step_env(step_key, state.env_state, actions)

            expected_tkns = env.env.label_f(expected_env_state)
            expected_syms = {agent: state.tkn2sym[expected_tkns[agent]] for agent in env.agents}
            expected_dfas = {
                agent: state.dfas[agent].advance(expected_syms[agent]).minimize()
                for agent in env.agents
            }
            expected_dones = {
                agent: jnp.logical_or(expected_env_dones[agent], expected_dfas[agent].n_states <= 1)
                for agent in env.agents
            }
            expected_dones["__all__"] = jnp.all(jnp.array([expected_dones[agent] for agent in env.agents]))
            expected_dfa_reward_min = jnp.min(jnp.array([
                expected_dfas[agent].reward(binary=env.binary_reward) for agent in env.agents
            ]))
            expected_rewards = {
                agent: (expected_env_rewards[agent] + expected_dfa_reward_min
                        if expected_dones["__all__"] else expected_env_rewards[agent])
                for agent in env.agents
            }

            assert jnp.array_equal(new_state.env_state.positions, expected_env_state.positions)
            assert jnp.array_equal(new_state.env_state.velocities, expected_env_state.velocities)
            assert jnp.array_equal(new_state.env_state.time, expected_env_state.time)
            for agent in env.agents:
                assert jnp.array_equal(new_state.env_obs[agent], expected_env_obs[agent])
                assert expected_dfas[agent] == new_state.dfas[agent]  # DFAx.__eq__ is bisimulation-style
                assert new_state.init_dfas[agent] == state.init_dfas[agent]
                assert dones[agent] == expected_dones[agent]
                assert jnp.array_equal(rewards[agent], expected_rewards[agent])
            assert dones["__all__"] == expected_dones["__all__"]
            assert jnp.array_equal(new_state.sym2tkn, state.sym2tkn)
            assert jnp.array_equal(new_state.tkn2sym, state.tkn2sym)

            assert_obs_matches(obs, new_state)

            state = new_state
            done = dones["__all__"]
            steps += 1
            assert steps <= env.env.max_steps_in_episode

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
    parser.add_argument("--test-drone-env", action="store_true")
    parser.add_argument("--test-dfa-wrapper", action="store_true")
    parser.add_argument("--visualize-init-state", action="store_true")
    parser.add_argument("--generate-gif", action="store_true")
    args = parser.parse_args()

    if args.test_drone_env:
        test(DroneEnv(n_agents=1, use_displacement_action=False))
        test(DroneEnv(n_agents=3, use_displacement_action=False))
        test(DroneEnv(n_agents=1, use_displacement_action=True))
        test(DroneEnv(n_agents=3, use_displacement_action=True, x_low=-2.0, x_high=2.0, y_low=-1.5, y_high=1.5, z_low=0.0, z_high=2.0))

    if args.test_dfa_wrapper:
        env = DroneEnv(n_agents=1, max_steps_in_episode=20)
        test_dfa_wrapper(DFAWrapper(env=env, sampler=RADSampler(n_tokens=env.n_tokens)))

        env = DroneEnv(n_agents=3, max_steps_in_episode=20)
        test_dfa_wrapper(DFAWrapper(env=env, sampler=RADSampler(n_tokens=env.n_tokens)))

    if args.visualize_init_state:
        env = DroneEnv(n_agents=1)
        _, state = env.reset(jax.random.PRNGKey(0))
        visualize_drone_state(env, state)

    if args.generate_gif:
        env = DroneEnv(n_agents=1, max_steps_in_episode=100)
        trace = rollout(env)
        animate_drone_trace(env, trace, save_path="drone_trace.gif")
        print("Saved gif to drone_trace.gif")
