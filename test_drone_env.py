import argparse
import jax
import numpy as np
import jax.numpy as jnp
from dfa_gym import DroneEnv, DFAWrapper, visualize_drone_state, animate_drone_trace
from dfax import DFAx
from dfax.samplers import RADSampler
from dfax.utils import list2batch, batch2graph

def simulate_step(position, velocity, acceleration, action,
                  POLICY_DT, FILTER_DT, TAU, MAX_ACCEL, VELOCITY_SCALE):
    """The deployment's velocity filter step, as sent by the collaborator deploying on
    the real drone -- verbatim, except that its module-level constants are arguments.
    Single agent, float64, no geofence.
    """
    # Assumes EMA disabled and optional velocity norm cap disabled.
    # policy action normalized [-1,1].
    target_velocity = VELOCITY_SCALE * np.clip(action, -1.0, 1.0)

    p = position.copy()
    v = velocity.copy()
    a = acceleration.copy()

    n = int(np.ceil(POLICY_DT / FILTER_DT))
    h = POLICY_DT / n

    # Hold the policy target constant throughout this 0.1-second interval.
    for _ in range(n):
        jerk = (target_velocity - v) / TAU**2 - 2.0 * a / TAU

        a_next = a + h * jerk

        # Vector-norm acceleration limit, not a per-axis limit.
        magnitude = np.linalg.norm(a_next)
        if magnitude > MAX_ACCEL:
            a_next *= MAX_ACCEL / magnitude

        v_next = v + h * a_next
        p += 0.5 * h * (v + v_next)

        v = v_next
        a = a_next

    return p, v, a


def simulate_step_geofenced(position, velocity, acceleration, action, env):
    """`simulate_step` plus DroneEnv's geofence: after every substep, an axis at (or
    pushed past) a wall while moving into it is pinned at the wall and its velocity and
    acceleration are zeroed.
    """
    low, high = np.asarray(env.low, dtype=np.float64), np.asarray(env.high, dtype=np.float64)
    target_velocity = env.max_speed * np.clip(action, -1.0, 1.0)

    p = position.copy()
    v = velocity.copy()
    a = acceleration.copy()

    n = int(np.ceil(env.dt / env.filter_dt))
    h = env.dt / n

    for _ in range(n):
        jerk = (target_velocity - v) / env.velocity_filter_tau**2 - 2.0 * a / env.velocity_filter_tau
        a_next = a + h * jerk
        magnitude = np.linalg.norm(a_next)
        if magnitude > env.max_filter_accel:
            a_next *= env.max_filter_accel / magnitude
        v_next = v + h * a_next
        p_next = p + 0.5 * h * (v + v_next)

        pinned = ((p_next <= low) & (v_next < 0)) | ((p_next >= high) & (v_next > 0))
        p = np.clip(p_next, low, high)
        v = np.where(pinned, 0.0, v_next)
        a = np.where(pinned, 0.0, a_next)

    return p, v, a


def to_np(x):
    return np.asarray(x, dtype=np.float64)


def test(env, n=100, seed=42):

    print(f"Running tests for {env.name} (n_agents={env.n_agents}, use_displacement_action={env.use_displacement_action}, use_velocity_filter={env.use_velocity_filter}).")
    key = jax.random.PRNGKey(seed)

    for i in range(n):

        key, subkey = jax.random.split(key)
        _, state = env.reset(key=subkey)
        assert jnp.all(state.positions >= env.low) and jnp.all(state.positions <= env.high)
        assert jnp.all(state.velocities == 0.0)
        assert jnp.all(state.accelerations == 0.0)
        assert jnp.array_equal(state.prev_positions, state.positions)
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

            if env.use_velocity_filter:
                # Recompute in float64 from the same prior state, so only one step of
                # float32 rounding separates this from the jitted step_env.
                expected = [
                    simulate_step_geofenced(
                        to_np(state.positions[i]), to_np(state.velocities[i]), to_np(state.accelerations[i]),
                        to_np(actions[agent]), env,
                    )
                    for i, agent in enumerate(env.agents)
                ]
                expected_positions, expected_velocities, expected_accelerations = (np.stack(x) for x in zip(*expected))
                assert np.allclose(new_state.positions, expected_positions, atol=1e-5)
                assert np.allclose(new_state.velocities, expected_velocities, atol=1e-5)
                assert np.allclose(new_state.accelerations, expected_accelerations, atol=1e-4)
                assert jnp.all(jnp.linalg.norm(new_state.accelerations, axis=-1) <= env.max_filter_accel + 1e-5)
            else:
                clipped = jnp.clip(jnp.stack([actions[a] for a in env.agents]), -env.max_action, env.max_action)
                delta = clipped if env.use_displacement_action else clipped * env.dt
                expected_positions = jnp.clip(state.positions + delta, env.low, env.high)
                expected_velocities = (expected_positions - state.positions) / env.dt
                # atol on velocities is looser: dividing by dt amplifies float32 rounding
                # differences between this eager recomputation and the jitted step_env.
                assert jnp.allclose(new_state.positions, expected_positions)
                assert jnp.allclose(new_state.velocities, expected_velocities, atol=1e-4)
                assert jnp.all(new_state.accelerations == 0.0)
            assert jnp.all(expected_positions >= env.low) and jnp.all(expected_positions <= env.high)
            assert jnp.all(jnp.abs(expected_velocities) <= env.max_speed + 1e-6)
            assert jnp.array_equal(new_state.prev_positions, state.positions)

            for i, agent in enumerate(env.agents):
                assert jnp.allclose(obs[agent], jnp.concatenate([expected_positions[i], expected_velocities[i]]), atol=1e-4)
                assert jnp.array_equal(obs[agent], jnp.concatenate([new_state.positions[i], new_state.velocities[i]]))

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


def test_filter_matches_reference(env, n=10, seed=42):
    """Checks DroneEnv's velocity filter against the deployment's `simulate_step`.

    `env` must have a box large enough that the geofence (which the deployment
    doesn't have) is never hit. Each step is compared both one-step (the reference
    run from the env's own prior state) and open-loop (a float64 reference
    trajectory carried across the whole episode, as in the deployment rollout).
    """

    assert env.use_velocity_filter
    print(f"Running filter reference tests for {env.name} (n_agents={env.n_agents}, tau={env.velocity_filter_tau}, "
          f"max_filter_accel={env.max_filter_accel}, filter_dt={env.filter_dt}, max_speed={env.max_speed}).")
    constants = dict(POLICY_DT=env.dt, FILTER_DT=env.filter_dt, TAU=env.velocity_filter_tau,
                     MAX_ACCEL=env.max_filter_accel, VELOCITY_SCALE=env.max_speed)
    key = jax.random.PRNGKey(seed)

    for i in range(n):

        key, subkey = jax.random.split(key)
        _, state = env.reset(key=subkey)
        trajectory = [(np.zeros(3), np.zeros(3), np.zeros(3)) for _ in env.agents]
        done = False

        while not done:
            key, action_key, step_key = jax.random.split(key, 3)
            # also exercise the [-1, 1] action clipping
            action_keys = jax.random.split(action_key, env.n_agents)
            actions = {
                agent: jax.random.uniform(k, (3,), minval=-1.5, maxval=1.5)
                for agent, k in zip(env.agents, action_keys)
            }

            obs, new_state, _, dones, _ = env.step_env(step_key, state, actions)

            for j, agent in enumerate(env.agents):
                p, v, a = simulate_step(
                    to_np(state.positions[j]), to_np(state.velocities[j]), to_np(state.accelerations[j]),
                    to_np(actions[agent]), **constants,
                )
                assert np.allclose(new_state.positions[j], p, atol=1e-5)
                assert np.allclose(new_state.velocities[j], v, atol=1e-5)
                assert np.allclose(new_state.accelerations[j], a, atol=1e-4)

                trajectory[j] = simulate_step(*trajectory[j], to_np(actions[agent]), **constants)
                p, v, a = trajectory[j]
                assert np.allclose(new_state.positions[j], p, atol=1e-4)
                assert np.allclose(new_state.velocities[j], v, atol=1e-4)
                assert np.allclose(new_state.accelerations[j], a, atol=1e-3)
                assert np.all(p > to_np(env.low)) and np.all(p < to_np(env.high)), "geofence hit; use a larger box"

                assert jnp.array_equal(obs[agent], jnp.concatenate([new_state.positions[j], new_state.velocities[j]]))

            state = new_state
            done = dones["__all__"]

        print(f"Test completed for {i + 1} samples.", end="\r")

    print(f"Test completed for {n} samples.")


def test_label_f_on_entry():
    """Checks that label_f fires a token only on the step its region is entered (no
    "sticky color"): staying inside yields -1, and re-triggering requires exiting first.

    Scripted displacement actions of up to 0.8 per axis land each agent exactly on
    region centers of the default box (r = 0.2), well away from region boundaries.
    """

    print("Running label_f on-entry tests for DroneEnv.")
    env = DroneEnv(n_agents=2, use_displacement_action=True, max_speed=8.0)
    stay = (0.0, 0.0, 0.0)
    moves = {
        # Enter the token-1 circle at (0, -0.8), stay twice, exit to the origin, re-enter.
        "agent_0": [(0.0, -0.8, 0.0), stay, stay, (0.0, 0.8, 0.0), (0.0, -0.8, 0.0)],
        # Enter corridor 2, move straight into corner 0, straight back into corridor 2, stay.
        "agent_1": [(-0.8, -0.4, 0.0), (0.0, -0.4, 0.0), (0.0, 0.4, 0.0), stay, stay],
    }
    expected_labels = {"agent_0": [1, -1, -1, -1, 1], "agent_1": [2, 0, 2, -1, -1]}

    # "Reach token 1, then reach token 1": token 1 moves s0 -> s1 -> s2 (accepting, absorbing).
    transitions = jnp.tile(jnp.arange(3, dtype=jnp.int32)[:, None], (1, env.n_tokens))
    transitions = transitions.at[0, 1].set(1).at[1, 1].set(2)
    dfa = DFAx.create(start=0, transitions=transitions, labels=jnp.array([False, False, True]))

    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    assert all(label == -1 for label in env.label_f(state).values())

    for t in range(len(expected_labels["agent_0"])):
        actions = {agent: jnp.array(moves[agent][t], dtype=jnp.float32) for agent in env.agents}
        _, state, _, _, _ = env.step_env(key, state, actions)
        labels = env.label_f(state)
        for agent in env.agents:
            assert labels[agent] == expected_labels[agent][t], (agent, t, labels[agent])

        dfa = dfa.advance(labels["agent_0"])
        # Sitting on token 1 must not count as visiting it twice; only the re-entry does.
        assert dfa.reward() == (1.0 if t == 4 else 0.0)

    print("Test completed.")


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
            assert jnp.array_equal(new_state.env_state.accelerations, expected_env_state.accelerations)
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
    parser.add_argument("--test-filter-reference", action="store_true")
    parser.add_argument("--test-label-f", action="store_true")
    parser.add_argument("--test-dfa-wrapper", action="store_true")
    parser.add_argument("--visualize-init-state", action="store_true")
    parser.add_argument("--generate-gif", action="store_true")
    args = parser.parse_args()

    if args.test_drone_env:
        test(DroneEnv(n_agents=1, use_displacement_action=False))
        test(DroneEnv(n_agents=3, use_displacement_action=False))
        test(DroneEnv(n_agents=3, max_speed=2.0))
        test(DroneEnv(n_agents=1, use_displacement_action=True))
        test(DroneEnv(n_agents=3, use_displacement_action=True, x_low=-2.0, x_high=2.0, y_low=-1.5, y_high=1.5, z_low=0.0, z_high=2.0))
        test(DroneEnv(n_agents=1, use_velocity_filter=True))
        test(DroneEnv(n_agents=3, use_velocity_filter=True))
        test(DroneEnv(n_agents=3, use_velocity_filter=True, max_speed=2.0, x_low=-2.0, x_high=2.0, y_low=-1.5, y_high=1.5, z_low=0.0, z_high=2.0))

    if args.test_filter_reference:
        big_box = dict(x_low=-50.0, x_high=50.0, y_low=-50.0, y_high=50.0, z_low=-50.0, z_high=50.0)
        test_filter_matches_reference(DroneEnv(n_agents=1, use_velocity_filter=True, **big_box))
        test_filter_matches_reference(DroneEnv(n_agents=3, use_velocity_filter=True, max_speed=2.0, velocity_filter_tau=0.2,
                                               max_filter_accel=5.0, filter_dt=0.01, **big_box))

    if args.test_label_f:
        test_label_f_on_entry()

    if args.test_dfa_wrapper:
        env = DroneEnv(n_agents=1, max_steps_in_episode=20)
        test_dfa_wrapper(DFAWrapper(env=env, sampler=RADSampler(n_tokens=env.n_tokens)))

        env = DroneEnv(n_agents=3, max_steps_in_episode=20)
        test_dfa_wrapper(DFAWrapper(env=env, sampler=RADSampler(n_tokens=env.n_tokens)))

        env = DroneEnv(n_agents=3, use_velocity_filter=True, max_steps_in_episode=20)
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
