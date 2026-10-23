import jax
import jax.numpy as jnp
from dfax.samplers import RADSampler
from dfa_gym import DFABisimEnv, DFADynBisimEnv, DFAWrapper, TokenEnv, DroneEnv

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


def test_dfa_wrapper_requires_label_f():

    print("Running test for DFAWrapper's label_f requirement.")

    DFAWrapper(env=TokenEnv())

    class EnvWithoutLabelF:
        def __init__(self, num_agents, n_tokens):
            self.num_agents = num_agents
            self.n_tokens = n_tokens

    try:
        DFAWrapper(env=EnvWithoutLabelF(num_agents=3, n_tokens=10))
        raise RuntimeError("Expected AssertionError for env missing label_f.")
    except AssertionError:
        print("Test completed.")


def accept_distance(dfa):
    # BFS oracle for the shortest distance from the current state to an accepting state, inf if none is reachable
    transitions, labels = dfa.transitions.tolist(), dfa.labels.tolist()
    dist, frontier = {int(dfa.start): 0}, [int(dfa.start)]
    while frontier:
        next_frontier = []
        for s in frontier:
            for t in transitions[s]:
                if t not in dist:
                    dist[t] = dist[s] + 1
                    next_frontier.append(t)
        frontier = next_frontier
    return min((d for s, d in dist.items() if labels[s]), default=float("inf"))


def test_dfa_wrapper_pbrs_args():

    print("Running test for DFAWrapper's gamma and beta arguments.")

    DFAWrapper(env=TokenEnv())
    DFAWrapper(env=TokenEnv(), gamma=0.99, beta=0.9)
    DFAWrapper(env=TokenEnv(), gamma=1.0, beta=0.5)

    for gamma, beta in [(0.99, None), (None, 0.9), (0.99, 1.0), (0.99, 0.0), (1.5, 0.9), (0.0, 0.9)]:
        try:
            DFAWrapper(env=TokenEnv(), gamma=gamma, beta=beta)
            raise RuntimeError(f"Expected AssertionError for gamma={gamma} and beta={beta}.")
        except AssertionError:
            pass

    print("Test completed.")


def test_dfa_wrapper_pbrs(env, gamma, beta, n=20, seed=42, **kwargs):
    # The reward difference between the shaped wrapper and the same wrapper without shaping is the shaping term
    shaped = DFAWrapper(env=env, gamma=gamma, beta=beta, **kwargs)
    base = DFAWrapper(env=env, **kwargs)

    print(f"Running PBRS tests for DFAWrapper(env={env.name}, n_agents={env.num_agents}, gamma={gamma}, beta={beta}).")
    key = jax.random.PRNGKey(seed)

    def potential(dfa, done):
        return 0.0 if done else beta ** accept_distance(dfa)

    def assert_same_except_potentials(state, base_state):
        assert jax.tree.all(jax.tree.map(jnp.array_equal, state.replace(potentials=base_state.potentials), base_state))
        assert all(base_state.potentials[agent] == 0 for agent in env.agents)

    def check_reset(state, base_state):
        # Agents with a trivial goal start done, so their potential is zero from the start
        assert_same_except_potentials(state, base_state)
        dones = {agent: bool(state.dfas[agent].n_states <= 1) for agent in env.agents}
        for agent in env.agents:
            expected = potential(state.dfas[agent], dones[agent])
            assert 0.0 <= expected <= 1.0
            assert jnp.allclose(state.potentials[agent], expected)
        return {agent: float(state.potentials[agent]) for agent in env.agents}, dones

    key, subkey = jax.random.split(key)
    _, state = shaped.reset(subkey)
    _, base_state = base.reset(subkey)
    init_potentials, prev_dones = check_reset(state, base_state)
    returns = {agent: 0.0 for agent in env.agents}
    t, episodes, n_progress = 0, 0, 0

    while episodes < n:
        key, action_key, step_key = jax.random.split(key, 3)
        action_keys = jax.random.split(action_key, env.num_agents)
        actions = {agent: shaped.action_space(agent).sample(k) for agent, k in zip(env.agents, action_keys)}

        # step auto-resets on __all__, which also checks that the next episode starts from the reset potentials
        obs, new_state, rewards, dones, infos = shaped.step(step_key, state, actions)
        base_obs, new_base_state, base_rewards, base_dones, base_infos = base.step(step_key, base_state, actions)

        assert jax.tree.all(jax.tree.map(jnp.array_equal, obs, base_obs))
        assert jax.tree.all(jax.tree.map(jnp.array_equal, dones, base_dones))
        assert_same_except_potentials(new_state, new_base_state)

        # infos holds the reward without shaping and the shaping term, zero without shaping
        assert set(infos.keys()) == set(base_infos.keys()) == {"unshaped_reward", "pbrs_reward"}
        assert infos["unshaped_reward"].shape == infos["pbrs_reward"].shape == (env.num_agents,)
        assert jnp.array_equal(infos["unshaped_reward"], base_infos["unshaped_reward"])
        assert jnp.array_equal(base_infos["unshaped_reward"], jnp.array([base_rewards[agent] for agent in env.agents]))
        assert jnp.array_equal(base_infos["pbrs_reward"], jnp.zeros(env.num_agents))

        for i, agent in enumerate(env.agents):
            shaping = float(rewards[agent] - base_rewards[agent])
            prev_potential = float(state.potentials[agent])
            # Every agent is done on __all__, and new_state is then already the reset state
            next_potential = 0.0 if dones["__all__"] else potential(new_state.dfas[agent], bool(dones[agent]))
            assert 0.0 <= next_potential <= 1.0
            assert abs(float(infos["pbrs_reward"][i]) - shaping) < 1e-5
            assert abs(float(rewards[agent] - infos["unshaped_reward"][i] - infos["pbrs_reward"][i])) < 1e-5
            if not dones["__all__"]:
                assert jnp.allclose(new_state.potentials[agent], next_potential)

            assert abs(shaping - (gamma * next_potential - prev_potential)) < 1e-5, \
                f"Shaping is {shaping} but gamma * {next_potential} - {prev_potential} was expected."
            if prev_dones[agent] and dones[agent]:
                assert shaping == 0, f"{agent} is done but got shaping {shaping}."
            n_progress += (not dones[agent]) and (next_potential != prev_potential)

            returns[agent] += gamma ** t * shaping
            # The discounted shaping telescopes to -potential(s_0) at the agent's own termination, and is zero afterwards
            if dones[agent]:
                assert abs(returns[agent] + init_potentials[agent]) < 1e-4, \
                    f"{agent} has discounted shaping {returns[agent]} but {-init_potentials[agent]} was expected."

        if dones["__all__"]:
            init_potentials, prev_dones = check_reset(new_state, new_base_state)
            returns = {agent: 0.0 for agent in env.agents}
            t, episodes = 0, episodes + 1
            print(f"Test completed for {episodes} samples.", end="\r")
        else:
            prev_dones = {agent: bool(dones[agent]) for agent in env.agents}
            t += 1
            assert t <= env.max_steps_in_episode

        state, base_state = new_state, new_base_state

    assert n_progress > 0, "Shaping never changed between non-terminal states."
    print(f"Test completed for {n} samples.")


if __name__ == '__main__':

    test(DFABisimEnv())
    test(DFADynBisimEnv())
    test_dfa_wrapper_requires_label_f()
    test_dfa_wrapper_pbrs_args()
    test_dfa_wrapper_pbrs(TokenEnv(n_agents=1, max_steps_in_episode=30), gamma=0.9, beta=0.5)
    test_dfa_wrapper_pbrs(TokenEnv(n_agents=3, max_steps_in_episode=30), gamma=0.99, beta=0.9, binary_reward=False)
    test_dfa_wrapper_pbrs(TokenEnv(n_agents=3, max_steps_in_episode=30), gamma=1.0, beta=0.5,
                          sampler=RADSampler(n_tokens=5), dynamic_alphabet=True)
    drone_env = DroneEnv(n_agents=3, max_steps_in_episode=20)
    test_dfa_wrapper_pbrs(drone_env, gamma=0.99, beta=0.9, sampler=RADSampler(n_tokens=drone_env.n_tokens))
