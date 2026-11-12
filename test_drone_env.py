import math
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


def simulate_tracking_step(ref_position, ref_velocity, ref_acceleration, position, velocity, acceleration, action,
                           POLICY_DT, FILTER_DT, TAU, MAX_ACCEL, VELOCITY_SCALE, KP, KV, MOTOR_TAU):
    """The collaborator's point-mass model of the stack's controller tracking the
    velocity filter: their per-substep loop, with `simulate_step`'s filter substep as
    the reference's step. Single agent, float64, no geofence.
    Returns ((ref_p, ref_v, ref_a), (p, v, a)).
    """
    target_velocity = VELOCITY_SCALE * np.clip(action, -1.0, 1.0)
    kp, kv, tau = np.asarray(KP), np.asarray(KV), np.asarray(MOTOR_TAU)

    ref_p, ref_v, ref_a = ref_position.copy(), ref_velocity.copy(), ref_acceleration.copy()
    p, v, a = position.copy(), velocity.copy(), acceleration.copy()

    n = int(np.ceil(POLICY_DT / FILTER_DT))
    h = POLICY_DT / n

    for _ in range(n):
        # Advance the existing reference filter.
        jerk = (target_velocity - ref_v) / TAU**2 - 2.0 * ref_a / TAU
        ref_a_next = ref_a + h * jerk
        magnitude = np.linalg.norm(ref_a_next)
        if magnitude > MAX_ACCEL:
            ref_a_next *= MAX_ACCEL / magnitude
        ref_v_next = ref_v + h * ref_a_next
        ref_p += 0.5 * h * (ref_v + ref_v_next)
        ref_v = ref_v_next
        ref_a = ref_a_next

        # Advance actual point-mass tracking.
        a_request = ref_a + kp * (ref_p - p) + kv * (ref_v - v)
        a += (1 - np.exp(-h / tau)) * (a_request - a)
        v_next = v + h * a
        p += 0.5 * h * (v + v_next)
        v = v_next

    return (ref_p, ref_v, ref_a), (p, v, a)


def filter_constants(env):
    """`simulate_step`'s deployment constants for `env`."""
    return dict(POLICY_DT=env.dt, FILTER_DT=env.filter_dt, TAU=env.velocity_filter_tau,
                MAX_ACCEL=env.max_filter_accel, VELOCITY_SCALE=env.max_speed)


def tracking_constants(env):
    """`simulate_tracking_step`'s constants for `env`."""
    return dict(**filter_constants(env), KP=env.tracking_kp, KV=env.tracking_kv, MOTOR_TAU=env.motor_tau)


def expected_tracking_step(env, state, actions):
    """`simulate_tracking_step` for every agent from `state`, stacked over agents:
    returns (ref_positions, ref_velocities, ref_accelerations, positions, velocities, accelerations)."""
    steps = [
        simulate_tracking_step(
            to_np(state.ref_positions[i]), to_np(state.ref_velocities[i]), to_np(state.ref_accelerations[i]),
            to_np(state.positions[i]), to_np(state.velocities[i]), to_np(state.accelerations[i]),
            to_np(actions[agent]), **tracking_constants(env),
        )
        for i, agent in enumerate(env.agents)
    ]
    return tuple(np.stack(x) for x in zip(*(ref + drone for ref, drone in steps)))


def assert_tracking_state(state, expected, atol):
    """Asserts `state`'s reference and drone arrays match `expected_tracking_step`'s output."""
    names = ("ref_positions", "ref_velocities", "ref_accelerations", "positions", "velocities", "accelerations")
    for name, value, tol in zip(names, expected, atol):
        assert np.allclose(getattr(state, name), value, atol=tol), (name, np.abs(to_np(getattr(state, name)) - value).max())


def to_np(x):
    return np.asarray(x, dtype=np.float64)


def test(env, n=100, seed=42):
    """Checks DroneEnv.step_env against an independent recomputation: unclipped
    dynamics (the deployment's `simulate_step` for the velocity filter, and
    `simulate_tracking_step` for the tracking controller), and an episode that ends
    on the step any agent leaves [low, high], with `out_of_bounds_reward` for each
    agent that left.
    """

    print(f"Running tests for {env.name} (n_agents={env.n_agents}, use_displacement_action={env.use_displacement_action}, "
          f"use_velocity_filter={env.use_velocity_filter}, use_tracking_controller={env.use_tracking_controller}, "
          f"out_of_bounds_reward={env.out_of_bounds_reward}).")
    key = jax.random.PRNGKey(seed)
    low, high = to_np(env.low), to_np(env.high)
    n_out_of_bounds = 0

    for i in range(n):

        key, subkey = jax.random.split(key)
        _, state = env.reset(key=subkey)
        assert jnp.all(state.positions >= env.low) and jnp.all(state.positions <= env.high)
        assert jnp.all(state.velocities == 0.0)
        assert jnp.all(state.accelerations == 0.0)
        assert jnp.array_equal(state.prev_positions, state.positions)
        assert jnp.array_equal(state.ref_positions, state.positions)
        assert jnp.all(state.ref_velocities == 0.0) and jnp.all(state.ref_accelerations == 0.0)
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

            if env.use_tracking_controller:
                # Recompute in float64 from the same prior state, as for the velocity filter.
                expected = expected_tracking_step(env, state, actions)
                assert_tracking_state(new_state, expected, atol=(1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-4))
                expected_positions, expected_velocities = expected[3], expected[4]
                assert jnp.all(jnp.linalg.norm(new_state.ref_accelerations, axis=-1) <= env.max_filter_accel + 1e-5)
                # The reference stays within max_speed; the drone tracking it, within vel_gain times that.
                assert np.all(np.abs(expected[1]) <= env.max_speed + 1e-6)
            elif env.use_velocity_filter:
                # Recompute in float64 from the same prior state, so only one step of
                # float32 rounding separates this from the jitted step_env.
                expected = [
                    simulate_step(
                        to_np(state.positions[i]), to_np(state.velocities[i]), to_np(state.accelerations[i]),
                        to_np(actions[agent]), **filter_constants(env),
                    )
                    for i, agent in enumerate(env.agents)
                ]
                expected_positions, expected_velocities, expected_accelerations = (np.stack(x) for x in zip(*expected))
                assert np.allclose(new_state.positions, expected_positions, atol=1e-5)
                assert np.allclose(new_state.velocities, expected_velocities, atol=1e-5)
                assert np.allclose(new_state.accelerations, expected_accelerations, atol=1e-4)
                assert jnp.all(jnp.linalg.norm(new_state.accelerations, axis=-1) <= env.max_filter_accel + 1e-5)
            else:
                clipped = np.clip(np.stack([to_np(actions[a]) for a in env.agents]), -env.max_action, env.max_action)
                delta = clipped if env.use_displacement_action else clipped * env.dt
                expected_positions = to_np(state.positions) + delta
                expected_velocities = delta / env.dt
                assert np.allclose(new_state.positions, expected_positions, atol=1e-5)
                assert np.allclose(new_state.velocities, expected_velocities, atol=1e-4)
                assert jnp.all(new_state.accelerations == 0.0)
            if not env.use_tracking_controller:
                assert jnp.all(new_state.ref_positions == 0.0) and jnp.all(new_state.ref_velocities == 0.0)
                assert jnp.all(new_state.ref_accelerations == 0.0)
            assert np.all(np.abs(expected_velocities) <= env.max_speed * env.vel_gain + 1e-6)
            assert jnp.array_equal(new_state.prev_positions, state.positions)

            for i, agent in enumerate(env.agents):
                assert np.allclose(obs[agent], np.concatenate([expected_positions[i], expected_velocities[i]]), atol=1e-4)
                assert jnp.array_equal(obs[agent], jnp.concatenate([new_state.positions[i], new_state.velocities[i]]))
                assert env.observation_space(agent).contains(obs[agent])

            # From the env's own positions, so float32 rounding can't flip a near-wall verdict.
            positions = to_np(new_state.positions)
            out_of_bounds = np.any((positions < low) | (positions > high), axis=-1)
            expected_done = bool(new_state.time >= env.max_steps_in_episode or out_of_bounds.any())
            for i, agent in enumerate(env.agents):
                assert rewards[agent] == (env.out_of_bounds_reward if out_of_bounds[i] else 0.0)
                assert dones[agent] == expected_done
            assert dones["__all__"] == expected_done
            if not expected_done:
                assert np.all((positions >= low) & (positions <= high))
            n_out_of_bounds += bool(out_of_bounds.any())

            state = new_state
            done = dones["__all__"]
            steps += 1
            assert steps <= env.max_steps_in_episode

        print(f"Test completed for {i + 1} samples.", end="\r")

    assert n_out_of_bounds > 0, "No episode ended by leaving the box."
    print(f"Test completed for {n} samples ({n_out_of_bounds} ended by leaving the box).")


def test_filter_matches_reference(env, n=10, seed=42):
    """Checks DroneEnv's velocity filter against the deployment's `simulate_step`.

    `env` must have a box large enough that no drone leaves it, so every episode runs
    to max_steps_in_episode. Each step is compared both one-step (the reference
    run from the env's own prior state) and open-loop (a float64 reference
    trajectory carried across the whole episode, as in the deployment rollout).
    """

    assert env.use_velocity_filter
    print(f"Running filter reference tests for {env.name} (n_agents={env.n_agents}, tau={env.velocity_filter_tau}, "
          f"max_filter_accel={env.max_filter_accel}, filter_dt={env.filter_dt}, max_speed={env.max_speed}).")
    constants = filter_constants(env)
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
                assert np.all(p > to_np(env.low)) and np.all(p < to_np(env.high)), "drone left the box; use a larger box"

                assert jnp.array_equal(obs[agent], jnp.concatenate([new_state.positions[j], new_state.velocities[j]]))

            state = new_state
            done = dones["__all__"]

        print(f"Test completed for {i + 1} samples.", end="\r")

    print(f"Test completed for {n} samples.")


def test_tracking_matches_reference(env, n=10, seed=42):
    """Checks DroneEnv's tracking controller against the collaborator's
    `simulate_tracking_step`, like `test_filter_matches_reference` (one-step and
    open-loop; `env` needs a box no drone leaves). Also checks that the reference is
    exactly what the velocity filter alone would do with the same actions, and that
    the drone genuinely lags it while staying close.
    """

    assert env.use_tracking_controller
    print(f"Running tracking reference tests for {env.name} (n_agents={env.n_agents}, tracking_kp={env.tracking_kp}, "
          f"tracking_kv={env.tracking_kv}, motor_tau={env.motor_tau}, filter_dt={env.filter_dt}, max_speed={env.max_speed}).")
    filter_env = DroneEnv(
        n_agents=env.n_agents, x_low=float(env.low[0]), x_high=float(env.high[0]), y_low=float(env.low[1]),
        y_high=float(env.high[1]), z_low=float(env.low[2]), z_high=float(env.high[2]), max_speed=env.max_speed,
        dt=env.dt, use_velocity_filter=True, velocity_filter_tau=env.velocity_filter_tau,
        max_filter_accel=env.max_filter_accel, filter_dt=env.filter_dt, max_steps_in_episode=env.max_steps_in_episode,
    )
    key = jax.random.PRNGKey(seed)
    max_lag = 0.0

    for i in range(n):

        key, subkey = jax.random.split(key)
        _, state = env.reset(key=subkey)
        _, filter_state = filter_env.reset(key=subkey)
        trajectory = [((np.zeros(3),) * 3, (np.zeros(3),) * 3) for _ in env.agents]
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
            _, filter_state, _, _, _ = filter_env.step_env(step_key, filter_state, actions)

            assert_tracking_state(new_state, expected_tracking_step(env, state, actions),
                                  atol=(1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-4))

            for j, agent in enumerate(env.agents):
                trajectory[j] = simulate_tracking_step(*trajectory[j][0], *trajectory[j][1], to_np(actions[agent]),
                                                       **tracking_constants(env))
                (ref_p, ref_v, ref_a), (p, v, a) = trajectory[j]
                for value, expected, tol in zip(
                    (new_state.ref_positions, new_state.ref_velocities, new_state.ref_accelerations,
                     new_state.positions, new_state.velocities, new_state.accelerations),
                    (ref_p, ref_v, ref_a, p, v, a), (1e-4, 1e-4, 1e-3, 1e-4, 1e-4, 1e-3),
                ):
                    assert np.allclose(value[j], expected, atol=tol), (agent, np.abs(to_np(value[j]) - expected).max())
                assert np.all(p > to_np(env.low)) and np.all(p < to_np(env.high)), "drone left the box; use a larger box"
                assert jnp.array_equal(obs[agent], jnp.concatenate([new_state.positions[j], new_state.velocities[j]]))
                assert np.all(np.abs(v) <= env.max_speed * env.vel_gain + 1e-6)

            # The reference is the velocity filter alone, driven by the same actions.
            assert np.allclose(new_state.ref_positions, filter_state.positions, atol=1e-5)
            assert np.allclose(new_state.ref_velocities, filter_state.velocities, atol=1e-5)
            assert np.allclose(new_state.ref_accelerations, filter_state.accelerations, atol=1e-4)
            max_lag = max(max_lag, float(jnp.max(jnp.abs(new_state.ref_positions - new_state.positions))))

            state = new_state
            done = dones["__all__"]

        print(f"Test completed for {i + 1} samples.", end="\r")

    # The drone lags its reference, but tracks it.
    assert 1e-3 < max_lag < 0.5 * env.max_speed, max_lag
    print(f"Test completed for {n} samples (max |ref_position - position| = {max_lag:.3f}).")


def test_tracking_args():
    """Checks DroneEnv's tracking-controller argument checks, and that the default
    gains' velocity gain is the impulse response's L1 norm (> 1: the drone overshoots)."""

    print("Running tracking controller argument tests for DroneEnv.")
    for kwargs in (
        dict(use_tracking_controller=True),  # needs the velocity filter as its reference
        dict(use_tracking_controller=True, use_displacement_action=True),
        dict(use_velocity_filter=True, use_tracking_controller=True, tracking_kp=(7.1, 7.1)),
        dict(use_velocity_filter=True, use_tracking_controller=True, motor_tau=(0.08, 0.0, 0.06)),
        dict(use_velocity_filter=True, use_tracking_controller=True, tracking_kv=(-2.4, 2.4, 6.7)),  # unstable
    ):
        try:
            DroneEnv(**kwargs)
        except AssertionError:
            continue
        raise AssertionError(f"DroneEnv({kwargs}) did not raise")

    env = DroneEnv(use_velocity_filter=True, use_tracking_controller=True)
    assert env.vel_gain[0] == env.vel_gain[1] and np.all(env.vel_gain > 1.0), env.vel_gain
    assert np.all(DroneEnv(use_velocity_filter=True).vel_gain == 1.0)
    # The gains don't affect the other modes.
    assert np.all(DroneEnv(use_velocity_filter=True, tracking_kv=(-2.4, 2.4, 6.7)).vel_gain == 1.0)
    print(f"Test completed (default vel_gain = {env.vel_gain}).")


def test_label_f_on_entry():
    """Checks that label_f fires a token only on the step its region is entered (no
    "sticky color"): staying inside yields -1, and re-triggering requires exiting first.

    Scripted displacement actions land each agent exactly on region centers (looked
    up from env.label_regions()), well away from region boundaries.
    """

    print("Running label_f on-entry tests for DroneEnv.")
    env = DroneEnv(n_agents=2, use_displacement_action=True, max_speed=8.0)
    regions = env.label_regions()

    def center(index):
        _, kind, params = regions[index]
        if kind == "circle":
            return float(params[0]), float(params[1])
        return 0.5 * float(params[0] + params[1]), 0.5 * float(params[2] + params[3])

    # label_regions() order: corners 0-3 (token 0), edge midpoints 4-7 (token 1), corridors 8-15 (tokens 2-9).
    corner_0, bottom_mid, corridor_2 = center(0), center(4), center(8)
    assert [regions[k][0] for k in (0, 4, 8)] == [0, 1, 2]
    origin = (0.0, 0.0)
    waypoints = {
        # Enter the bottom edge-midpoint circle (token 1), stay twice, exit to the origin, re-enter.
        "agent_0": [bottom_mid, bottom_mid, bottom_mid, origin, bottom_mid],
        # Enter corridor 2, move into corner 0, back into corridor 2, stay.
        "agent_1": [corridor_2, corner_0, corridor_2, corridor_2, corridor_2],
    }
    expected_labels = {"agent_0": [1, -1, -1, -1, 1], "agent_1": [2, 0, 2, -1, -1]}
    # Displacements between consecutive waypoints (z stays at 0, inside every region's z range).
    moves = {
        agent: [(x1 - x0, y1 - y0, 0.0) for (x0, y0), (x1, y1) in zip([origin] + points, points)]
        for agent, points in waypoints.items()
    }
    assert all(abs(d) <= env.max_action for agent_moves in moves.values() for move in agent_moves for d in move)

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


def test_label_regions(env):
    """Checks the label-region layout: every region lies (in x/y) inside
    [low + region_margin, high - region_margin], and neighboring regions are exactly
    region_gap apart (no pair closer, and some pair at exactly that distance).
    """

    print(f"Running label region tests for {env.name} (low={env.low.tolist()}, high={env.high.tolist()}, "
          f"region_margin={env.region_margin}, region_gap={env.region_gap}).")
    inner_low = to_np(env.low[:2]) + env.region_margin
    inner_high = to_np(env.high[:2]) - env.region_margin
    z_mid = 0.5 * float(env.low[2] + env.high[2])

    shapes = []
    for token, kind, params in env.label_regions():
        params = [float(v) for v in params]
        assert params[-2:] == [z_mid - 0.75, z_mid + 0.25]
        if kind == "circle":
            cx, cy, r = params[:3]
            assert r > 0
            lo, hi = np.array([cx - r, cy - r]), np.array([cx + r, cy + r])
        else:
            x_lo, x_hi, y_lo, y_hi = params[:4]
            assert x_lo < x_hi and y_lo < y_hi
            lo, hi = np.array([x_lo, y_lo]), np.array([x_hi, y_hi])
        assert np.all(lo >= inner_low - 1e-6) and np.all(hi <= inner_high + 1e-6), (token, kind, params)
        shapes.append((kind, params[:4] if kind == "rect" else params[:3]))

    def rect_point_dist(rect, x, y):
        x_lo, x_hi, y_lo, y_hi = rect
        return np.hypot(max(x_lo - x, 0.0, x - x_hi), max(y_lo - y, 0.0, y - y_hi))

    def dist(a, b):
        (kind_a, pa), (kind_b, pb) = a, b
        if kind_a == "circle" and kind_b == "circle":
            return np.hypot(pa[0] - pb[0], pa[1] - pb[1]) - pa[2] - pb[2]
        if kind_a == "rect" and kind_b == "rect":
            dx = max(pb[0] - pa[1], pa[0] - pb[1], 0.0)
            dy = max(pb[2] - pa[3], pa[2] - pb[3], 0.0)
            return np.hypot(dx, dy)
        (_, circle), (_, rect) = (a, b) if kind_a == "circle" else (b, a)
        return rect_point_dist(rect, circle[0], circle[1]) - circle[2]

    gaps = [dist(a, b) for i, a in enumerate(shapes) for b in shapes[i + 1:]]
    assert min(gaps) >= env.region_gap - 1e-5, min(gaps)
    assert abs(min(gaps) - env.region_gap) < 1e-5, min(gaps)

    print("Test completed.")


def test_obs_noise(n=20, seed=42, **kwargs):
    """Checks DroneEnv(**kwargs)'s observation noise (kwargs must enable it) against a
    noise-free copy stepped with the same keys and actions: the true state, rewards and
    dones are identical (the noise never leaks into them), and the observation noise is
    zero-mean, truncated to +-3 std, with the std of a +-3-truncated normal, fresh per
    agent and per step, and reproducible from the key.
    """

    env = DroneEnv(**kwargs)
    clean_env = DroneEnv(**{**kwargs, "obs_noise_std": 0.0})
    assert env.use_obs_noise and not clean_env.use_obs_noise
    print(f"Running observation noise tests for {env.name} (n_agents={env.n_agents}, "
          f"use_velocity_filter={env.use_velocity_filter}, use_tracking_controller={env.use_tracking_controller}, "
          f"obs_noise_std={env.obs_noise_std}).")
    scale = to_np(env.obs_noise_scale)
    key = jax.random.PRNGKey(seed)
    noises = []

    def record(obs, clean_obs, state):
        step_noise = []
        for i, agent in enumerate(env.agents):
            truth = jnp.concatenate([state.positions[i], state.velocities[i]])
            assert jnp.array_equal(clean_obs[agent], truth)
            assert env.observation_space(agent).contains(obs[agent])
            step_noise.append(to_np(obs[agent]) - to_np(truth))
        step_noise = np.stack(step_noise)
        assert np.all(np.abs(step_noise) <= env.OBS_NOISE_MAX_SIGMAS * scale + 1e-5)
        if env.n_agents > 1:
            assert not np.allclose(step_noise[0], step_noise[1]), "agents got the same noise"
        if noises:
            assert not np.allclose(step_noise, noises[-1]), "consecutive steps got the same noise"
        noises.append(step_noise)

    for i in range(n):

        key, subkey = jax.random.split(key)
        obs, state = env.reset(key=subkey)
        clean_obs, clean_state = clean_env.reset(key=subkey)
        assert jax.tree.all(jax.tree.map(jnp.array_equal, state, clean_state))
        record(obs, clean_obs, state)
        done = False

        while not done:
            key, action_key, step_key = jax.random.split(key, 3)
            action_keys = jax.random.split(action_key, env.n_agents)
            actions = {agent: env.action_space(agent).sample(k) for agent, k in zip(env.agents, action_keys)}

            prev_state = state
            obs, state, rewards, dones, _ = env.step_env(step_key, prev_state, actions)
            clean_obs, clean_state, clean_rewards, clean_dones, _ = clean_env.step_env(step_key, clean_state, actions)
            assert jax.tree.all(jax.tree.map(jnp.array_equal, state, clean_state))
            assert rewards == clean_rewards and dones == clean_dones
            record(obs, clean_obs, state)

            # Same key, same noise.
            again, _, _, _, _ = env.step_env(step_key, prev_state, actions)
            assert all(jnp.array_equal(again[agent], obs[agent]) for agent in env.agents)
            done = dones["__all__"]

        print(f"Test completed for {i + 1} samples.", end="\r")

    noise = np.concatenate(noises).reshape(-1, 6)
    assert len(noise) >= 1000, f"only {len(noise)} noise samples; raise n"
    # Std of a standard normal truncated to +-k.
    k = env.OBS_NOISE_MAX_SIGMAS
    pdf = math.exp(-0.5 * k ** 2) / math.sqrt(2 * math.pi)
    truncated_std = math.sqrt(1 - 2 * k * pdf / math.erf(k / math.sqrt(2)))
    for j in range(6):
        if scale[j] == 0:
            assert np.all(noise[:, j] == 0)
            continue
        expected_std = scale[j] * truncated_std
        assert abs(noise[:, j].mean()) < 4 * expected_std / math.sqrt(len(noise)), (j, noise[:, j].mean())
        assert abs(noise[:, j].std() / expected_std - 1) < 0.05, (j, noise[:, j].std(), expected_std)

    print(f"Test completed for {n} samples ({len(noise)} noise samples).")


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
            assert jnp.array_equal(new_state.env_state.ref_positions, expected_env_state.ref_positions)
            assert jnp.array_equal(new_state.env_state.ref_velocities, expected_env_state.ref_velocities)
            assert jnp.array_equal(new_state.env_state.ref_accelerations, expected_env_state.ref_accelerations)
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
    parser.add_argument("--test-tracking-reference", action="store_true")
    parser.add_argument("--test-label-f", action="store_true")
    parser.add_argument("--test-regions", action="store_true")
    parser.add_argument("--test-obs-noise", action="store_true")
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
        test(DroneEnv(n_agents=3, out_of_bounds_reward=-1.0))
        test(DroneEnv(n_agents=1, use_velocity_filter=True, out_of_bounds_reward=-1.0))
        tracking = dict(use_velocity_filter=True, use_tracking_controller=True)
        test(DroneEnv(n_agents=1, **tracking))
        test(DroneEnv(n_agents=3, **tracking))
        test(DroneEnv(n_agents=3, max_speed=2.0, x_low=-2.0, x_high=2.0, y_low=-1.5, y_high=1.5, z_low=0.0, z_high=2.0, **tracking))
        test(DroneEnv(n_agents=1, out_of_bounds_reward=-1.0, **tracking))

    if args.test_regions:
        test_label_regions(DroneEnv())
        test_label_regions(DroneEnv(x_low=-2.0, x_high=2.0, y_low=-1.5, y_high=1.5, z_low=0.0, z_high=2.0))
        test_label_regions(DroneEnv(region_margin=0.0, region_gap=0.0))
        test_label_regions(DroneEnv(region_margin=0.4, region_gap=0.05))

    if args.test_obs_noise:
        test_obs_noise(n_agents=3, obs_noise_std=(0.02, 0.1))
        test_obs_noise(n_agents=1, use_velocity_filter=True, obs_noise_std=0.05)
        test_obs_noise(n_agents=2, obs_noise_std=(0.0, 0.1))
        test_obs_noise(n_agents=2, use_velocity_filter=True, use_tracking_controller=True, obs_noise_std=(0.02, 0.1))

    big_box = dict(x_low=-50.0, x_high=50.0, y_low=-50.0, y_high=50.0, z_low=-50.0, z_high=50.0)
    if args.test_filter_reference:
        test_filter_matches_reference(DroneEnv(n_agents=1, use_velocity_filter=True, **big_box))
        test_filter_matches_reference(DroneEnv(n_agents=3, use_velocity_filter=True, max_speed=2.0, velocity_filter_tau=0.2,
                                               max_filter_accel=5.0, filter_dt=0.01, **big_box))

    if args.test_tracking_reference:
        test_tracking_args()
        test_tracking_matches_reference(DroneEnv(n_agents=1, use_velocity_filter=True, use_tracking_controller=True, **big_box))
        test_tracking_matches_reference(DroneEnv(n_agents=3, use_velocity_filter=True, use_tracking_controller=True,
                                                 max_speed=2.0, velocity_filter_tau=0.2, max_filter_accel=5.0, filter_dt=0.01,
                                                 tracking_kp=(5.0, 6.0, 10.0), tracking_kv=(2.0, 3.0, 5.0),
                                                 motor_tau=(0.05, 0.1, 0.07), **big_box))

    if args.test_label_f:
        test_label_f_on_entry()

    if args.test_dfa_wrapper:
        env = DroneEnv(n_agents=1, max_steps_in_episode=20)
        test_dfa_wrapper(DFAWrapper(env=env, sampler=RADSampler(n_tokens=env.n_tokens)))

        env = DroneEnv(n_agents=3, max_steps_in_episode=20)
        test_dfa_wrapper(DFAWrapper(env=env, sampler=RADSampler(n_tokens=env.n_tokens)))

        env = DroneEnv(n_agents=3, use_velocity_filter=True, max_steps_in_episode=20)
        test_dfa_wrapper(DFAWrapper(env=env, sampler=RADSampler(n_tokens=env.n_tokens)))

        env = DroneEnv(n_agents=3, obs_noise_std=(0.02, 0.1), max_steps_in_episode=20)
        test_dfa_wrapper(DFAWrapper(env=env, sampler=RADSampler(n_tokens=env.n_tokens)))

        env = DroneEnv(n_agents=3, use_velocity_filter=True, use_tracking_controller=True, max_steps_in_episode=20)
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
