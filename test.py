import jax
from dfa_gym import TokenEnv, DFABisimEnv, DFAWrapper

def test(env):

    key = jax.random.PRNGKey(1133)

    n = 100

    for i in range(n):

        key, subkey = jax.random.split(key)
        obs, state = env.reset(key=subkey)
        # env.render(state)
        done = False
        steps = 0

        while not done:
            keys = jax.random.split(key, env.num_agents + 1)
            key, subkeys = keys[0], keys[1:]
            actions = {agent: env.action_space(agent).sample(subkeys[i]) for i, agent in enumerate(env.agents)}
            # actions = {}
            # for agent in env.agents:
            #     actions[agent] = int(input(f"Action for {agent}\n"))
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(actions=actions, state=state, key=subkey)
            # env.render(state)
            # print(obs)
            # jax.numpy.set_printoptions(threshold=10000)
            # print(obs)
            # for i in obs:
            #     print(i)
            #     print(obs[i])
            #     print(obs[i].shape)
            # print("Label:", env.env.label_f(state.env_state))
            # print(rewards)
            # print(dones)
            # print(actions)
            # env.render(state)
            # input()
            done = dones["__all__"]
            steps += 1

        print(f"Test completed for {i + 1} samples.", end="\r")

    print(f"Test completed for {n} samples.")

# layout = """
# 8....#0....1
# .....#......
# ..b..#......
# .....#3....2
# .....#####a#
# A...........
# B...........
# .....#####b#
# .....#4....5
# ..a..#......
# .....#......
# 9....#7....6
# """

# layout = """
# [ 8 ][   ][   ][   ][   ][   ][   ][ # ][ 0 ][   ][   ][ 1 ]
# [   ][   ][   ][   ][   ][   ][   ][ # ][   ][   ][   ][   ]
# [   ][   ][ b ][   ][   ][   ][   ][ # ][   ][   ][   ][   ]
# [   ][   ][   ][   ][   ][   ][   ][ # ][ 3 ][   ][   ][ 2 ]
# [   ][   ][   ][   ][   ][   ][   ][ # ][ # ][ # ][#,a][ # ]
# [   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ A ][   ]
# [   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ]
# [   ][   ][   ][   ][   ][   ][   ][ # ][ # ][ # ][#,b][ # ]
# [   ][   ][ B ][   ][   ][   ][   ][ # ][ 4 ][   ][   ][ 5 ]
# [   ][   ][ a ][   ][   ][   ][   ][ # ][   ][   ][   ][   ]
# [   ][   ][   ][   ][   ][   ][   ][ # ][   ][   ][   ][   ]
# [ 9 ][   ][   ][   ][   ][   ][   ][ # ][ 7 ][   ][   ][ 6 ]
# """

# layout = """
#     [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
#     [ # ][ 8 ][   ][   ][   ][   ][   ][ # ][ 0 ][   ][ 1 ][ # ]
#     [ # ][   ][   ][   ][   ][   ][   ][ # ][   ][   ][   ][ # ]
#     [ # ][   ][ b ][   ][   ][   ][   ][ # ][ 3 ][   ][ 2 ][ # ]
#     [ # ][   ][   ][   ][   ][   ][   ][ # ][ # ][#,a][ # ][ # ]
#     [ # ][ A ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ # ]
#     [ # ][ B ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ # ]
#     [ # ][   ][   ][   ][   ][   ][   ][ # ][ # ][#,b][ # ][ # ]
#     [ # ][   ][ a ][   ][   ][   ][   ][ # ][ 4 ][   ][ 5 ][ # ]
#     [ # ][   ][   ][   ][   ][   ][   ][ # ][   ][   ][   ][ # ]
#     [ # ][ 9 ][   ][   ][   ][   ][   ][ # ][ 7 ][   ][ 6 ][ # ]
#     [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
# """


# layout = """
#     [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
#     [ # ][   ][   ][   ][   ][   ][   ][#,a][ 0 ][   ][ 1 ][ # ]
#     [ # ][   ][   ][ b ][ b ][ b ][   ][#,a][   ][ 4 ][   ][ # ]
#     [ # ][   ][   ][ b ][ b ][ b ][   ][#,a][ 3 ][   ][ 2 ][ # ]
#     [ # ][   ][   ][ b ][ b ][ b ][   ][#,a][#,a][#,a][#,a][ # ]
#     [ # ][ A ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ # ]
#     [ # ][ B ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ # ]
#     [ # ][   ][   ][ a ][ a ][ a ][   ][#,b][#,b][#,b][#,b][ # ]
#     [ # ][   ][   ][ a ][ a ][ a ][   ][#,b][ 5 ][   ][ 6 ][ # ]
#     [ # ][   ][   ][ a ][ a ][ a ][   ][#,b][   ][ 9 ][   ][ # ]
#     [ # ][   ][   ][   ][   ][   ][   ][#,b][ 8 ][   ][ 7 ][ # ]
#     [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
# """

layout = """
    [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    [ # ][ 0 ][   ][ 2 ][ # ][ 1 ][   ][ 3 ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][ a ][ 8 ][   ][#,a][ B ][ 9 ][ a ][ # ]
    [ # ][   ][   ][   ][ # ][   ][ A ][   ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][ 6 ][   ][ 4 ][ # ][ 7 ][   ][ 5 ][ # ]
    [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    """

if __name__ == '__main__':
    # test(env=TokenEnv(is_circular=True, is_walled=True, collision_reward=-1e2))
    # test(env=DFABisimEnv())
    # test(env=DFAWrapper(env=TokenEnv(grid_shape=(4,7), n_token_repeat=1, n_agents=2, is_circular=False, is_walled=True)))
    from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler
    test(env=DFAWrapper(
        TokenEnv(layout=layout, max_steps_in_episode=200),
        sampler=ReachAvoidSampler(max_size=4, p=None, prob_stutter=1.0)
        )
    )
    # env=TokenEnv(layout=layout)
    # print(env.init_state)
    # env.render(env.init_state)
    # input()

