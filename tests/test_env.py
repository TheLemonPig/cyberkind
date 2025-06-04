from envs.gridworld.env import GridWorldEnv


def test_env_step():
    env = GridWorldEnv(size=2)
    obs = env.reset()
    obs2, reward, done, _ = env.step(1)
    assert obs2.shape == (2, 2)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
