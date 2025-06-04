from gymnasium.envs.registration import register

register(
    id='GridWorldMultiAgent-v0',
    entry_point='envs.gridworld.env:GridWorldEnv',
)

