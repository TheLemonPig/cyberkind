from envs.gridworld import register 

import gymnasium
from envs.gridworld.agents import Agent

def main():
    # 1) Create two agents with different start positions
    alice = Agent(agent_id='alice', start_pos=[0, 0], vision_range=1, max_hunger=15.0)
    bob   = Agent(agent_id='bob',   start_pos=[4, 4], vision_range=1, max_hunger=15.0)

    # 2) Make the environment (this triggers register.py behind the scenes)
    env = gymnasium.make(
        'GridWorldMultiAgent-v0',
        size=5,
        agents=[alice, bob],
        resources=None  # will spawn defaults in env
    )

    # 3) Run a short random-action loop
    obs, info = env.reset()
    print("Initial Observations:")
    for agent_id, patch in obs.items():
        print(f"{agent_id} sees:\n{patch}\n")

    for step_idx in range(10):
        # Random action for each agent
        action_dict = {
            'alice': env.action_space.spaces['alice'].sample(),
            'bob':   env.action_space.spaces['bob'].sample()
        }

        obs, rewards, done, info = env.step(action_dict)
        print(f"Step {step_idx + 1}:")
        print("  Actions:", action_dict)
        print("  Rewards:", rewards)
        print("  Done:", done)
        print("  Next Observations:")
        for agent_id, patch in obs.items():
            print(f"    {agent_id} sees:\n{patch}\n")

        # Simple text render
        env.render()

        if done:
            print("All agents starved. Episode over.")
            break

    env.close()

if __name__ == "__main__":
    main()
