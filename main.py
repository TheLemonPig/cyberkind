from envs.gridworld import register 
from viz_utils import launch_visualization
import gymnasium
from envs.gridworld.agents import Agent

def main():
    # 1) Define agent parameters and create agents list
    agent_configs = [
        {'agent_id': 'alice', 'start_pos': [0, 0], 'vision_range': 1, 'max_hunger': 15.0},
        # {'agent_id': 'bob',   'start_pos': [4, 4], 'vision_range': 1, 'max_hunger': 15.0}
    ]
    agents = [Agent(**cfg) for cfg in agent_configs]

    # 2) Make the environment
    grid_size = 5  # Define the size of the grid
    env = gymnasium.make(
        'GridWorldMultiAgent-v0',
        size=grid_size,
        agents=agents,
        resources=None,
        end_on_starve=False
    )

    # Helper to capture full state snapshot
    def capture_snapshot(env):
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        agent_positions = {a.agent_id: tuple(a.position) for a in base_env.agents}
        agent_orientations = {a.agent_id: a.orientation for a in base_env.agents}
        resource_states = []
        for res in base_env.resources:
            resource_states.append({
                'position': tuple(res.position),
                'timer': res.timer,
                'type': res.type
            })
        return {
            'agent_positions': agent_positions,
            'agent_orientations': agent_orientations,
            'resource_states': resource_states
        }

    # 3) Run a short random‐action loop and record histories for replay
    snapshots, obs_history, action_history, reward_history, done_history = [], [], [], [], []

    obs, info = env.reset()
    snapshots.append(capture_snapshot(env))
    obs_history.append(obs)

    print("Initial Observations:")
    for agent_id, patch in obs.items():
        print(f"{agent_id} sees:\n{patch}\n")

    for step_idx in range(100):
        # sample every agent’s action
        action_dict = {
            a.agent_id: env.action_space.spaces[a.agent_id].sample()
            for a in agents
        }
        obs, rewards, done, info = env.step(action_dict)

        snapshots.append(capture_snapshot(env))
        action_history.append(action_dict)
        obs_history.append(obs)
        reward_history.append(rewards)
        done_history.append(done)

        print(f"Step {step_idx + 1}:")
        print("  Actions:", action_dict)
        print("  Rewards:", rewards)
        print("  Done:", done)
        print("  Next Observations:")
        for agent_id, patch in obs.items():
            print(f"    {agent_id} sees:\n{patch}\n")

        env.render()
        if done:
            print("All agents starved. Episode over.")
            break

    # 4) Launch visualization in replay mode
    launch_visualization(snapshots, obs_history, grid_size=grid_size)
    env.close()


if __name__ == "__main__":
    main()