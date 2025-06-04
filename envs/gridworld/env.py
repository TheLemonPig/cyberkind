import gymnasium
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from gymnasium.spaces import Discrete, Box, Dict
from .agents import Agent

class Resource:
    """
    Represents one resource (e.g. food) in the grid.
    - position: [x, y]
    - value: amount by which hunger is reduced when eaten
    - respawn_time: number of steps after consumption before it reappears
    """
    def __init__(self, resource_type, position, value, respawn_time):
        self.type = resource_type         # e.g. 'food'
        self.position = list(position)    # [x, y]
        self.value = value                # e.g. 5.0 hunger units
        self.respawn_time = respawn_time  # e.g. 10 steps
        self.timer = 0                    # 0 means available

    def step(self):
        """Decrease timer until resource is available again."""
        if self.timer > 0:
            self.timer -= 1

    def is_available(self):
        """True if timer == 0 (so the resource can be eaten)."""
        return self.timer == 0

    def consume(self):
        """
        Called when an agent eats this resource.
        Sets timer = respawn_time so it disappears until timer counts down.
        """
        self.timer = self.respawn_time


class GridWorldEnv(gymnasium.Env):
    """
    A multi-agent GridWorld where:
      - Agents have hunger needs and a local field of view.
      - Resources (food) appear and respawn over time.
      - Agents gain positive reward for eating, negative reward for hunger.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, size=5, agents=None, resources=None):
        """
        :param size: Grid dimension (size x size).
        :param agents: List of Agent instances (if None, user must pass them in).
        :param resources: List of Resource instances (if None, we auto-spawn default food).
        """
        super().__init__()
        self.size = size

        # Action/Observation spaces will be Dicts, once we know agent IDs.
        self.action_space = None
        self.observation_space = None

        # Flag to indicate episode end (all agents starved).
        self.done = False

        # Lists of Agent and Resource objects
        self.agents = agents if agents is not None else []
        self.resources = resources if resources is not None else []

        # If no resources were provided, spawn a few default food items
        if not self.resources:
            self._spawn_default_resources()

        # Build spaces now that we have agent IDs
        self._build_spaces()

        # Finally, do an initial reset
        self.reset()

    def _spawn_default_resources(self, num_food=3, respawn_time=10):
        """Place `num_food` random food resources on the grid at init."""
        for _ in range(num_food):
            pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            # value=5 hunger points, respawn after 10 steps
            self.resources.append(Resource('food', pos, value=5.0, respawn_time=respawn_time))

    def _build_spaces(self):
        """
        Creates:
          - self.observation_space = Dict({agent_id: Box(...)})
          - self.action_space = Dict({agent_id: Discrete(4)})
        Each agent sees a (2r+1)x(2r+1) patch, and has 4 discrete moves.
        """
        obs_space_dict = {}
        act_space_dict = {}
        for agent in self.agents:
            r = agent.vision_range
            vision_dim = 2 * r + 1
            # Each cell ∈ {0,1,2} → we'll let Box(low=0, high=2)
            obs_space_dict[agent.agent_id] = Box(
                low=0.0, high=2.0,
                shape=(vision_dim, vision_dim),
                dtype=np.float32
            )
            act_space_dict[agent.agent_id] = Discrete(4)

        self.observation_space = Dict(obs_space_dict)
        self.action_space = Dict(act_space_dict)

    def reset(self, *, seed=None, options=None):
        """
        Reset all agents and resources. Gymnasium passes in optional `seed` and `options` here.
        We ignore `seed` and `options` unless you want to implement deterministic seeding later.

        Returns:
        obs_dict : dict of per-agent observations
        info     : an empty dict (no extra info to return)
        """
        # If you ever want to support seeding:
        if seed is not None:
            # Example: random.seed(seed)
            pass

        # Reset each agent (position + hunger)
        for agent in self.agents:
            agent.reset()

        # Reset resource timers
        for res in self.resources:
            res.timer = 0

        self.done = False

        # Build per-agent observations
        obs_dict = {agent.agent_id: agent.observe(self) for agent in self.agents}

        # Gymnasium expects (obs, info) from reset()
        return obs_dict, {}

    def step(self, action_dict):
        """
        action_dict: {agent_id: action (0/1/2/3), ...}
        Returns:
          obs_dict   : {agent_id: local_patch}
          reward_dict: {agent_id: float}
          done_flag  : bool (True if episode is over for all agents)
          info       : {}
        """
        if self.done:
            raise RuntimeError("Cannot call step() on a finished environment. Call reset() first.")

        # 1) Initialize rewards to zero
        reward_dict = {agent.agent_id: 0.0 for agent in self.agents}

        # 2) Increase hunger & apply negative hunger penalty
        for agent in self.agents:
            prev_hunger = agent.hunger
            agent.update_hunger(amount=1.0)  # +1 hunger per step
            delta_hunger = agent.hunger - prev_hunger
            # Example penalty: -0.1 reward for each hunger point gained
            reward_dict[agent.agent_id] -= 0.1 * delta_hunger

        # 3) Move each agent according to action, clipped to grid
        for agent in self.agents:
            act = action_dict.get(agent.agent_id, None)
            if act is not None:
                x, y = agent.position
                # 0=up, 1=right, 2=down, 3=left
                new_x = min(self.size - 1, max(0, x + (act == 2) - (act == 0)))
                new_y = min(self.size - 1, max(0, y + (act == 1) - (act == 3)))
                agent.position = [new_x, new_y]

        # 4) Tick resource timers (respawn countdown)
        for res in self.resources:
            res.step()

        # 5) Check consumption: if an agent lands on an available resource
        for agent in self.agents:
            for res in self.resources:
                if res.is_available() and agent.position == res.position:
                    res.consume()              # resource disappears for respawn_time
                    agent.consume(res)         # agent.hunger -= res.value
                    reward_dict[agent.agent_id] += res.value  # positive reward

        # 6) Build next observations & done flags
        obs_dict = {}
        done_dict = {}
        for agent in self.agents:
            obs_dict[agent.agent_id] = agent.observe(self)
            done_dict[agent.agent_id] = (agent.hunger >= agent.max_hunger)

        # 7) If ALL agents are done, end episode
        self.done = all(done_dict.values())

        # 8) Gym expects a single boolean "done"; info can carry per-agent dones if needed.
        info_dict = {'agent_dones': done_dict}

        return obs_dict, reward_dict, self.done, info_dict

    def render(self, mode='human'):
        """
        (Optional) Print a simple ASCII render of the grid:
         - '.' empty
         - 'A' any agent
         - 'F' available food
         - 'R' respawning food (timer > 0)
        """
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # Mark resources
        for res in self.resources:
            x, y = res.position
            if res.is_available():
                grid[x][y] = 'F'
            else:
                grid[x][y] = 'R'

        # Mark agents (overwrite food if same cell)
        for agent in self.agents:
            x, y = agent.position
            grid[x][y] = 'A'

        print("\n".join("".join(row) for row in grid))
        print("-" * self.size)

    def _get_obs_range(self, min_x, max_x, min_y, max_y):
        """
        Returns a cropped (max_x-min_x+1)×(max_y-min_y+1) array where
        - cells with another agent are 1.0
        - cells with an available resource are 2.0
        - everything else is 0.0
        """
        grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1), dtype=np.float32)

        for agent in self.agents:
            x, y = agent.position
            if min_x <= x <= max_x and min_y <= y <= max_y:
                grid[x - min_x, y - min_y] = 1.0

        for res in self.resources:
            x, y = res.position
            if (
                min_x <= x <= max_x 
                and min_y <= y <= max_y 
                and res.is_available()
            ):
                grid[x - min_x, y - min_y] = 2.0

        return grid

# class GridWorldEnv:
#     def __init__(self, size=5):
#         self.size = size
#         self.action_space = Discrete(4)
#         self.observation_space = Box(low=0, high=1, shape=(size, size), dtype=np.float32)
#         self.reset()

#     def reset(self):
#         self.agent_pos = [0, 0]
#         self.done = False
#         return self._get_obs()

#     def step(self, action):
#         if self.done:
#             raise RuntimeError("Cannot call step() on a finished environment. Call reset() first.")
#         self.agent_pos[0] = min(self.size - 1, max(0, self.agent_pos[0] + (action == 1) - (action == 3)))
#         self.agent_pos[1] = min(self.size - 1, max(0, self.agent_pos[1] + (action == 2) - (action == 0)))
#         obs = self._get_obs()
#         reward = 1.0 if self.agent_pos == [self.size - self.size//2, self.size - self.size//2] else 0.0
#         self.done = reward == 1.0
#         return obs, reward, self.done, {}

#     def _get_obs(self):
#         obs = np.zeros((self.size, self.size), dtype=np.float32)
#         obs[self.agent_pos[0], self.agent_pos[1]] = 1.0
#         return obs
