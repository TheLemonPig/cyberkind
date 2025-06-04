import numpy as np

class Agent:
    """
    Represents one agent in the grid. Tracks position, hunger, and vision.
    """
    def __init__(self, agent_id, start_pos, vision_range=2, max_hunger=10.0):
        """
        :param agent_id: Unique identifier (string or int).
        :param start_pos: [x, y] initial location (integers in [0, size-1]).
        :param vision_range: radius r; agent sees a (2r+1)x(2r+1) patch.
        :param max_hunger: threshold at which agent is considered "done" (starved).
        """
        self.agent_id = agent_id
        self.start_pos = list(start_pos)
        self.position = list(start_pos)
        self.vision_range = vision_range
        self.hunger = 0.0
        self.max_hunger = max_hunger

    def reset(self):
        """Bring agent back to its starting position and reset hunger."""
        self.position = list(self.start_pos)
        self.hunger = 0.0

    def observe(self, env):
        """
        Returns a (2r+1)x(2r+1) array representing the local patch around this agent.
        - 0.0 = empty cell
        - 1.0 = another agent in that cell
        - 2.0 = available resource in that cell
        """
        x, y = self.position
        r = self.vision_range

        min_x, max_x = max(0, x - r), min(env.size - 1, x + r)
        min_y, max_y = max(0, y - r), min(env.size - 1, y + r)

        return env._get_obs_range(min_x, max_x, min_y, max_y)

    def update_hunger(self, amount=1.0):
        """
        Increase hunger by `amount`, clamped at max_hunger.
        If hunger >= max_hunger, this agent is effectively 'dead' / done.
        """
        self.hunger = min(self.max_hunger, self.hunger + amount)

    def consume(self, resource):
        """
        Called when agent steps onto a resource cell. Reduces hunger by resource.value.
        """
        if resource.type == 'food':
            self.hunger = max(0.0, self.hunger - resource.value)
