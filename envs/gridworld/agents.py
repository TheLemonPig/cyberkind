import numpy as np

class Agent:
    """
    Represents one agent in the grid. Tracks position, hunger, and vision.
    """
    def __init__(self, agent_id, start_pos, vision_range=2, max_hunger=10.0, orientation=0):
        """
        :param agent_id: Unique identifier (string or int).
        :param start_pos: [x, y] initial location (integers in [0, size-1]).
        :param vision_range: radius r; agent sees a (2r+1)x(2r+1) patch.
        :param max_hunger: threshold at which agent is considered "done" (starved).
        :param orientation: direction agent is facing (0=up, 1=right, 2=down, 3=left).
        """
        self.agent_id = agent_id
        self.start_pos = list(start_pos)
        self.position = list(start_pos)
        self.vision_range = vision_range
        self.hunger = 0.0
        self.max_hunger = max_hunger
        self.orientation = orientation

    def reset(self):
        """Bring agent back to its starting position and reset hunger."""
        self.position = list(self.start_pos)
        self.hunger = 0.0

    def observe(self, env):
        """
        Returns a (2r+1)x(2r+1) array representing the local patch around this agent,
        masked to the agent's vision cone based on orientation.
        - 0.0 = empty cell or outside vision cone
        - 1.0 = another agent in that cell within vision cone
        - 2.0 = available resource in that cell within vision cone
        """
        x, y = self.position
        r = self.vision_range

        min_x, max_x = max(0, x - r), min(env.size - 1, x + r)
        min_y, max_y = max(0, y - r), min(env.size - 1, y + r)

        patch = env._get_obs_range(min_x, max_x, min_y, max_y)

        # Create mask for vision cone
        mask = np.zeros_like(patch, dtype=bool)
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                dx = i - r
                dy = j - r
                if self.orientation == 0:  # up
                    if dx < 0 and abs(dy) <= -dx:
                        mask[i, j] = True
                elif self.orientation == 1:  # right
                    if dy > 0 and abs(dx) <= dy:
                        mask[i, j] = True
                elif self.orientation == 2:  # down
                    if dx > 0 and abs(dy) <= dx:
                        mask[i, j] = True
                elif self.orientation == 3:  # left
                    if dy < 0 and abs(dx) <= -dy:
                        mask[i, j] = True

        patch[~mask] = 0.0
        return patch

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
