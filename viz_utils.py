# viz_utils.py

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

# viz_utils.py

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button


def launch_visualization(snapshots, obs_history, fps=2):
    """
    Replay tool: Displays a recorded trajectory using:
      - snapshots: list of full-state dicts (agent_positions, agent_orientations, & resource_states)
      - obs_history: list of per-agent local observations (dicts of arrays)

    Each index i corresponds to the state after step i (index 0 is initial state).
    """

    # Determine agent IDs, vision_range, and grid_size
    agent_ids = list(obs_history[0].keys())
    vision_dim = obs_history[0][agent_ids[0]].shape[0]
    vision_range = (vision_dim - 1) // 2
    grid_size = max(
        max(pos[0], pos[1]) for pos in snapshots[0]['agent_positions'].values()
    ) + 1

    # Set up figure and axes
    fig = plt.figure(figsize=(8, 6))
    ax_full   = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
    ax_patch1 = plt.subplot2grid((3, 3), (0, 2))
    ax_patch2 = plt.subplot2grid((3, 3), (1, 2))

    ax_button_prev    = plt.axes([0.55, 0.05, 0.08, 0.05])
    ax_button_next    = plt.axes([0.65, 0.05, 0.08, 0.05])
    ax_button_restart = plt.axes([0.75, 0.05, 0.08, 0.05])
    ax_button_quit    = plt.axes([0.85, 0.05, 0.08, 0.05])

    button_prev    = Button(ax_button_prev,    "Prev")
    button_next    = Button(ax_button_next,    "Next")
    button_restart = Button(ax_button_restart, "Restart")
    button_quit    = Button(ax_button_quit,    "Quit")

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # State index for replay
    idx = 0
    num_steps = len(snapshots) - 1  # last index after final step

    def draw_full_grid_from_snapshot(ax, snapshot):
        """
        Draw the entire grid using a snapshot dict with:
          - snapshot['agent_positions']: {agent_id: (x, y)}
          - snapshot['agent_orientations']: {agent_id: orientation (0=up,1=right,2=down,3=left)}
          - snapshot['resource_states']: list of {'position':(x,y), 'timer':t, 'type':...}

        Agents are red circles with a small arrow indicating orientation. 
        Resources: green squares if timer==0, gray if timer>0.
        """
        ax.clear()
        size = grid_size

        # Draw grid lines
        for x in range(size + 1):
            ax.plot([x, x], [0, size], color="black", linewidth=1)
        for y in range(size + 1):
            ax.plot([0, size], [y, y], color="black", linewidth=1)

        # Draw resources
        for res in snapshot['resource_states']:
            x, y = res['position']
            color = "green" if res['timer'] == 0 else "lightgray"
            square = mpatches.Rectangle(
                (y + 0.1, size - x - 1 + 0.1),
                0.8, 0.8,
                facecolor=color
            )
            ax.add_patch(square)

        # Draw agents (red circle) and then an arrow for orientation
        for agent_id, pos in snapshot['agent_positions'].items():
            x, y = pos
            # Draw main circle
            circle = mpatches.Circle(
                (y + 0.5, size - x - 0.5),
                radius=0.3,
                facecolor="red",
                edgecolor="black"
            )
            ax.add_patch(circle)

            # Draw orientation arrow:
            ori = snapshot['agent_orientations'][agent_id]
            cx, cy = (y + 0.5, size - x - 0.5)  # center
            if ori == 0:      # up
                dx, dy = (0, 0.3)
            elif ori == 1:    # right
                dx, dy = (0.3, 0)
            elif ori == 2:    # down
                dx, dy = (0, -0.3)
            else:             # ori == 3 (left)
                dx, dy = (-0.3, 0)

            # Use a simple arrow
            ax.arrow(
                cx, cy,
                dx, dy,
                head_width=0.15,
                head_length=0.15,
                fc="white",
                ec="white",
                length_includes_head=True
            )

            # Optionally label the agent initial
            ax.text(
                cx, cy,
                agent_id[0].upper(),
                color="black",
                weight="bold",
                fontsize=10,
                ha="center",
                va="center"
            )

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Full Grid (Step {idx} of {num_steps})")

    def draw_agent_patch(ax, patch, title):
        """
        Shows a small 2D patch for one agent.
        Values: 0.0 = empty, 1.0 = agent, 2.0 = resource
        We map: {0 → white, 1 → red, 2 → green}
        """
        ax.clear()
        color_map = {
            0.0: (1, 1, 1),  # white
            1.0: (1, 0, 0),  # red
            2.0: (0, 1, 0)   # green
        }
        h, w = patch.shape
        img = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                img[i, j] = color_map[patch[i, j]]

        ax.imshow(img, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def draw_current():
        # Draw full grid from snapshot[idx]
        draw_full_grid_from_snapshot(ax_full, snapshots[idx])

        # Draw the two agents' patches from obs_history[idx]
        patch0 = obs_history[idx][agent_ids[0]]
        patch1 = obs_history[idx][agent_ids[1]]
        draw_agent_patch(ax_patch1, patch0, title=f"{agent_ids[0].capitalize()}'s View")
        draw_agent_patch(ax_patch2, patch1, title=f"{agent_ids[1].capitalize()}'s View")

        plt.draw()

    # Button handlers to modify idx
    def on_prev(event):
        nonlocal idx
        if idx > 0:
            idx -= 1
            draw_current()

    def on_next(event):
        nonlocal idx
        if idx < num_steps:
            idx += 1
            draw_current()

    def on_restart(event):
        nonlocal idx
        idx = 0
        draw_current()

    def on_quit(event):
        plt.close(fig)
        sys.exit(0)

    # Wire up buttons
    button_prev.on_clicked(on_prev)
    button_next.on_clicked(on_next)
    button_restart.on_clicked(on_restart)
    button_quit.on_clicked(on_quit)

    # Initial draw
    draw_current()
    plt.show()
    
# def launch_visualization(snapshots, obs_history, fps=2):
#     """
#     Replay tool: Displays a recorded trajectory using:
#       - snapshots: list of full-state dicts (agent_positions & resource_states)
#       - obs_history: list of per-agent local observations (dicts of arrays)

#     Each index i corresponds to the state after step i (index 0 is initial state).
#     """

#     # Determine agent IDs, vision_range, and grid_size
#     agent_ids = list(obs_history[0].keys())
#     vision_dim = obs_history[0][agent_ids[0]].shape[0]
#     vision_range = (vision_dim - 1) // 2
#     grid_size = max(
#         max(pos[0], pos[1]) for pos in snapshots[0]['agent_positions'].values()
#     ) + 1

#     # Set up figure and axes
#     fig = plt.figure(figsize=(8, 6))
#     ax_full   = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
#     ax_patch1 = plt.subplot2grid((3, 3), (0, 2))
#     ax_patch2 = plt.subplot2grid((3, 3), (1, 2))

#     ax_button_prev    = plt.axes([0.55, 0.05, 0.08, 0.05])
#     ax_button_next    = plt.axes([0.65, 0.05, 0.08, 0.05])
#     ax_button_restart = plt.axes([0.75, 0.05, 0.08, 0.05])
#     ax_button_quit    = plt.axes([0.85, 0.05, 0.08, 0.05])

#     button_prev    = Button(ax_button_prev,    "Prev")
#     button_next    = Button(ax_button_next,    "Next")
#     button_restart = Button(ax_button_restart, "Restart")
#     button_quit    = Button(ax_button_quit,    "Quit")

#     plt.tight_layout(rect=[0, 0.1, 1, 1])

#     # State index for replay
#     idx = 0
#     num_steps = len(snapshots) - 1  # last index after final step

#     def draw_full_grid_from_snapshot(ax, snapshot):
#         """
#         Draw the entire grid using a snapshot dict with:
#           - snapshot['agent_positions']: {agent_id: (x, y)}
#           - snapshot['resource_states']: list of {'position':(x,y), 'timer':t, 'type':...}

#         Agents are red circles with label; resources: green squares if timer==0, gray if timer>0.
#         """
#         ax.clear()
#         size = grid_size

#         # Draw grid lines
#         for x in range(size + 1):
#             ax.plot([x, x], [0, size], color="black", linewidth=1)
#         for y in range(size + 1):
#             ax.plot([0, size], [y, y], color="black", linewidth=1)

#         # Draw resources
#         for res in snapshot['resource_states']:
#             x, y = res['position']
#             color = "green" if res['timer'] == 0 else "lightgray"
#             square = mpatches.Rectangle(
#                 (y + 0.1, size - x - 1 + 0.1),
#                 0.8, 0.8,
#                 facecolor=color
#             )
#             ax.add_patch(square)

#         # Draw agents
#         for agent_id, pos in snapshot['agent_positions'].items():
#             x, y = pos
#             circle = mpatches.Circle(
#                 (y + 0.5, size - x - 0.5),
#                 radius=0.3,
#                 facecolor="red",
#                 edgecolor="black"
#             )
#             ax.add_patch(circle)
#             ax.text(
#                 y + 0.5,
#                 size - x - 0.5,
#                 agent_id[0].upper(),
#                 color="white",
#                 weight="bold",
#                 fontsize=12,
#                 ha="center",
#                 va="center"
#             )

#         ax.set_xlim(0, size)
#         ax.set_ylim(0, size)
#         ax.set_aspect("equal")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(f"Full Grid (Step {idx} of {num_steps})")

#     def draw_agent_patch(ax, patch, title):
#         """
#         Shows a small 2D patch for one agent.
#         Values: 0.0 = empty, 1.0 = agent, 2.0 = resource
#         We map: {0 → white, 1 → red, 2 → green}
#         """
#         ax.clear()
#         color_map = {
#             0.0: (1, 1, 1),  # white
#             1.0: (1, 0, 0),  # red
#             2.0: (0, 1, 0)   # green
#         }
#         h, w = patch.shape
#         img = np.zeros((h, w, 3))
#         for i in range(h):
#             for j in range(w):
#                 img[i, j] = color_map[patch[i, j]]

#         ax.imshow(img, origin="upper")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(title)

#     def draw_current():
#         # Draw full grid from snapshot[idx]
#         draw_full_grid_from_snapshot(ax_full, snapshots[idx])

#         # Draw the two agents' patches from obs_history[idx]
#         patch0 = obs_history[idx][agent_ids[0]]
#         patch1 = obs_history[idx][agent_ids[1]]
#         draw_agent_patch(ax_patch1, patch0, title=f"{agent_ids[0].capitalize()}'s View")
#         draw_agent_patch(ax_patch2, patch1, title=f"{agent_ids[1].capitalize()}'s View")

#         plt.draw()

#     # Button handlers to modify idx
#     def on_prev(event):
#         nonlocal idx
#         if idx > 0:
#             idx -= 1
#             draw_current()

#     def on_next(event):
#         nonlocal idx
#         if idx < num_steps:
#             idx += 1
#             draw_current()

#     def on_restart(event):
#         nonlocal idx
#         idx = 0
#         draw_current()

#     def on_quit(event):
#         plt.close(fig)
#         sys.exit(0)

#     # Wire up buttons
#     button_prev.on_clicked(on_prev)
#     button_next.on_clicked(on_next)
#     button_restart.on_clicked(on_restart)
#     button_quit.on_clicked(on_quit)

#     # Initial draw
#     draw_current()
#     plt.show()


# # import sys
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as mpatches
# # from matplotlib.widgets import Button

# # def launch_visualization(snapshots, obs_history, fps=2):
# #     import sys
# #     import matplotlib.pyplot as plt
# #     import matplotlib.patches as mpatches
# #     import numpy as np
# #     from matplotlib.widgets import Button

# #     agent_ids = list(obs_history[0].keys())
# #     vision_dim = obs_history[0][agent_ids[0]].shape[0]
# #     vision_range = (vision_dim - 1) // 2
# #     grid_size = max(max(pos[0], pos[1]) for pos in snapshots[0]['agent_positions'].values()) + 1

# #     fig = plt.figure(figsize=(8, 6))
# #     ax_full = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
# #     ax_patch1 = plt.subplot2grid((3, 3), (0, 2))
# #     ax_patch2 = plt.subplot2grid((3, 3), (1, 2))

# #     ax_button_prev = plt.axes([0.55, 0.05, 0.08, 0.05])
# #     ax_button_next = plt.axes([0.65, 0.05, 0.08, 0.05])
# #     ax_button_restart = plt.axes([0.75, 0.05, 0.08, 0.05])
# #     ax_button_quit = plt.axes([0.85, 0.05, 0.08, 0.05])

# #     button_prev = Button(ax_button_prev, "Prev")
# #     button_next = Button(ax_button_next, "Next")
# #     button_restart = Button(ax_button_restart, "Restart")
# #     button_quit = Button(ax_button_quit, "Quit")

# #     plt.tight_layout(rect=[0, 0.1, 1, 1])

# #     idx = 0
# #     num_steps = len(snapshots) - 1

# #     def draw_full_grid_from_snapshot(ax, snapshot):
# #         ax.clear()
# #         size = grid_size
# #         for x in range(size + 1):
# #             ax.plot([x, x], [0, size], color="black", linewidth=1)
# #         for y in range(size + 1):
# #             ax.plot([0, size], [y, y], color="black", linewidth=1)

# #         for res in snapshot['resource_states']:
# #             x, y = res['position']
# #             color = "green" if res['timer'] == 0 else "lightgray"
# #             square = mpatches.Rectangle((y + 0.1, size - x - 1 + 0.1), 0.8, 0.8, facecolor=color)
# #             ax.add_patch(square)

# #         for agent_id, pos in snapshot['agent_positions'].items():
# #             x, y = pos
# #             circle = mpatches.Circle((y + 0.5, size - x - 0.5), radius=0.3, facecolor="red", edgecolor="black")
# #             ax.add_patch(circle)
# #             ax.text(y + 0.5, size - x - 0.5, agent_id[0].upper(), color="white", weight="bold", fontsize=12, ha="center", va="center")

# #         ax.set_xlim(0, size)
# #         ax.set_ylim(0, size)
# #         ax.set_aspect("equal")
# #         ax.set_xticks([])
# #         ax.set_yticks([])
# #         ax.set_title("Full Grid (Step {} of {})".format(idx, num_steps))

# #     def draw_agent_patch(ax, patch, title):
# #         ax.clear()
# #         color_map = {0.0: (1, 1, 1), 1.0: (1, 0, 0), 2.0: (0, 1, 0)}
# #         h, w = patch.shape
# #         img = np.zeros((h, w, 3))
# #         for i in range(h):
# #             for j in range(w):
# #                 img[i, j] = color_map[patch[i, j]]
# #         ax.imshow(img, origin="upper")
# #         ax.set_xticks(range(w))
# #         ax.set_yticks(range(h))
# #         offsets = np.arange(-vision_range, vision_range + 1)
# #         ax.set_xticklabels(offsets)
# #         ax.set_yticklabels(offsets[::-1])
# #         ax.tick_params(length=0)
# #         ax.set_title(title)

# #     def draw_current():
# #         draw_full_grid_from_snapshot(ax_full, snapshots[idx])
# #         patch0 = obs_history[idx][agent_ids[0]]
# #         patch1 = obs_history[idx][agent_ids[1]]
# #         draw_agent_patch(ax_patch1, patch0, title=f"{agent_ids[0].capitalize()}'s View")
# #         draw_agent_patch(ax_patch2, patch1, title=f"{agent_ids[1].capitalize()}'s View")
# #         plt.draw()

# #     def on_prev(event):
# #         nonlocal idx
# #         if idx > 0:
# #             idx -= 1
# #             draw_current()

# #     def on_next(event):
# #         nonlocal idx
# #         if idx < num_steps:
# #             idx += 1
# #             draw_current()

# #     def on_restart(event):
# #         nonlocal idx
# #         idx = 0
# #         draw_current()

# #     def on_quit(event):
# #         plt.close(fig)
# #         sys.exit(0)

# #     button_prev.on_clicked(on_prev)
# #     button_next.on_clicked(on_next)
# #     button_restart.on_clicked(on_restart)
# #     button_quit.on_clicked(on_quit)

# #     draw_current()
# #     plt.show()


# # # def launch_visualization(env, fps=2):
# # #     """
# # #     Opens a Matplotlib window with:
# # #       - Full‐grid view on the left
# # #       - One agent’s local view on top right
# # #       - Another agent’s local view below it
# # #       - Buttons: Play/Pause, Step, Restart, Quit

# # #     You must have exactly two agents (“alice” and “bob”) defined in `env.agents`.
# # #     If you have more or fewer agents, you can extend this function accordingly.

# # #     :param env: An already‐created GridWorldEnv (or gym.make(…)) instance.
# # #     :param fps: Frames per second when “Play” is active.
# # #     """
# # #     env = getattr(env, "unwrapped", env)
# # #     # Extract agents’ vision_range (we assume both use the same value here)
# # #     vision_range = env.agents[0].vision_range
# # #     grid_size = env.size

# # #     # --- 1) Create the figure and axes ---
# # #     fig = plt.figure(figsize=(8, 6))
# # #     ax_full  = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
# # #     ax_patch1 = plt.subplot2grid((3, 3), (0, 2))
# # #     ax_patch2 = plt.subplot2grid((3, 3), (1, 2))

# # #     ax_button_play    = plt.axes([0.70, 0.05, 0.08, 0.05])
# # #     ax_button_step    = plt.axes([0.80, 0.05, 0.08, 0.05])
# # #     ax_button_restart = plt.axes([0.90, 0.05, 0.08, 0.05])
# # #     ax_button_quit    = plt.axes([0.60, 0.05, 0.08, 0.05])

# # #     button_play    = Button(ax_button_play,    "Play")
# # #     button_step    = Button(ax_button_step,    "Step")
# # #     button_restart = Button(ax_button_restart, "Restart")
# # #     button_quit    = Button(ax_button_quit,    "Quit")

# # #     plt.tight_layout(rect=[0, 0.1, 1, 1])

# # #     # --- 2) Helper functions for drawing ---
# # #     def draw_full_grid(ax, env):
# # #         ax.clear()
# # #         size = env.size
# # #         # grid lines
# # #         for x in range(size + 1):
# # #             ax.plot([x, x], [0, size], color="black", linewidth=1)
# # #         for y in range(size + 1):
# # #             ax.plot([0, size], [y, y], color="black", linewidth=1)

# # #         # resources
# # #         for res in env.resources:
# # #             x, y = res.position
# # #             color = "green" if res.is_available() else "lightgray"
# # #             square = mpatches.Rectangle(
# # #                 (y + 0.1, size - x - 1 + 0.1), 0.8, 0.8,
# # #                 facecolor=color
# # #             )
# # #             ax.add_patch(square)

# # #         # agents
# # #         for agent in env.agents:
# # #             x, y = agent.position
# # #             circle = mpatches.Circle(
# # #                 (y + 0.5, size - x - 0.5),
# # #                 radius=0.3,
# # #                 facecolor="red",
# # #                 edgecolor="black"
# # #             )
# # #             ax.add_patch(circle)
# # #             ax.text(
# # #                 y + 0.5, size - x - 0.5,
# # #                 agent.agent_id[0].upper(),
# # #                 color="white", weight="bold", fontsize=12,
# # #                 ha="center", va="center"
# # #             )

# # #         ax.set_xlim(0, size)
# # #         ax.set_ylim(0, size)
# # #         ax.set_aspect("equal")
# # #         ax.set_xticks([])
# # #         ax.set_yticks([])
# # #         ax.set_title("Full Grid")

# # #     def draw_agent_patch(ax, patch, title):
# # #         ax.clear()
# # #         # map {0→white, 1→red, 2→green}
# # #         color_map = {
# # #             0.0: (1, 1, 1),
# # #             1.0: (1, 0, 0),
# # #             2.0: (0, 1, 0)
# # #         }
# # #         h, w = patch.shape
# # #         img = np.zeros((h, w, 3))
# # #         for i in range(h):
# # #             for j in range(w):
# # #                 img[i, j] = color_map[patch[i, j]]

# # #         ax.imshow(img, origin="upper")

# # #         # set ticks at each cell
# # #         ax.set_xticks(range(w))
# # #         ax.set_yticks(range(h))

# # #         # compute offsets so they match the number of ticks
# # #         x_offsets = np.arange(w) - (w // 2)
# # #         y_offsets = (np.arange(h) - (h // 2))[::-1]

# # #         ax.set_xticklabels(x_offsets)
# # #         ax.set_yticklabels(y_offsets)

# # #         ax.set_title(title)
# # #         ax.tick_params(length=0)

# # #     # --- 3) Playback state ---
# # #     is_running = False
# # #     timer = fig.canvas.new_timer(interval=1000 // fps)

# # #     current_obs = None
# # #     current_info = None

# # #     # --- 4) Core step & draw logic ---
# # #     def step_and_draw():
# # #         nonlocal current_obs, current_info, is_running
# # #         # for demo, use random actions:
# # #         action_dict = {
# # #             agent.agent_id: env.action_space.spaces[agent.agent_id].sample()
# # #             for agent in env.agents
# # #         }

# # #         obs, rewards, terminated, truncated, info = env.step(action_dict)
# # #         done = terminated or truncated

# # #         current_obs = obs
# # #         current_info = info
# # #         draw_current()

# # #         if done:
# # #             on_restart(None)

# # #     def draw_current():
# # #         draw_full_grid(ax_full, env)

# # #         # We assume two agents in order:  env.agents[0], env.agents[1]
# # #         agent0_id = env.agents[0].agent_id
# # #         agent1_id = env.agents[1].agent_id

# # #         patch0 = current_obs[agent0_id]
# # #         patch1 = current_obs[agent1_id]

# # #         draw_agent_patch(ax_patch1, patch0, title=f"{agent0_id.capitalize()}'s View")
# # #         draw_agent_patch(ax_patch2, patch1, title=f"{agent1_id.capitalize()}'s View")

# # #     # --- 5) Timer callback & button handlers ---
# # #     def on_timer_event(event):
# # #         if is_running:
# # #             step_and_draw()

# # #     def on_play_pause(event):
# # #         nonlocal is_running
# # #         if not is_running:
# # #             is_running = True
# # #             timer.start()
# # #             button_play.label.set_text("Pause")
# # #         else:
# # #             is_running = False
# # #             timer.stop()
# # #             button_play.label.set_text("Play")

# # #     def on_step(event):
# # #         if not is_running:
# # #             step_and_draw()

# # #     def on_restart(event):
# # #         nonlocal current_obs, current_info, is_running
# # #         is_running = False
# # #         button_play.label.set_text("Play")
# # #         timer.stop()
# # #         current_obs, current_info = env.reset()
# # #         draw_current()
# # #         plt.draw()

# # #     def on_quit(event):
# # #         plt.close(fig)
# # #         sys.exit(0)

# # #     # --- 6) Wire everything up ---
# # #     timer.add_callback(on_timer_event, None)
# # #     button_play.on_clicked(on_play_pause)
# # #     button_step.on_clicked(on_step)
# # #     button_restart.on_clicked(on_restart)
# # #     button_quit.on_clicked(on_quit)

# # #     # --- 7) Initial reset & draw, then show the window ---
# # #     current_obs, current_info = env.reset()
# # #     draw_current()
# # #     plt.show()
