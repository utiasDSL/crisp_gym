import json
import numpy as np

file_name = "acts_20251015-185324.json"
with open(f"trajectories/{file_name}", "r") as f:
    actions_fw = np.array(json.load(f))

# go down
i = 0
reduced_actions = []
cumulative_action = np.zeros(3)
while actions_fw[i][6] == 0.0:
    cumulative_action += actions_fw[i][:3]
    i += 1
reduced_actions.append(
    np.concatenate((cumulative_action, [0.0, 0.0, 0.0, 1.0]), axis=None)
)
# close gripper
while actions_fw[i][6] != 0.0:
    reduced_actions.append(actions_fw[i])
    i += 1

# go up
cumulative_action = np.zeros(3)
while np.all(actions_fw[i][:2] == 0.0):
    cumulative_action += actions_fw[i][:3]
    i += 1
reduced_actions.append(
    np.concatenate((cumulative_action, [0.0, 0.0, 0.0, 0.0]), axis=None)
)
print(f"Up movement: {cumulative_action[2]}")
delta_z = cumulative_action[2]

# move to target x,y,z staying at least 1.4cm above the gripping height
cumulative_action = np.zeros(3)
while True:
    if delta_z + cumulative_action[2] + actions_fw[i][2] < 0.012:
        break
    cumulative_action += actions_fw[i][:3]
    i += 1
reduced_actions.append(
    np.concatenate((cumulative_action, [0.0, 0.0, 0.0, 0.0]), axis=None)
)

# keep remaining actions
while i < len(actions_fw):
    reduced_actions.append(actions_fw[i])
    i += 1


# Visualize actions and reduced actions with different gripper states in different colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib import colors as mcolors
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import pyplot as plt


# Visualize actions and reduced actions with different gripper states in different colors
def plot_3d_trajectory(actions, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    points = np.cumsum(actions[:, :3], axis=0)
    points = np.vstack([np.zeros((1, 3)), points])  # Start at origin
    segments = np.concatenate(
        [points[:-1, np.newaxis, :], points[1:, np.newaxis, :]], axis=1
    )

    # Create a colormap based on gripper state
    gripper_states = actions[:, 6]
    norm = mcolors.BoundaryNorm(boundaries=[-0.1, 0.5, 1.1], ncolors=2)
    cmap = cm.get_cmap("coolwarm", 2)

    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(gripper_states[:-1])
    lc.set_linewidth(2)
    ax.add_collection(lc)

    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(points[:, 2].min(), points[:, 2].max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Create a custom legend
    legend_elements = [
        Patch(facecolor=cmap(0), edgecolor="k", label="Gripper Closed"),
        Patch(facecolor=cmap(1), edgecolor="k", label="Gripper Open"),
    ]
    ax.legend(handles=legend_elements)

    plt.show()


safe_reduced_actions = []
last_gripper_state = 1.0
for a in reduced_actions:
    if a[6] != last_gripper_state and not np.all(a[:6] == 0.0):
        # Insert a  pure gripper action
        safe_reduced_actions.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a[6]]))
        last_gripper_state = a[6]
    safe_reduced_actions.append(a)
reduced_actions = safe_reduced_actions
# for a in reduced_actions:
#     print(a)

plot_3d_trajectory(np.array(actions_fw), "Original Actions")
plot_3d_trajectory(np.array(reduced_actions), "Reduced Actions")
with open(f"trajectories/v2_crafted_rel_{file_name}", "w") as f:
    json.dump(list(map(list, reduced_actions)), f)
