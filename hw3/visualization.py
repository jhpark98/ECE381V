import numpy as np
import matplotlib.pyplot as plt

''' Visualization '''
def gen_heatmap_value(value_table, policy_table, trajectory, plot=False, save=False, name="policy_iteration_value.png", title="Policy Iteration"):

    grid_value = value_table[::-1, :]
    grid_policy = policy_table[::-1, :, :]

    # color_map = 'plasma'
    color_map = 'viridis'
    # color_map = 'cividis'
    # color_map = 'magma'
    plt.imshow(grid_value, cmap=color_map, interpolation='nearest', alpha=0.5)

    # Show values in each cell with adjusted font size and color for clarity
    for (i, j), val in np.ndenumerate(grid_value):
        plt.text(j, i, f"{val:.2f}", ha='center', va='center',
                color='black', fontsize=5, fontweight='bold')

    # Add grid lines between cells
    plt.gca().set_xticks(np.arange(-0.5, grid_value.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, grid_value.shape[0], 1), minor=True)
    plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
    plt.tick_params(which="minor", size=0)
    
    nrows, ncols = grid_policy.shape[:2]
    
    # Center x and y ticks between cells and remove major ticks from minor grid
    plt.xticks(np.arange(ncols), np.arange(ncols), fontsize=8)
    plt.yticks(np.arange(nrows), np.arange(nrows)[::-1], fontsize=8)

    # Remove margins around the heatmap
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Add color bar with adjustments for spacing and label size
    color_bar = plt.colorbar(label="Intensity", pad=0.02, aspect=30)
    color_bar.ax.tick_params(labelsize=8)

    # Add title and axis labels with adjusted font size
    plt.title(title, fontsize=10, fontweight='bold')
    plt.xlabel("X-axis", fontsize=9)
    plt.ylabel("Y-axis", fontsize=9)
    
    # Draw trajectory on the heatmap
    for i in range(len(trajectory) - 1):
        start = trajectory[i]
        end = trajectory[i + 1]

        # Reverse the y-coordinate for grid visualization
        start_y = grid_value.shape[0] - 1 - start[0]
        end_y = grid_value.shape[0] - 1 - end[0]

        # Draw a red line between the consecutive trajectory points
        plt.plot([start[1], end[1]], [start_y, end_y], color='red', linewidth=2, label="Trajectory" if i == 0 else "")

    # Add a legend for the trajectory
    plt.legend(loc='upper left', fontsize=8)

    ''' Add arrows showing policy

    # Overlay arrows to show optimal actions
    n = value_table.shape[0]
    
    # Correct action vectors matching your action indices
    action_vectors = {
        0: (0, -1),   # up
        1: (0, 1),    # down
        2: (-1, 0),   # left
        3: (1, 0)     # right
    }
    offset_vectors = {
        0: (0, -0.25),  # up
        1: (0, 0.25),   # down
        2: (-0.25, 0),  # left
        3: (0.25, 0)    # right
    }

    U = np.zeros((n, n))
    V = np.zeros((n, n))
    X_arrow = np.zeros((n, n))
    Y_arrow = np.zeros((n, n))

    arrow_length = 0.2

    X, Y = np.meshgrid(np.arange(n), np.arange(n))

    for i in range(n):
        for j in range(n):
            if np.sum(grid_policy[i, j]) == 0:
                continue  # terminal state
            
            # Find all action indices with the maximum value
            max_value = np.max(grid_policy[i, j])
            max_action_indices = np.where(grid_policy[i, j] == max_value)[0]
            
            for action_idx in max_action_indices:
                dx, dy = action_vectors[action_idx]
                offset_x, offset_y = offset_vectors[action_idx]
                
                # Update X and Y to plot multiple arrows
                X_arrow[i, j] = X[i, j] + offset_x
                Y_arrow[i, j] = Y[i, j] + offset_y
                U[i, j] = dx * arrow_length
                V[i, j] = dy * arrow_length


    plt.quiver(X_arrow, Y_arrow, U, V, angles='xy', scale_units='xy', scale=1, color='red', width=0.003)
    '''

    # Plot or save the figure based on parameters
    if plot:
        plt.show()
    
    if save:
        print(f"saving {name} ...")
        plt.savefig(name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()


def gen_heatmap_policy(policy_table, plot=False, save=False, name="policy_iteration_policy.png", title="Policy Iteration"):
    ''' Generates a heatmap of the policy, color-coding actions and displaying action values. '''
    # Reverse the grid to match coordinate system
    grid_policy = policy_table[::-1, :, :]
    nrows, ncols = grid_policy.shape[:2]

    # Create a white background image
    bg_img = np.ones((nrows, ncols, 3))  # RGB

    plt.figure()
    plt.imshow(bg_img, interpolation='nearest')

    # Define consistent colors for actions
    action_colors = {
        0: (1, 0, 0),     # up: red
        1: (0, 1, 0),     # down: green
        2: (0, 0, 1),     # left: blue
        3: (0.5, 0, 0.5)  # right: purple
    }

    # Combine masks into one image
    final_img = np.zeros((nrows, ncols, 3))
    for action in range(4):
        color = action_colors[action]

        # Create a mask where the action is optimal
        mask = np.zeros((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                policy = grid_policy[i, j, :]
                if len(policy) == 0:
                    continue  # terminal state or invalid cell
                max_value = np.max(policy)
                optimal_actions = np.where(policy == max_value)[0]
                if action in optimal_actions:
                    mask[i, j] = 1  # Mark cell where action is optimal

        # Apply color to the final image based on the mask
        for k in range(3):  # RGB channels
            final_img[:, :, k] += color[k] * mask

    # Clip the values to ensure valid RGB values
    final_img = np.clip(final_img, 0, 1)

    # Display the final image
    plt.imshow(final_img, interpolation='nearest')

    # Add text for action values in each cell
    for i in range(nrows):
        for j in range(ncols):
            policy = grid_policy[i, j, :]
            if np.sum(policy) == 0:  # terminal state
                continue
            max_value = np.max(policy)
            optimal_actions = np.where(policy == max_value)[0]
            # Display the action number (e.g., 0 for up, 1 for down, etc.)
            action_text = '/'.join(map(str, optimal_actions))  # Handle multiple optimal actions
            plt.text(j, i, action_text, ha='center', va='center', color='black', fontsize=8, fontweight='bold')

    # Add grid lines between cells
    plt.gca().set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
    plt.tick_params(which="minor", size=0)

    # Center x and y ticks between cells
    plt.xticks(np.arange(ncols), np.arange(ncols), fontsize=8)
    plt.yticks(np.arange(nrows), np.arange(nrows)[::-1], fontsize=8)

    # Remove margins around the heatmap
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Add title and axis labels with adjusted font size
    plt.title(title, fontsize=10, fontweight='bold')
    plt.xlabel("X-axis", fontsize=9)
    plt.ylabel("Y-axis", fontsize=9)

    # Add legend with matching colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1, 0, 0), edgecolor='r', label='Up (0)'),
        Patch(facecolor=(0, 1, 0), edgecolor='g', label='Down (1)'),
        Patch(facecolor=(0, 0, 1), edgecolor='b', label='Left (2)'),
        Patch(facecolor=(0.5, 0, 0.5), edgecolor=(0.5, 0, 0.5), label='Right (3)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Plot or save the figure based on parameters
    if plot:
        plt.show()
    if save:
        print(f"saving {name} ...")
        plt.savefig(name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
