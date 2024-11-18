import numpy as np
import matplotlib.pyplot as plt

class ValueIteration():
    def __init__(self, env, gamma):

        self.env = env

        self.discount_factor = gamma

        self.value_table = np.zeros((self.env.WIDTH, self.env.HEIGHT)) # for value iteration
        self.policy_table = np.full((self.env.WIDTH, self.env.HEIGHT, 4), 0.25) # to keep track of intention
        self.iter_value = 0

        self.prev_value_table = None

    ''' Helper Functions '''
    def get_value(self, state):
        return self.value_table[state[0]][state[1]]

    def check_value_convergence(self, threshold=1e-1):
        ''' True if converged '''
        delta = np.linalg.norm(self.value_table - self.prev_value_table)
        return (delta < threshold)
    
    def get_results(self, save=False):
        # self.grid = self.value_table[::-1]
        self.grid = self.value_table
        print('\n'.join(' '.join(f"{x:.2f}" for x in row) for row in self.grid)) # grid view
        optimal_value_PI = self.grid
        optimal_policy_PI = self.update_policy_table()
        return optimal_value_PI, optimal_policy_PI

    def update_value_table(self):
        ''' update value table using value update (or Bellman update/back-up) '''

        next_value_table = np.zeros((self.env.WIDTH, self.env.HEIGHT))

        # compute Bellman optimality equation for all states
        for state in self.env.get_all_states():

            # value for terminal state = 0
            if tuple(state) == tuple(self.env.goal):
                next_value_table[state[0]][state[1]] = 0.0
                continue # terminal state reached. no action needed.
            
            # Bellman optimality equation
            value_lst = []
            for action in self.env.possible_actions: # (0,1,2,3)
                next_state_lst = [self.env.state_after_action(state, action) for action in self.env.possible_actions]
                reward = np.array([self.env.get_reward(next_state) for next_state in next_state_lst]) # (4,)
                next_value = np.array([self.get_value(next_state) for next_state in next_state_lst]) # (4,)
                value_lst.append(np.sum(self.env.TP[action] * (reward + self.discount_factor * next_value)))
            next_value_table[state[0]][state[1]] = max(value_lst)
        
        self.prev_value_table = self.value_table
        self.value_table = next_value_table # update value table
        return next_value_table
    
    def update_policy_table(self):
        ''' using current value function, compute optimal policy '''
        next_policy_table = np.zeros((self.env.WIDTH, self.env.HEIGHT, 4))
        for state in self.env.get_all_states():
            
            if tuple(state) == tuple(self.env.goal):

                continue # no policy at the terminal state
            
            value_lst = []
            for action in self.env.possible_actions: # (0,1,2,3)
                next_state_lst = [self.env.state_after_action(state, action) for action in self.env.possible_actions]
                reward = np.array([self.env.get_reward(next_state) for next_state in next_state_lst]) # (4,)
                next_value = np.array([self.get_value(next_state) for next_state in next_state_lst]) # (4,)
                value_lst.append(np.sum(self.env.TP[action] * (reward + self.discount_factor * next_value)))
            
            next_policy_table[state[0]][state[1]] = np.eye(1, 4, np.argmax(np.array(value_lst)))[0] # update policy for each state
            self.policy_table = next_policy_table

        return self.policy_table

    ''' Run and Simulate '''
    def run(self, epsilon):
        ''' run the value iteration algorithm until convergence is made '''
        converged = False
        while not converged:
            self.update_value_table()
            self.update_policy_table()
            self.iter_value += 1
            converged = self.check_value_convergence(epsilon)

        print("----- ----- Summary ----- -----")
        print(f"{self.iter_value} sweeps over the state space requried for convergence.")
        # print(f"Value table converged at {self.iter_value}")

    def simulate(self):
        ''' at each state, follow the optimal policy until the goal state is reached '''
        
        cur_state = self.start
        goal = self.env.goal

        trajectory = []
        trajectory.append(cur_state)
        while cur_state != goal:
            action = self.optimal_policy[cur_state[0]][cur_state[1]]
            next_state = self.next_state(cur_state, action) # stochastic
            cur_state = next_state # update
            trajectory(cur_state)
    
    ''' Visualization '''
    def gen_heatmap_value(self, plot=False, save=False, name="heatmap_value.png"):

        self.grid_value = self.value_table[::-1, :]
        self.grid_policy = self.policy_table[::-1, :, :]

        color_map = 'viridis'
        plt.imshow(self.grid_value, cmap=color_map, interpolation='nearest')

        # Show values in each cell with adjusted font size and color for clarity
        for (i, j), val in np.ndenumerate(self.grid_value):
            plt.text(j, i, f"{val:.2f}", ha='center', va='center',
                    color='black', fontsize=5, fontweight='bold')

        # Add grid lines between cells
        plt.gca().set_xticks(np.arange(-0.5, self.grid_value.shape[1], 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.grid_value.shape[0], 1), minor=True)
        plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
        plt.tick_params(which="minor", size=0)
        
        nrows, ncols = self.grid_policy.shape[:2]
        
        # Center x and y ticks between cells and remove major ticks from minor grid
        plt.xticks(np.arange(ncols), np.arange(ncols), fontsize=8)
        plt.yticks(np.arange(nrows), np.arange(nrows)[::-1], fontsize=8)

        # Remove margins around the heatmap
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Add color bar with adjustments for spacing and label size
        color_bar = plt.colorbar(label="Intensity", pad=0.02, aspect=30)
        color_bar.ax.tick_params(labelsize=8)

        # Add title and axis labels with adjusted font size
        plt.title("Value Iteration", fontsize=10, fontweight='bold')
        plt.xlabel("X-axis", fontsize=9)
        plt.ylabel("Y-axis", fontsize=9)
        # Remove inversion of y-axis to match coordinate system
        # plt.gca().invert_yaxis()

        # Overlay arrows to show optimal actions
        n = self.value_table.shape[0]
        
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
                if len(self.grid_policy[i, j]) == 0:
                    continue  # terminal state
                
                # Find all action indices with the maximum value
                max_value = np.max(self.grid_policy[i, j])
                max_action_indices = np.where(self.grid_policy[i, j] == max_value)[0]
                
                for action_idx in max_action_indices:
                    dx, dy = action_vectors[action_idx]
                    offset_x, offset_y = offset_vectors[action_idx]
                    
                    # Update X and Y to plot multiple arrows
                    X_arrow[i, j] = X[i, j] + offset_x
                    Y_arrow[i, j] = Y[i, j] + offset_y
                    U[i, j] = dx * arrow_length
                    V[i, j] = dy * arrow_length


        plt.quiver(X_arrow, Y_arrow, U, V, angles='xy', scale_units='xy', scale=1,
                color='black', width=0.003)

        # Plot or save the figure based on parameters
        if plot:
            plt.show()
        
        if save:
            print("saving figure ...")
            plt.savefig(name, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
    
def gen_heatmap_policy(self, plot=False, save=False, name="heatmap_policy.png"):
    '''Generates a heatmap of the policy, color-coding actions and displaying action values.'''
    # Reverse the grid to match coordinate system
    self.grid_value = self.value_table[::-1, :]
    self.grid_policy = self.policy_table[::-1, :, :]

    nrows, ncols = self.grid_policy.shape[:2]

    # Create a white background image
    bg_img = np.ones((nrows, ncols, 3))  # RGB

    plt.figure()
    plt.imshow(bg_img, interpolation='nearest')

    # Define consistent colors with transparency for both grid and legend
    action_colors = {
        0: (1, 0, 0, 0.2),     # up: red with 20% transparency
        1: (0, 1, 0, 0.2),     # down: green with 20% transparency
        2: (0, 0, 1, 0.2),     # left: blue with 20% transparency
        3: (0.5, 0, 0.5, 0.2)  # right: purple with 20% transparency
    }

    for action in range(4):
        color = action_colors[action][:3]  # Remove transparency for color fill in grid
        transparency = action_colors[action][3]  # Transparency level

        # Create a mask where the action is optimal
        mask = np.zeros((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                policy = self.grid_policy[i, j, :]
                if len(policy) == 0:
                    continue  # terminal state or invalid cell
                max_value = np.max(policy)
                optimal_actions = np.where(policy == max_value)[0]
                if action in optimal_actions:
                    mask[i, j] = 1  # Mark cell where action is optimal

            # Create a color image using the mask
            color_img = np.zeros((nrows, ncols, 3))
            for k in range(3):
                color_img[:, :, k] = color[k] * mask

            # Overlay the color image with specified transparency
            plt.imshow(color_img, interpolation='nearest', alpha=transparency)

    # Add text for action values in each cell
    for i in range(nrows):
        for j in range(ncols):
            policy = self.grid_policy[i, j, :]
            if len(policy) == 0:
                continue  # terminal state or invalid cell
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
    plt.title("Policy Heatmap", fontsize=10, fontweight='bold')
    plt.xlabel("X-axis", fontsize=9)
    plt.ylabel("Y-axis", fontsize=9)

    # Add legend with matching colors and transparency
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1, 0, 0, 0.2), edgecolor='r', label='Up (0)'),
        Patch(facecolor=(0, 1, 0, 0.2), edgecolor='g', label='Down (1)'),
        Patch(facecolor=(0, 0, 1, 0.2), edgecolor='b', label='Left (2)'),
        Patch(facecolor=(0.5, 0, 0.5, 0.2), edgecolor=(0.5, 0, 0.5), label='Right (3)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Plot or save the figure based on parameters
    if plot:
        plt.show()
    if save:
        print(f"saving {name} ...")
        plt.savefig(name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
