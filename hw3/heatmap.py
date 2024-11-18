import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_grid_heatmap(values, sgoal, figsize=(12, 10)):
    """
    Plots a heatmap of the grid values with grid coordinates.

    Parameters:
    - values (np.ndarray): 2D array of shape (n, n) containing the values to plot.
    - sgoal (tuple): The goal state coordinates (x_goal, y_goal) to highlight on the heatmap.
    - figsize (tuple): Size of the matplotlib figure.
    """
    n = values.shape[0]
    
    # Create coordinate labels
    x_labels = [f'({x},0)' for x in range(n)]
    y_labels = [f'({0},{y})' for y in range(n)]
    
    # Initialize the matplotlib figure
    plt.figure(figsize=figsize)
    
    # Create the heatmap
    ax = sns.heatmap(
        values,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=.5,
        linecolor='gray'
    )
    
    # Set labels and title
    plt.title('Optimal Value Function Heatmap')
    plt.xlabel('x1 Coordinate')
    plt.ylabel('x2 Coordinate')
    
    # Invert y-axis to have (0,0) at bottom-left
    plt.gca().invert_yaxis()
    
    # Highlight the goal state
    goal_x, goal_y = sgoal
    ax.add_patch(plt.Rectangle((goal_x, n - 1 - goal_y), 1, 1, fill=False, edgecolor='red', lw=3))
    plt.text(goal_x + 0.5, n - 1 - goal_y + 0.5, 'Goal', color='red', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    n = 20
    # Example: Initialize a random value function for demonstration
    np.random.seed(0)  # For reproducibility
    values = np.random.rand(n, n)
    
    # Define the goal state
    sgoal = (17, 6)
    
    # Set the goal state's value to a high number for visibility
    values[sgoal[1], sgoal[0]] = 10.0  # Note: rows correspond to y-axis
    
    # Plot the heatmap
    plot_grid_heatmap(values, sgoal)
