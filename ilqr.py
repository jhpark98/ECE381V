import numpy as np


def forward_pass_initial(x0, u_bar, cartpole):

    N = u_bar.shape[1]   # Horizon length
    n_x = x0.shape[0]    # State dimension

    # Initialize state trajectory
    x_bar = np.zeros((n_x, N + 1)) # (n, N+1)
    x_bar[:, 0] = x0.reshape(-1,)

    # Forward pass through the horizon
    # for k in range(N):
    #     x_bar_k = x_bar[:, k].reshape(-1, 1)
    #     u_bar_k = u_bar[:, k].reshape(-1, 1)
    #     x_next = cartpole.next_step(x_bar_k, u_bar_k)
    #     x_bar[:, k + 1] = x_next.reshape(-1,) 
    
    return x_bar, u_bar

# Forward pass (simulate the system for a given control sequence)
def forward_pass(x0, x_bar, u_bar, K, d, cartpole):
    """
    Input:
        x_bar: 2D array of shape (n, N + 1)
        u_bar: 2D array of shape (m, N)
    """
    N = u_bar.shape[1]   # Horizon length
    n_x = x0.shape[0]    # State dimension
    n_u = u_bar.shape[0] # Control dimension

    # Initialize state trajectory
    x = np.zeros((n_x, N + 1))
    x[:, 0] = x0.reshape(-1,)

    # Initialize control trajectory
    u = np.zeros((n_u, N))

    # Forward pass through the horizon
    for k in range(N):
        x_k     = x[:, k].reshape(-1, 1)
        x_bar_k = x_bar[:, k].reshape(-1, 1)
        delta_u_k = K[k] @ (x_k - x_bar_k) + d[k]
        u[:, k] = u_bar[:, k] + delta_u_k
        u_k = u[:, k].reshape(-1, 1)
        x_next = cartpole.next_step(x_k, u_k) # h(x_k, u_k)
        x[:, k + 1] = x_next.reshape(-1,)
    return x, u

# Backward pass (dynamic programming step to update control policy)
def backward_pass(Q_N, Q, R, x_bar, u_bar, cartpole):
    """
    Input:
        x_bar: 2D array of shape (n, N+1)
        u_bar: 2D array of shape (m, N)
    """
    N = u_bar.shape[1]  # Horizon length

    # Initialize terminal cost-to-go
    V_k = [None] * (N + 1)
    v_k = [None] * (N + 1)
    V_k[N] = Q_N
    x_N = x_bar[:, N]
    v_k[N] = Q_N @ x_N
    
    # Storage for feedback gains and feedforward terms
    K = [None] * N  # Feedback gains
    d = [None] * N  # Feedforward gains

    for k in range(N - 1, -1, -1):

        v_next = v_k[k + 1].reshape(-1, 1)
        V_next = V_k[k + 1]

        x_bar_k = x_bar[:, k].reshape(-1, 1)
        u_bar_k = u_bar[:, k].reshape(-1, 1)

        A_k, B_k = cartpole.approx_A_B(x_bar_k, u_bar_k)

        # Compute intermediate terms
        S_x = Q @ x_bar_k + (A_k.T @ v_next)
        S_u = R @ u_bar_k + (B_k.T @ v_next)
        
        S_xx = Q + (A_k.T @ V_next @ A_k)
        S_uu = R + B_k.T @ V_next @ B_k
        S_ux = B_k.T @ V_next @ A_k

        # Compute feedback and feedforward terms
        S_uu_inv = np.linalg.inv(S_uu)
        K[k] = -S_uu_inv @ S_ux
        d[k] = -S_uu_inv @ S_u        

        # Update V_k and v_k for the next step
        v_k[k] = S_x + (K[k].T @ S_u) + (S_ux.T @ d[k]) + (K[k].T @ S_uu @ d[k])
        V_k[k] = S_xx + (2* K[k].T @ S_ux) + (K[k].T @ S_uu @ K[k])
    return K, d

# iLQR algorithm
def iLQR(T, T_s, x0, u_bar, cartpole, threshold=1):
    """
    Inputs:
        x0: 2D array of shape (n, 1)
    """
    
    Q_N = np.eye(4) * 100
    Q = np.eye(4)
    R = np.eye(1) * 0.1

    N = int(T / T_s)  # Number of time steps

    # Step 0: Initial control guess (zeros)
    # u_bar = 10 * np.random.normal(-1.0, 0.1, (R.shape[1], N))

    prev_cost = np.inf
    iteration = 0  # Track the number of iterations

    # Step 1: Initial reference trajectory
    # Forward pass: compute state trajectory for current control sequence
    x_bar, _ = forward_pass_initial(x0, u_bar, cartpole)

    """
        Initial setup:
            x_bar and u_bar are all zeors, except x0 is an initial condition.
    """

    # Start iLQR    
    while True:
        # Backward pass: compute optimal control updates (feedback K and feedforward d gains)
        K, d = backward_pass(Q_N, Q, R, x_bar, u_bar, cartpole)

        # Forward pass: update control sequence using feedback and feedforward gains
        x_bar, u_bar = forward_pass(x0, x_bar, u_bar, K, d, cartpole)
        
        # Compute the current cost
        # current_cost = compute_cost(x_bar, u_bar, Q, R, Q_N)
        current_cost = (0.5 * x_bar[:, 0].T @ Q @ x_bar[:, 0]) + (0.5 * u_bar[:, 0].T @ R @ u_bar[:, 0])

        # Print iteration number and current cost
        # print(f"Iteration {iteration + 1}, Cost: {current_cost}")

        # Check the difference between the current and previous cost
        cost_diff = abs(current_cost - prev_cost)
        if (current_cost < 6000) and (cost_diff < threshold):
            print(f"Converged after {iteration + 1} iterations with cost difference {cost_diff}")
            break

        prev_cost = current_cost
        iteration += 1
    
    J_list = np.zeros(N+1)
    for k in range(N):
        J_list[k] = (0.5 * x_bar[:, k].T @ Q @ x_bar[:, k]) + (0.5 * u_bar[:, k].T @ R @ u_bar[:, k])
    J_list[N] = (0.5 * x_bar[:, -1].T @ Q_N @ x_bar[:, -1])

    return x_bar, u_bar, J_list

def compute_cost(x, u, Q, R, Q_N):
    N = u.shape[1]
    cost = 0
    for k in range(N):
        cost += (0.5 * x[:, k].T @ Q @ x[:, k]) + (0.5 * u[:, k].T @ R @ u[:, k])
    cost += (0.5 * x[:, -1].T @ Q_N @ x[:, -1])
    return cost