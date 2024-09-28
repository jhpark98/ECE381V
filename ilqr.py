import numpy as np

def forward_pass_initial(x0, u_bar, cartpole):

    N = u_bar.shape[1]   # Horizon length
    n_x = x0.shape[0]    # State dimension
    n_u = u_bar.shape[0] # Control dimension

    # Initialize state trajectory
    x = np.zeros((n_x, N + 1)) # (n, N+1)
    x[:, 0] = x0.reshape(-1,)

    # Initialize control trajectory
    # u = np.zeros((n_u, N))
    u = np.random.normal(0.0, 0.1, (n_u, N))

    # Forward pass through the horizon
    for k in range(N):
        x_k = x[:, k].reshape(-1, 1)
        u_k = u[:, k].reshape(-1, 1)
        x_next = cartpole.next_step(x_k, u_k)
        x[:, k + 1] = x_next.reshape(-1,)
    return x, u

# Forward pass (simulate the system for a given control sequence)
def forward_pass(x0, x_bar, u_bar, K, d, cartpole):
    """
    Input:
        x_bar: 2D array of shape (n, N+1)
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
        x_next = cartpole.next_step(x_k, u_k)
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
def iLQR(T, T_s, x0, cartpole, threshold=1e-4):
    """
    Inputs:
        x0: 2D array of shape (n, 1)
    """
    
    Q_N = np.eye(4) * 100
    Q = np.eye(4)
    R = np.eye(1) * 0.1

    N = int(T / T_s)  # Number of time steps

    # Step 0: Initial control guess (zeros)
    u_bar = np.zeros((R.shape[1], N))
    prev_cost = np.inf
    iteration = 0  # Track the number of iterations

    # Step 1: Initial reference trajectory
    # Forward pass: compute state trajectory for current control sequence
    x_bar, _ = forward_pass_initial(x0, u_bar, cartpole)

    while True:

        # Step 2: Backward pass to update control
        # Backward pass: compute optimal control updates (feedback K and feedforward d gains)
        K, d = backward_pass(Q_N, Q, R, x_bar, u_bar, cartpole)

        # Step 3: Update control sequence using feedback and feedforward gains
        # We perform a new forward pass to compute the next state and control trajectory
        x_bar, u_bar = forward_pass(x0, x_bar, u_bar, K, d, cartpole)

        # Compute the current cost
        current_cost = compute_cost(x_bar, u_bar, Q, R, Q_N)

        # Print iteration number and current cost
        print(f"Iteration {iteration + 1}, Cost: {current_cost}")

        # Check the difference between the current and previous cost
        cost_diff = abs(current_cost - prev_cost)
        if cost_diff < threshold:
            print(f"Converged after {iteration + 1} iterations with cost difference {cost_diff}")
            break  # Exit the loop if cost change is below the threshold

        prev_cost = current_cost  # Update the previous cost for the next iteration
        iteration += 1  # Increment the iteration counter
    
    # J_list = [None] * N        # List to store costs (0 to N)
    # for k in range(N):
    #     x_k = x_list[k]
    #     u_k = F[k] @ x_k
    #     u_list[k] = u_k
    #     # xu = (x_k, u_k)
    #     x_next = cartpole.next_step(x_k, u_k) # x_k+1
    #     x_list[k+1] = x_next

    #     # Calculate cost at each step (penalizing deviation from the desired state)
    #     J_k = 0.5 * (x_k.T @ P[k] @ x_k)
    #     J_list[k] = J_k

    # # Terminal cost (penalizing deviation from the terminal state)
    # x_N = x_list[-1].reshape(-1, 1)
    # J_N = 0.5 * x_N.T @ Q_N @ x_N
    # J_list.append(J_N)

    return x_bar, u_bar, None

# Dummy compute_cost function for illustration (implement according to your specific problem)
def compute_cost(x, u, Q, R, Q_f):
    N = u.shape[1]
    cost = 0
    for k in range(N):
        cost += 0.5 * x[:, k].T @ Q @ x[:, k] + 0.5 * u[:, k].T @ R @ u[:, k]
    cost += x[:, -1].T @ Q_f @ x[:, -1]
    return cost