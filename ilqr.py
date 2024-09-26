import numpy as np

def forward_pass_initial(x0, u_bar, T_s):

    N = u_bar.shape[1]   # Horizon length
    n_x = x0.shape[0]    # State dimension
    n_u = u_bar.shape[0] # Control dimension

    # Initialize state trajectory
    x = np.zeros((n_x, N + 1))
    x[:, 0] = x0

    # Initialize control trajectory
    u = np.zeros((n_u, N))

    # Forward pass through the horizon
    for k in range(N):
        x[:, k + 1] = h_lin(x[:, k], u[:, k])

    return x, u

# Forward pass (simulate the system for a given control sequence)
def forward_pass(x0, xu_bar, K, d):

    x_bar, u_bar = xu_bar # reference trajectory

    N = u_bar.shape[1]   # Horizon length
    n_x = x0.shape[0]    # State dimension
    n_u = u_bar.shape[0] # Control dimension

    # Initialize state trajectory
    x = np.zeros((n_x, N + 1))
    x[:, 0] = x0

    # Initialize control trajectory
    u = np.zeros((n_u, N))

    # Forward pass through the horizon
    for k in range(N):
        delta_u_k = K[k] @ (x[:, k] - x_bar[:, k]) + d[k]
        u[:, k] = u_bar[:, k] + delta_u_k
        x[:, k + 1] = h(x[:, k], u[:, k])

    return x, u

# Backward pass (dynamic programming step to update control policy)
def backward_pass(Q, R, Q_f, xu_bar):

    x_bar, u_bar = xu_bar # reference trajectory

    N = x_bar.shape[1]  # Horizon length
    n_x = x_bar.shape[0]  # State dimension
    n_u = u_bar.shape[0]  # Control dimension

    # Initialize terminal cost-to-go
    V_k = [None] * (N + 1)
    V_k = [None] * (N + 1)
    V_k[-1] = Q_f
    v_k[-1] = Q_f @ x_bar[:, -1]

    # Storage for feedback gains and feedforward terms
    K = np.zeros((N - 1, n_u, n_x))  # Feedback gains
    d = np.zeros((N - 1, n_u))       # Feedforward gains

    for k in range(N - 2, -1, -1):
        V_next = V_k[-1]
        v_next = v_k[-1]

        A_k = grad_x_h(grad_x_f, values, T_s)
        B_k = grad_u_h(grad_u_f, values, T_s)

        # Compute intermediate terms
        S_x = Q @ x_bar[:, k] + (A_k.T @ v_next)
        S_u = R @ u_bar[:, k] + (B_k.T @ v_next)
        S_xx = Q + (A_k.T @ V_next @ A_k)
        S_uu = R + B_k.T @ V_next @ B_k
        S_ux = B_k.T @ V_next @ A_k

        # Compute feedback and feedforward terms
        S_uu_inv = np.linalg.inv(S_uu)
        K[k] = -S_uu_inv @ S_ux
        d[k] = -S_uu_inv @ S_x

        # Update V_x and V_xx for the next step
        v_k[k] = S_x + (K[k].T @ S_u) + (S_ux.T @ d[k]) + (K[k].T @ S_uu @ d[k])
        V_k[k] = S_xx + (2* K[k].T @ S_ux) + (K[k].T @ S_uu @ K[k])
    return K, d

# iLQR algorithm
def iLQR(Q, Q_N, R, T, T_s, x0, threshold=1e-4)

    N = int(T / T_s)  # Number of time steps

    # Step 0: Initial control guess (zeros)
    u_bar = np.zeros((B.shape[1], N))
    prev_cost = np.inf
    iteration = 0  # Track the number of iterations

    # Step 1: Initial reference trajectory
    # Forward pass: compute state trajectory for current control sequence
    x_bar, _ = forward_pass_initial(x0, u_bar)

    while True:

        # Step 2: Backward pass to update control
        # Backward pass: compute optimal control updates (feedback K and feedforward d gains)
        K, d = backward_pass(Q, R, Q_f, x, u)

        # Step 3: Update control sequence using feedback and feedforward gains
        # We perform a new forward pass to compute the next state and control trajectory
        x, u = forward_pass(x0, x, u, K, d)

        # Compute the current cost
        current_cost = compute_cost(x, u, Q, R, Q_f)

        # Print iteration number and current cost
        print(f"Iteration {iteration + 1}, Cost: {current_cost}")

        # Check the difference between the current and previous cost
        cost_diff = abs(current_cost - prev_cost)
        if cost_diff < threshold:
            print(f"Converged after {iteration + 1} iterations with cost difference {cost_diff}")
            break  # Exit the loop if cost change is below the threshold

        prev_cost = current_cost  # Update the previous cost for the next iteration
        iteration += 1  # Increment the iteration counter

    return x, u

# Dummy compute_cost function for illustration (implement according to your specific problem)
def compute_cost(x, u, Q, R, Q_f):
    N = u.shape[1]
    cost = 0
    for k in range(N):
        cost += 0.5 * x[:, k].T @ Q @ x[:, k] + 0.5 * u[:, k].T @ R @ u[:, k]
    cost += x[:, -1].T @ Q_f @ x[:, -1]
    return cost