import numpy as np

""" Simulate LQR """
def LQR(T, T_s, x_bar, u_bar, x_0, cartpole):
    """
    Execute the LQR dynamic programming algorithm for a given system, including terminal state information.

    Parameters:
    T (float): Time horizon.
    T_s (float): Sampling time.
    Xu_bar (np.ndarray): Linearizing w.r.t.
    x_0 (np.ndarray): Initial state.

    Returns:
    x_list (list): List of state vectors over time.
    u_list (list): List of control vectors over time.
    J_list (list): List of costs over time.
    """

    # Cost matrices
    Q_N = cartpole.Q_N
    Q = cartpole.Q
    R = cartpole.R
    N = int(T / T_s)  # Number of time steps = 1000

    # Pre-allocate arrays
    P = [None] * (N + 1)  # Cost-to-go matrices
    F = [None] * N        # Feedback gain matrices = L_k
    x_list = [None] * (N + 1)  # List to store states (0 to N)
    x_list[0] = x_0
    u_list = [None] * N        # List to store control inputs (0 to N - 1)
    C_list = [None] * N        # List to store costs (0 to N)

    A, B = cartpole.approx_A_B(x_bar, u_bar)

    # Initialize terminal cost matrix (based on terminal state error)
    P[N] = Q_N
    # Backward dynamic programming recursion to find F and P
    for k in range(N - 1, -1, -1):
        P_next = P[k + 1]
        F[k] = - np.linalg.inv(R + B.T @ P_next @ B) @ (B.T @ P_next @ A)
        P[k] = Q + (F[k].T @ R @  F[k]) + ((A + B @ F[k]).T @ P_next @ (A + B @ F[k]))

    # Forward simulation to compute the state and control trajectories
    for k in range(N):
        x_k = x_list[k]
        u_k = F[k] @ x_k
        u_list[k] = u_k
        # xu = (x_k, u_k)
        x_next = cartpole.next_step(x_k, u_k) # x_k+1
        x_list[k+1] = x_next

        # Calculate cost at each step (penalizing deviation from the desired state)
        # C_k = 0.5 * (x_k.T @ P[k] @ x_k) # optimal cost-to-go (meaning, for t horizon)
        C_k = 0.5 * (x_k.T @ Q @ x_k + u_k.T @ R @ u_k) # per-step cost -> J_k+1 - J_k
        C_list[k] = C_k

    # Terminal cost (penalizing deviation from the terminal state)
    x_N = x_list[-1].reshape(-1, 1)
    C_N = 0.5 * x_N.T @ Q_N @ x_N
    C_list.append(C_N)

    x_list = np.hstack(x_list).T
    u_list = np.vstack(u_list)
    C_list = np.vstack(C_list)

    return x_list, u_list, C_list, F, P