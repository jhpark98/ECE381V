import numpy as np

"""## Simulate LQR"""
# def lqr_dp(Q, Q_N, R, T, T_s, xu_N, xu_bar, x_0, cartpole):
def lqr_dp(T, T_s, xu_bar, x_0, cartpole):
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

    Q_N = np.eye(4) * 100
    Q = np.eye(4)
    R = np.eye(1) * 0.1

    N = int(T / T_s)  # Number of time steps = 1000

    # Unpack
    x_bar, u_bar = xu_bar

    # Pre-allocate arrays
    P = [None] * (N + 1)  # Cost-to-go matrices
    F = [None] * N        # Feedback gain matrices = L_k
    x_list = [None] * (N + 1)  # List to store states (0 to N)
    x_list[0] = x_0
    u_list = [None] * N        # List to store control inputs (0 to N - 1)
    J_list = [None] * N        # List to store costs (0 to N)

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
        J_k = 0.5 * (x_k.T @ P[k] @ x_k)
        J_list[k] = J_k

    # Terminal cost (penalizing deviation from the terminal state)
    x_N = x_list[-1].reshape(-1, 1)
    J_N = 0.5 * x_N.T @ Q_N @ x_N
    J_list.append(J_N)

    x_list = np.hstack(x_list).T
    u_list = np.vstack(u_list)
    J_list = np.vstack(J_list)

    return x_list, u_list, J_list, F, P