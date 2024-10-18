import cvxpy as cp
import numpy as np

def QP(T, T_s, x_0, cartpole):

    # Problem parameters
    N = int(T / T_s)  # Time horizon
    n = 4   # State dimension
    m = 1   # Control dimension

    A, B = cartpole.A, cartpole.B

    # Cost matrices
    Q_N = cartpole.Q_N
    Q = cartpole.Q
    R = cartpole.R

    # Variables
    x = [cp.Variable((n, 1)) for _ in range(N+1)]  # State variables
    u = [cp.Variable((m, 1)) for _ in range(N)]    # Control variables

    # Objective function
    cost = 0.5 * cp.quad_form(x[N], Q_N)  # Terminal cost
    for k in range(N):
        cost += 0.5 * cp.quad_form(x[k], Q)  # State cost
        cost += 0.5 * cp.quad_form(u[k], R)  # Control cost

    # Constraints
    constraints = [x[0] == x_0]  # Initial state constraint
    for k in range(N):
        # Dynamics constraint with constant A, B, and time-varying d[k]
        constraints.append(x[k+1] == A @ x[k] + B @ u[k])

    # Solve the quadratic program
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Retrieve the solution
    x_arr = np.array([xk.value for xk in x]).reshape(-1, 4)
    u_arr = np.array([uk.value for uk in u]).reshape(-1, 1)

    # Pre-allocate J_arr
    J_arr = []

    # Compute cost for each time step (0 to N-1)
    for k in range(N):
        x_k = x_arr[k].reshape(-1, 1)  # State at time step k
        u_k = u_arr[k].reshape(-1, 1)  # Control at time step k

        # Compute the cost at step k
        J_k = 0.5 * (x_k.T @ Q @ x_k + u_k.T @ R @ u_k)
        J_arr.append(J_k.item())  # Store the cost as a scalar

    # Compute the terminal cost at step N
    x_N = x_arr[-1].reshape(-1, 1)  # Terminal state (x_N)
    J_N = 0.5 * (x_N.T @ Q_N @ x_N)
    J_arr.append(J_N.item())  # Store the terminal cost

    # Convert J_arr to a numpy array for further use
    J_arr = np.array(J_arr).reshape(-1, 1)

    assert x_arr.shape == (N+1, 4)
    assert u_arr.shape == (N, 1)
    assert J_arr.shape == (N+1, 1)

    return x_arr, u_arr, J_arr