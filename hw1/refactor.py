def grad_x_h(grad_x_f, values, T_s):

  """ ∇_x f(x, u) """
  grad_x_f_1_val, grad_x_f_2_val = grad_x_f

  I = np.eye(len(grad_x_f_1_val))

  grad_x_f_1_val = {var: derivative.subs(values) for var, derivative in grad_x_f_1_val.items()}
  grad_x_f_2_val = {var: derivative.subs(values) for var, derivative in grad_x_f_2_val.items()}

  grad_theta_f_1, grad_q_f_1, grad_theta_dot_f_1, grad_q_dot_f_1 = grad_x_f_1_val[theta], grad_x_f_1_val[q], grad_x_f_1_val[theta_dot], grad_x_f_1_val[q_dot]
  grad_theta_f_2, grad_q_f_2, grad_theta_dot_f_2, grad_q_dot_f_2 = grad_x_f_2_val[theta], grad_x_f_2_val[q], grad_x_f_2_val[theta_dot], grad_x_f_2_val[q_dot]
  grad_x_f = np.array([[0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0],
                       [grad_theta_f_1, grad_q_f_1, grad_theta_dot_f_1, grad_q_dot_f_1],
                       [grad_theta_f_2, grad_q_f_2, grad_theta_dot_f_2, grad_q_dot_f_2]])

  """ ∇_x h(x, u) """
  res = I + T_s * grad_x_f
  assert res.shape == (4, 4)
  return res

def grad_u_h(grad_u_f, values, T_s):

  """ ∇_u f(x, u) """
  grad_u_f_1_val, grad_u_f_2_val = grad_u_f

  grad_u_f_1_val = {var: derivative.subs(values) for var, derivative in grad_u_f_1_val.items()}
  grad_u_f_2_val = {var: derivative.subs(values) for var, derivative in grad_u_f_2_val.items()}

  grad_u_f_1 = grad_u_f_1_val[F]
  grad_u_f_2 = grad_u_f_2_val[F]
  grad_u_f = np.array([[0.0],
                       [0.0],
                       [grad_u_f_1],
                       [grad_u_f_2]]).astype(np.float64)

  """ ∇_u h(x, u) """
  res = T_s * grad_u_f
  assert res.shape == (4, 1)
  return res

def f(xu, values):
  """ Rate change of the state vector, f(x, u) """
  x, u = xu # current state & input
  _, _, theta_dot, q_dot = x[0, 0], x[1, 0], x[2, 0], x[3, 0] # Unpack state variables
  x_dot = np.array([[theta_dot],
                    [q_dot],
                    [f_1.subs(values)],
                    [f_2.subs(values)]]).astype(np.float64)
  assert x_dot.shape == (4, 1)
  return x_dot

def h(xu, T_s):
  """ Discretized, non-linearized full dynamics """
  x, u = xu # current state & input
  values = {
      theta: x[0, 0],
      q: x[1, 0],
      theta_dot: x[2, 0],
      q_dot: x[3, 0],
      F: u[0, 0],
  }

  x_next = T_s * f(xu, values)
  return x_next

def h_lin(xu, xu_bar, T_s):
  """ Discrete full dynamics linearized w.r.t. xu_bar  """
  x, u = xu             # current state & input
  x_bar, u_bar = xu_bar # reference of linearization

  # state, input, constant
  values = {
      theta: x_bar[0, 0],
      q: x_bar[1, 0],
      theta_dot: x_bar[2, 0],
      q_dot: x_bar[3, 0],

      F: u_bar[0, 0],
  }

  delta_x = x - x_bar
  delta_u = u - u_bar
  delta_x_next = grad_x_h(grad_x_f_val, values, T_s) @ delta_x + grad_u_h(grad_u_f_val, values, T_s) @ delta_u # delta_x_k+1
  x_next = x + delta_x_next
  return x_next

def discretize_system(xu_bar, T_s):
  """ Linearize discretized non-linear system w.r.t x_bar and u_bar. """
  x_bar, u_bar = xu_bar # reference of linearization
  # state, input, constant
  values = {
      theta: x_bar[0, 0],
      q: x_bar[1, 0],
      theta_dot: x_bar[2, 0],
      q_dot: x_bar[3, 0],
      F: u_bar[0, 0],
  }

  A = grad_x_h(grad_x_f, values, T_s)
  B = grad_u_h(grad_u_f, values, T_s)
  assert A.shape == (4, 4)
  assert B.shape == (4, 1)

  return A, B

"""## Set up the environment"""

# params = {
#   "M": 10.0,    # Mass M in kg
#   "m": 80.0,    # Mass m in kg
#   "c": 0.1,     # Damping coefficient c in Ns/m
#   "J": 5.0,     # Moment of inertia J in kgm^2/s^2
#   "l": 1.0,     # Length l in m
#   "gamma": 0.01, # Damping coefficient gamma in Nms
#   "g": 9.8      # Acceleration due to gravity in m/s^2
# }

"""## Simulate iLQR"""

# LQR cost function parameters
Q = np.eye(4)  # State cost matrix
R = np.eye(1)  # Control cost matrix
Q_f = np.eye(4)  # Terminal state cost matrix

# Simulation parameters
T = 10.0  # Total time
T_s = 0.01  # Sampling time

# Step 1 Forward pass
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

"""### Plot iLQR"""

x_traj_10, u_traj_10, cost_10, F_10, P_10 = iLQR(Q, Q_N, R, T, T_s, x0_10_deg, threshold=1e-4)
x_traj_10, u_traj_10, cost_10, F_10, P_10 = iLQR(Q, Q_N, R, T, T_s, x0_30_deg, threshold=1e-4)

plot_results(x_traj_10, u_traj_10, cost_10, T, T_s)

plot_results(x_traj_30, u_traj_30, cost_30, T, T_s)

import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, eig

# Model parameters

l_bar = 1.0  # length of bar
M = 10.0  # [kg]
m = 8  # [kg]
g = 9.8  # [m/s^2]

nx = 4  # number of state
nu = 1  # number of input
Q = np.diag([0.0, 1.0, 1.0, 0.0])  # state cost matrix
R = np.diag([0.01])  # input cost matrix

delta_t = 0.1  # time tick [s]
sim_time = 10.0  # simulation time [s]

u_pre = 0
x_rec = []
theta_rec = []
u_rec = []
u_del_rec = []

def main():         # x, x_dot, theta, theta_dot. repititive calculation of control and state for specified times
    x0 = np.array([ # initial value of state
        [0],
        [0.0],
        [np.pi*10/180],
        [0.0]
    ])

    x = np.copy(x0) # initial value of state. x is not an array, it's single state
    time = 0.0

    while sim_time > time:
        time += delta_t # do 100 time steps

        # calc control input. calculate the control and state 100 times
        u= lqr_control(x)

        A, B = get_model_matrix()
        x = A @ x + B @ u


    # Plot using subplots arranged horizontally
    fig, axs = plt.subplots(1, 4, figsize=(10, 2))
    # Plot x
    axs[0].plot(x_rec)
    axs[0].set_title('x (Position)')
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('x')
    # Plot theta
    axs[1].plot(np.rad2deg(theta_rec))
    axs[1].set_title('theta (Angle in Degrees)')
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('theta (degrees)')
    # Plot u
    axs[2].plot(u_rec)
    axs[2].set_title('u (Control Input)')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('u')
    # Plot delta_u
    axs[3].plot(u_del_rec)
    axs[3].set_title('delta_u (Control Input Variation)')
    axs[3].set_xlabel('Time Steps')
    axs[3].set_ylabel('delta_u')
    # Display the plots
    plt.tight_layout()
    plt.show()

def lqr_control(x):
    global u_pre
    A, B = get_model_matrix() # bring the model equation, which is x' = Ax + Bu
    K, _, _ = dlqr(A, B, Q, R) # Q,R is fixed, A, B is chaning with time. K is the gain matrix
    u = -K @ x # control update

    del_u = u - u_pre
    u_pre = u
    x_rec.append(x[0])
    theta_rec.append(x[2])
    u_rec.append(u[0,0])
    u_del_rec.append(del_u[0,0])
    return u


def dlqr(A, B, Q, R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    P = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    eigVals, eigVecs = eig(A - B @ K)
    return K, P, eigVals


def solve_DARE(A, B, Q, R, maxiter=150, eps=0.01):
    """
    Solve a discrete time_Algebraic Riccati equation (DARE)
    """
    P = Q

    for i in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ inv(R + B.T @ P @ B) @ B.T @ P @ A + Q



        if (abs(Pn - P)).max() < eps:
            break
        P = Pn

    return Pn


def get_model_matrix(): # x' = Ax + Bu
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    return A, B




def get_numpy_array_from_matrix(x):
    """
    get build-in list from matrix
    """
    return np.array(x).flatten()

def flatten(a):
    return np.array(a).flatten()


if __name__ == '__main__':
    main()

