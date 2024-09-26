import numpy as np
from cartpole import CartPole
from helper import plot_results
from lqr import lqr_dp
# from lqr import ilqr

def forward(x0, u_bar, cartpole, full=True):

  n_x = x0.shape[0]    # (4, 1)

  # Initialize state trajectory
  N = len(u_bar)
  x = np.zeros((n_x, N + 1))
  x[:, 0] = x0.reshape(-1,) # (-4,)
  
  x_approx = x0 
  u_approx = u_bar[0,0].reshape(-1, 1)
  assert x_approx.shape == (4, 1)
  assert u_approx.shape == (1, 1)
  A, B = cartpole.approx_A_B(x_approx, u_approx) # Linearized LQR

  # Forward pass through the horizon
  for k in range(N):
    x_k = x[:, k].reshape(-1,1)  # (4, 1)
    u_k = u_bar[k].reshape(-1,1) # (1, 1)

    if full: # non-linear dynamics
      x[:, k + 1] = cartpole.next_step(x0, u_bar[0,0].reshape(-1, 1)).reshape(-1,)
    else:    # linearized dynamics
      x[:, k + 1] = (A @ x_k + B @ u_k).reshape(-1,)
  return x

def main():
    T = 10.0   # 10.0 seconds time horizon
    T_s = 0.01 # 50 ms sampling time
    N = int(T / T_s)  # number of time steps
    
    cartpole = CartPole(T_s) # Initialize environment

    x0 = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1) # initial state (4, 1)
    u_bar = np.random.normal(0, 0.5, N).reshape(-1, 1)   # random action (N, 1)
    u_bar[0, 0] = 0.0
    x_full = forward(x0, u_bar, cartpole, full=True)
    x_lin = forward(x0, u_bar, cartpole, full=False)
    plot_results(x_full.reshape(-1, 4), u_bar, np.zeros(N+1).reshape(-1, 1), T, T_s)
    plot_results(x_lin.reshape(-1, 4), u_bar, np.zeros(N+1).reshape(-1, 1), T, T_s)

    print("Hello")
    import pdb; pdb.set_trace()

    # For LQR, start by estimating time-invariant matrices by linearizing the system around 0.
    x_bar, u_bar = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1), np.array([0.0]).reshape(-1, 1) # Linearizing w.r.t.

    xu_bar = (x_bar, u_bar)

    # Initial condition
    x0_10_deg = np.array([10.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    x0_30_deg = np.array([30.0, 0.0, 0.0, 0.0]).reshape(-1, 1)

    # Call the LQR DP function
    x_traj_10, u_traj_10, cost_10, F_10, P_10 = lqr_dp(T, T_s, xu_bar, x0_10_deg, cartpole)
    x_traj_30, u_traj_30, cost_30, F_30, P_30 = lqr_dp(T, T_s, xu_bar, x0_30_deg, cartpole)

    plot_results(x_traj_10, u_traj_10, cost_10, T, T_s)
    plot_results(x_traj_30, u_traj_30, cost_30, T, T_s)

if __name__ == '__main__':
    main()