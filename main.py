import numpy as np
import matplotlib.pyplot as plt

from cartpole import CartPole
from helper import plot_results
from lqr import lqr_dp
from ilqr import iLQR

np.set_printoptions(suppress=True)

def forward(x0, u_bar, cartpole, full=True):

  n_x = x0.shape[0]    # (4, 1)

  # Initialize state trajectory
  N = len(u_bar)
  x = np.zeros((n_x, N + 1))
  x[:, 0] = x0.reshape(-1,) # (4,)
  
  x_approx = x0
  u_approx = u_bar[0,0].reshape(-1, 1)
  assert x_approx.shape == (4, 1)
  assert u_approx.shape == (1, 1)
  
  A, B = cartpole.approx_A_B(x_approx, u_approx) # Linearized LQR

  # Forward pass through the horizon
  for k in range(N):
    # state & action @ time k
    x_k = x[:, k].reshape(-1,1)  # (4, 1)
    u_k = u_bar[k].reshape(-1,1) # (1, 1)

    if full: # non-linear dynamics
      x[:, k + 1] = cartpole.next_step(x_k, u_k).reshape(-1,)
    else:    # linearized dynamics
      x_next = A @ x_k + B @ u_k
      print(f"x_next: {x_next}")
      x[:, k + 1] = x_next.reshape(-1,)
  return x

def main():
    T = 10.0   # 10.0 seconds time horizon
    T_s = 0.01 # 50 ms sampling time
    N = int(T / T_s)  # number of time steps

    cartpole = CartPole(T_s) # Initialize environment

    # x0 = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1) # initial state (4, 1)
    # u_bar = np.array([0.1] * N).reshape(-1, 1)         # control input (N, 1)
    # u_bar[0, 0] = 0.0
    # x_full = forward(x0, u_bar, cartpole, full=True) # full dynamics
    # x_lin = forward(x0, u_bar, cartpole, full=False) # linear dynamics
    # plot_results(x_full.T, u_bar, np.zeros(N+1).reshape(-1, 1), T, T_s)
    # plot_results(x_lin.T, u_bar, np.zeros(N+1).reshape(-1, 1), T, T_s)
    # import pdb; pdb.set_trace()

    # a_dummy = np.ones(N+1).reshape(-1, 1)
    # b_dummy = a_dummy * 2.0
    # c_dummy = a_dummy * 3.0
    # d_dummy = a_dummy * 4.0
    # DUMMY = np.hstack([a_dummy, b_dummy, c_dummy, d_dummy]).reshape(-1, 4)
    # plot_results(DUMMY.reshape(-1, 4), u_bar, np.zeros(N+1).reshape(-1, 1), T, T_s)


    # For LQR, start by estimating time-invariant matrices by linearizing the system around 0.
    x_bar, u_bar = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1), np.array([0.0]).reshape(-1, 1) # Linearizing w.r.t.

    # Initial condition
    x0_10_deg = np.array([np.deg2rad(10.0), 0.0, 0.0, 0.0]).reshape(-1, 1)
    x0_30_deg = np.array([np.deg2rad(30.0), 0.0, 0.0, 0.0]).reshape(-1, 1)

    LQR = True
    iLRR = True

    u_10 = None
    u_30 = None

    # Call the LQR DP function
    if LQR:
        x_traj_10, u_traj_10, cost_10, F_10, P_10 = lqr_dp(T, T_s, x_bar, u_bar, x0_10_deg, cartpole)
        x_traj_30, u_traj_30, cost_30, F_30, P_30 = lqr_dp(T, T_s, x_bar, u_bar, x0_30_deg, cartpole)
        plot_results(x_traj_10, u_traj_10, cost_10, T, T_s, "LQR_10.png", True)
        plot_results(x_traj_30, u_traj_30, cost_30, T, T_s, "LQR_30.png", True)
        
    if iLRR:
        u_10 = np.zeros(N).reshape(1, -1)
        u_30 = np.zeros(N).reshape(1, -1)
            
        # Call the iLQR function
        x_traj_10, u_traj_10, cost_10 = iLQR(T, T_s, x0_10_deg, u_10, cartpole)
        x_traj_30, u_traj_30, cost_30 = iLQR(T, T_s, x0_30_deg, u_30, cartpole)
        plot_results(x_traj_10.T, u_traj_10.T, cost_10.reshape(-1, 1), T, T_s, "iLQR_10.png", True)
        plot_results(x_traj_30.T, u_traj_30.T, cost_30.reshape(-1, 1), T, T_s, "iLQR_30.png", True)

if __name__ == '__main__':
    main()