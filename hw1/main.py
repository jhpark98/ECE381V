import numpy as np

from cartpole import CartPole
from helper import plot_results, test_forward
from algorithm.lqr import LQR
from algorithm.ilqr import iLQR
from algorithm.qp import QP

np.set_printoptions(suppress=True)

def main():
    
    # test_forward(N, T, T_s, cartpole)

    T = 10.0   # 10.0 seconds time horizon
    T_s = 0.01 # 50 ms sampling time
    N = int(T / T_s)  # number of time steps
    
    Q_N = np.eye(4) * 100
    Q   = np.eye(4)
    R   = np.eye(1) * 0.1
    cartpole = CartPole(Q_N, Q, R, T_s) # Initialize environment

    # For LQR, start by estimating time-invariant matrices by linearizing the system around 0.
    x_bar, u_bar = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1), np.array([0.0]).reshape(-1, 1) # Linearizing w.r.t.

    # Initial condition
    x0_10_deg = np.array([np.deg2rad(10.0), 0.0, 0.0, 0.0]).reshape(-1, 1)
    x0_30_deg = np.array([np.deg2rad(30.0), 0.0, 0.0, 0.0]).reshape(-1, 1)

    run_LQR = True
    run_iLRR = True
    run_QP = True
    
    u_10 = None
    u_30 = None

    # Call the LQR DP function
    if run_LQR:
        x_traj_10, u_traj_10, c_10, _, _ = LQR(T, T_s, x_bar, u_bar, x0_10_deg, cartpole)
        x_traj_30, u_traj_30, c_30, _, _ = LQR(T, T_s, x_bar, u_bar, x0_30_deg, cartpole)
        plot_results(x_traj_10, u_traj_10, c_10, T, T_s, "results/LQR_10.png", True, "LQR with initial 10 degrees")
        plot_results(x_traj_30, u_traj_30, c_30, T, T_s, "results/LQR_30.png", True, "LQR with initial 30 degrees")
        
    if run_iLRR:
        u_10 = np.zeros(N).reshape(1, -1)
        u_30 = np.zeros(N).reshape(1, -1)
        # Call the iLQR function
        x_traj_10, u_traj_10, c_10 = iLQR(T, T_s, x0_10_deg, u_10, cartpole)
        x_traj_30, u_traj_30, c_30 = iLQR(T, T_s, x0_30_deg, u_30, cartpole)
        plot_results(x_traj_10.T, u_traj_10.T, c_10.reshape(-1, 1), T, T_s, "results/iLQR_10.png", True, "iLQR with initial 10 degrees")
        plot_results(x_traj_30.T, u_traj_30.T, c_30.reshape(-1, 1), T, T_s, "results/iLQR_30.png", True, "iLQR with initial 30 degrees")
    
    if run_QP:
        x_traj_10, u_traj_10, c_10 = QP(T, T_s, x0_10_deg, cartpole)
        x_traj_30, u_traj_30, c_30 = QP(T, T_s, x0_30_deg, cartpole)
        plot_results(x_traj_10, u_traj_10, c_10.reshape(-1, 1), T, T_s, "results/QP_10.png", True, "QP with initial 10 degrees")
        plot_results(x_traj_30, u_traj_30, c_30.reshape(-1, 1), T, T_s, "results/QP_30.png", True, "QP with initial 30 degrees")

if __name__ == '__main__':
    main()