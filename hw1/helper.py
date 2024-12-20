import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

""" ---------------------------------------- """

def plot_results(x_traj, u_traj, cost, T, T_s, fname, save=False, fig_title=None):
    """
    Plot the results of the LQR solution: state trajectory, control input, and accumulated cost over time.

    Parameters:
    x_traj (np.ndarray): State trajectory over time (shape: [N+1, state_dim]).
    u_traj (np.ndarray): Control input trajectory over time (shape: [N, control_dim]).
    cost (np.ndarray): Accumulated cost over time (shape: [N]).
    T (float): Total time horizon (in seconds).
    T_s (float): Sampling time (in seconds).
    fname (str): Filename to save the figure.
    save (bool): Whether to save the figure as a file.
    fig_title (str): Title for the entire figure.
    """
    N = int(T / T_s)  # number of time steps
    time = np.linspace(0, T, N+1)  # time vector for state trajectory
    time_u = np.linspace(0, T - T_s, N)  # time vector for control input and cost

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot state vector x
    axs[0].plot(time, x_traj[:, 0], label=r'$\theta$')
    axs[0].plot(time, x_traj[:, 1], label=r'$q$')
    axs[0].plot(time, x_traj[:, 2], label=r'$\dot{\theta}$')
    axs[0].plot(time, x_traj[:, 3], label=r'$\dot{q}$')
    axs[0].set_title('State Trajectory')
    axs[0].set_ylabel('State Value')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Plot control input u
    axs[1].plot(time_u, u_traj, label=r'$u$ (control input)', color='orange')
    axs[1].set_title('Control Input')
    axs[1].set_ylabel('Control Value')
    axs[1].grid(True)

    # Plot accumulated cost
    axs[2].plot(time, cost, label=r'Accumulated Cost', color='green')
    axs[2].set_title('Cost Over Time')
    axs[2].set_ylabel('Cost')
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True)

    # Set the overall figure title if provided
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)
    
    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to leave space for the figure title

    if save is True:
        plt.savefig(fname)
        print(f"saved {fname} ...")
    else:    
        plt.show()


def find_analytical():
    """ def find_f() """

    M, m, c, J, l, gamma, g = sp.symbols('M m c J l gamma g')          # Constant
    M_t, J_t = sp.symbols('M_t J_t')
    theta, q, theta_dot, q_dot = sp.symbols('theta q theta_dot q_dot') # State variables
    F = sp.symbols('F')                                                # Input variable

    # Rate of change of the state variables
    theta_dotdot, q_dotdot = sp.symbols('theta_dotdot q_dotdot')

    # Differential equations
    eq1 = sp.Eq((M_t * q_dotdot) - (m * l * theta_dotdot * sp.cos(theta)) + (c * q_dot) + (m * l * sp.sin(theta) * theta_dot**2), F)
    eq2 = sp.Eq((-m * l * sp.cos(theta) * q_dotdot) + (J_t * theta_dotdot) + (gamma * theta_dot) - (m * g * l * sp.sin(theta)), 0)

    # Solving for theta_dotdot and q_dotdot
    solution = sp.solve([eq1, eq2], (theta_dotdot, q_dotdot))

    # Assign f_1 and f_2
    f_1 = solution[theta_dotdot]
    f_2 = solution[q_dotdot]

    """ def compute_gradients(f_1, f_2): """
    # Take derivatives
    grad_x_f_1_sym = {var: sp.diff(f_1, var) for var in [theta, q, theta_dot, q_dot]}
    grad_x_f_2_sym = {var: sp.diff(f_2, var) for var in [theta, q, theta_dot, q_dot]}
    grad_u_f_1_sym = {var: sp.diff(f_1, var) for var in [F]}
    grad_u_f_2_sym = {var: sp.diff(f_2, var) for var in [F]}

    f_1_sym = f_1
    f_2_sym = f_2
    grad_x_f_1_sym = grad_x_f_1_sym
    grad_x_f_2_sym = grad_x_f_2_sym
    grad_u_f_1_sym = grad_u_f_1_sym
    grad_u_f_2_sym = grad_u_f_2_sym

    return f_1_sym, f_2_sym, grad_x_f_1_sym, grad_x_f_2_sym, grad_u_f_1_sym, grad_u_f_2_sym

def find_val(f_1_sym, f_2_sym, grad_x_f_1_sym, grad_x_f_2_sym, grad_u_f_1_sym, grad_u_f_2_sym):
    
    M, m, c, J, l, gamma, g = sp.symbols('M m c J l gamma g')          # Constant
    M_t, J_t = sp.symbols('M_t J_t')
    theta, q, theta_dot, q_dot = sp.symbols('theta q theta_dot q_dot') # State variables
    F = sp.symbols('F')

    values = {
        M: M_val,
        m: m_val,
        c: c_val,
        J: J_val,
        l: l_val,
        gamma: gamma_val,
        g: g_val,
        M_t: M_t_val,
        J_t: J_t_val
    }

    # Replace symbols with values
    f_1_val = f_1_sym.subs(values)
    f_2_val = f_2_sym.subs(values)
    grad_x_f_1_val = {var: derivative.subs(values) for var, derivative in grad_x_f_1_sym.items()}
    grad_x_f_2_val = {var: derivative.subs(values) for var, derivative in grad_x_f_2_sym.items()}
    grad_u_f_1_val = {var: derivative.subs(values) for var, derivative in grad_u_f_1_sym.items()}
    grad_u_f_2_val = {var: derivative.subs(values) for var, derivative in grad_u_f_2_sym.items()}

    # return f_1, f_2, (grad_x_f_1_sym, grad_x_f_2_sym), (grad_u_f_1_sym, grad_u_f_2_sym)
    # return f_1_val, f_2_val, (grad_x_f_1_val, grad_x_f_2_val), (grad_u_f_1_val, grad_u_f_2_val)

    f_1_val = f_1_val
    f_2_val = f_2_val
    grad_x_f_1_val = grad_x_f_1_val
    grad_x_f_2_val = grad_x_f_2_val
    grad_u_f_1_val = grad_u_f_1_val
    grad_u_f_2_val = grad_u_f_2_val


# ENV CONSTANT
M = 10.0     # kg
m = 8.0      # kg
c = 0.1      # Ns/m
J = 5.0      # kgm^2/s^2
l = 1.0      # m
gamma = 0.01 # Nms
g = 9.8      # m/s^2
M_t = M + m
J_t = J + m * l**2

def func_f_1(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = (-F*l*m*math.cos(theta)/(DET) - 
       M_t*g*l*m*math.sin(theta)/(DET) + 
       M_t*gamma*theta_dot/(DET) + 
       c*l*m*q_dot*math.cos(theta)/(DET) + 
       l**2*m**2*theta_dot**2*math.sin(theta)*math.cos(theta)/(DET))
    return res

def func_f_2(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = (-F*J_t/(DET) + 
       J_t*c*q_dot/(DET) + 
       J_t*l*m*theta_dot**2*math.sin(theta)/(DET) - 
       g*l**2*m**2*math.sin(theta)*math.cos(theta)/(DET) + 
       gamma*l*m*theta_dot*math.cos(theta)/(DET))
    return res

def func_grad_theta_f_1(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = (-2*F*l**3*m**3*math.sin(theta)*math.cos(theta)**2/(DET)**2 + 
         F*l*m*math.sin(theta)/(DET) - 
         2*M_t*g*l**3*m**3*math.sin(theta)**2*math.cos(theta)/(DET)**2 - 
         M_t*g*l*m*math.cos(theta)/(DET) + 
         2*M_t*gamma*l**2*m**2*theta_dot*math.sin(theta)*math.cos(theta)/(DET)**2 + 
         2*c*l**3*m**3*q_dot*math.sin(theta)*math.cos(theta)**2/(DET)**2 - 
         c*l*m*q_dot*math.sin(theta)/(DET) + 
         2*l**4*m**4*theta_dot**2*math.sin(theta)**2*math.cos(theta)**2/(DET)**2 - 
         l**2*m**2*theta_dot**2*math.sin(theta)**2/(DET) + 
         l**2*m**2*theta_dot**2*math.cos(theta)**2/(DET))
    return res

def func_grad_q_f_1(theta, q, theta_dot, q_dot, F):
    res = 0.0
    return res

def func_grad_theta_dot_f_1(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = (M_t*gamma/(DET) + 
             2*l**2*m**2*theta_dot*math.sin(theta)*math.cos(theta)/(DET))
    return res

def func_grad_q_dot_f_1(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = (c*l*m*math.cos(theta)/(DET))
    return res

def func_grad_theta_f_2(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = (-2*F*J_t*l**2*m**2*math.sin(theta)*math.cos(theta)/(DET)**2 + 
         2*J_t*c*l**2*m**2*q_dot*math.sin(theta)*math.cos(theta)/(DET)**2 + 
         2*J_t*l**3*m**3*theta_dot**2*math.sin(theta)**2*math.cos(theta)/(DET)**2 + 
         J_t*l*m*theta_dot**2*math.cos(theta)/(DET) - 
         2*g*l**4*m**4*math.sin(theta)**2*math.cos(theta)**2/(DET)**2 + 
         g*l**2*m**2*math.sin(theta)**2/(DET) - 
         g*l**2*m**2*math.cos(theta)**2/(DET) + 
         2*gamma*l**3*m**3*theta_dot*math.sin(theta)*math.cos(theta)**2/(DET)**2 - 
         gamma*l*m*theta_dot*math.sin(theta)/(DET))
    return res

def func_grad_q_f_2(theta, q, theta_dot, q_dot, F):
    res = 0.0
    return res

def func_grad_theta_dot_f_2(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = (2*J_t*l*m*theta_dot*math.sin(theta)/(DET) + 
             gamma*l*m*math.cos(theta)/(DET))
    return res

def func_grad_q_dot_f_2(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = (J_t*c/(DET))
    return res

def func_grad_u_f_1(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = -l*m*math.cos(theta)/(DET)
    return res

def func_grad_u_f_2(theta, q, theta_dot, q_dot, F):
    DET = -J_t*M_t + l**2*m**2*math.cos(theta)**2
    res = -J_t/(DET)
    return res

def func_grad_x_f(theta, q, theta_dot, q_dot, F):
    
    grad_theta_f_1     = func_grad_theta_f_1(theta, q, theta_dot, q_dot, F)
    grad_q_f_1         = func_grad_q_f_1(theta, q, theta_dot, q_dot, F)
    grad_theta_dot_f_1 = func_grad_theta_dot_f_1(theta, q, theta_dot, q_dot, F)
    grad_q_dot_f_1     = func_grad_q_dot_f_1(theta, q, theta_dot, q_dot, F)
    
    grad_theta_f_2     = func_grad_theta_f_2(theta, q, theta_dot, q_dot, F)
    grad_q_f_2         = func_grad_q_f_2(theta, q, theta_dot, q_dot, F)
    grad_theta_dot_f_2 = func_grad_theta_dot_f_2(theta, q, theta_dot, q_dot, F)
    grad_q_dot_f_2     = func_grad_q_dot_f_2(theta, q, theta_dot, q_dot, F)
    
    grad_x_f = np.array([[0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0],
                         [grad_theta_f_1, grad_q_f_1, grad_theta_dot_f_1, grad_q_dot_f_1],
                         [grad_theta_f_2, grad_q_f_2, grad_theta_dot_f_2, grad_q_dot_f_2]
                        ]).astype(np.float64)
    assert grad_x_f.shape == (4, 4)
    return grad_x_f

def func_grad_u_f(theta, q, theta_dot, q_dot, F):
    
    grad_u_f_1 = func_grad_u_f_1(theta, q, theta_dot, q_dot, F)
    
    grad_u_f_2 = func_grad_u_f_2(theta, q, theta_dot, q_dot, F)
    
    grad_u_f = np.array([[0.0],
                         [0.0],
                         [grad_u_f_1],
                         [grad_u_f_2]
                        ]).astype(np.float64)

    assert grad_u_f.shape == (4, 1)
    return grad_u_f

# f_1_sym, f_2_sym, grad_x_f_1_sym, grad_x_f_2_sym, grad_u_f_1_sym, grad_u_f_2_sym = find_analytical()

def test_forward(N, T, T_s, cartpole):
    x0 = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1) # initial state (4, 1)
    u_bar = np.array([0.1] * N).reshape(-1, 1)         # control input (N, 1)
    u_bar[0, 0] = 0.0
    x_full = forward(x0, u_bar, cartpole, full=True) # full dynamics
    x_lin = forward(x0, u_bar, cartpole, full=False) # linear dynamics
    plot_results(x_full.T, u_bar, np.zeros(N+1).reshape(-1, 1), T, T_s)
    plot_results(x_lin.T, u_bar, np.zeros(N+1).reshape(-1, 1), T, T_s)

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