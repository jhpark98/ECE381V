import numpy as np
from typing import Tuple
import sympy as sp

from helper import func_f_1, func_f_2, func_grad_x_f, func_grad_u_f

class CartPole:
    def __init__(self, Ts: float):
        """Initialize the cartpole environment
        Inputs:
            Ts: float (the simulation step size)
        """

        self.Ts = Ts

        # env const.
        self.M = 10.0     # kg
        self.m = 8.0      # kg
        self.c = 0.1      # Ns/m
        self.J = 5.0      # kgm^2/s^2
        self.l = 1.0      # m
        self.gamma = 0.01 # Nms
        self.g = 9.8      # m/s^2
        self.M_t = self.M + self.m
        self.J_t = self.J + self.m * self.l**2
        
        # self.det = 1 / (self.M_t*self.J_t - self.m**2*self.l**2)

        # self.f_1_sym = None
        # self.f_2_sym = None
        # self.grad_x_f_1_sym = None
        # self.grad_x_f_2_sym = None
        # self.grad_u_f_1_sym = None
        # self.grad_u_f_2_sym = None

        # self.f_1_val = None
        # self.f_2_val = None
        # self.grad_x_f_1_val = None
        # self.grad_x_f_2_val = None
        # self.grad_u_f_1_val = None
        # self.grad_u_f_2_val = None

        # self.find_analytical()


    def next_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """ h(x, u) full, non-linear dynamics """
        """
        For the given state and control, returns the next state
        Inputs:
            x: 2D array of shape (n, 1)
            u: 2D array of shape (m, 1)
        Returns:
            x_next: 2D array of shape (n, 1)
        """
        # state variables
        theta, q, theta_dot, q_dot, F = x[0, 0], x[1, 0], x[2, 0], x[3, 0], u[0, 0]
        # x_dot = f(x, u)
        x_dot = np.array([[x[2, 0]],
                          [x[3, 0]],
                          [func_f_1(theta, q, theta_dot, q_dot, F)],
                          [func_f_2(theta, q, theta_dot, q_dot, F)]]).astype(np.float64)

        ### TODO: HW1, approximate x_next with x + Ts * f(x, u)
        x_next = x + self.Ts * x_dot

        return x_next

    def grad_x_h(self, x, u):
        theta, q, theta_dot, q_dot, F = x[0, 0], x[1, 0], x[2, 0], x[3, 0], u[0, 0]
        I = np.eye(4)
        res = I + self.Ts * func_grad_x_f(theta, q, theta_dot, q_dot, F)
        assert res.shape == (4, 4)
        return res
    
    def grad_u_h(self, x, u):
        theta, q, theta_dot, q_dot, F = x[0, 0], x[1, 0], x[2, 0], x[3, 0], u[0, 0]
        res = self.Ts * func_grad_u_f(theta, q, theta_dot, q_dot, F)
        assert res.shape == (4, 1)
        return res

    def approx_A_B(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray]:
        """ h(x_bar, u_bar), linearized about (x_bar, u_bar) """
        """
        For the given state and control, returns approximations of the A and B matrices
        Inputs:
            x: 2D array of shape (n, 1)
            u: 2D array of shape (m, 1)
        Returns:
            A: 2D array of shape (n, n)
            B: 2D array of shape (n, m)
        """

        ### TODO: HW1, linearize the system around the given state and control
        A = self.grad_x_h(x, u)
        B = self.grad_u_h(x, u)

        return A, B