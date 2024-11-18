# This file contains the main code for the cartpole environment
import numpy as np

from typing import Tuple
from controllers import LQR_Controller, iLQR_Controller
from enum import Enum
from tqdm import tqdm


class ControllerType(Enum):
    """Controller types for the cartpole environment"""

    lqr = "lqr"
    ilqr = "ilqr"


class CartPole:
    def __init__(self, total_time: float, Ts: float, Q: np.ndarray, R: np.ndarray):
        """Initialize the cartpole environment
        Inputs:
            total_time: float
            Ts: float
            Q: 2D array of shape (n, n)
            R: 2D array of shape (m, m)
        """

        self.M = 10
        self.m = 8
        self.c = 0.1
        self.J = 5
        self.l = 1
        self.gamma = 0.01
        self.g = 9.8
        self.M_t = self.M + self.m
        self.Ts = Ts
        self.J_t = self.J + self.m * self.l**2
        self.total_time = total_time
        self.Q = Q
        self.R = R

    def calculate_ls(self, x: np.ndarray, u: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        For the given state and control returns the partial derivatives of the cost function
        Inputs:
            x: 2D array of shape (n, 1)
            u: 2D array of shape (m, 1)
            Q: 2D array of shape (n, n)
            R: 2D array of shape (m, m)
        Returns:
            l_x: 2D array of shape (n, 1)
            l_xx: 2D array of shape (n, n)
            l_u: 2D array of shape (m, 1)
            l_uu: 2D array of shape (m, m)
            l_ux: 2D array of shape (m, n)
        """

        l_x = Q @ x
        l_u = R @ u
        l_xx = Q
        l_uu = R
        l_ux = np.zeros((u.shape[0], x.shape[0]))

        return l_x, l_u, l_xx, l_uu, l_ux

    def approx_A_B(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray]:
        """
        For the given state and control, returns approximations of the A and B matrices
        Inputs:
            x: 2D array of shape (n, 1)
            u: 2D array of shape (m, 1)
        Returns:
            A: 2D array of shape (n, n)
            B: 2D array of shape (n, m)
        """

        theta = x[0, 0]
        q = x[1, 0]
        theta_dot = x[2, 0]
        q_dot = x[3, 0]
        F = u[0, 0]
        m = self.m
        l = self.l
        M_t = self.M_t
        J_t = self.J_t
        gamma = self.gamma
        c = self.c
        g = self.g

        A = np.zeros((x.shape[0], x.shape[0]))
        B = np.zeros((x.shape[0], u.shape[0]))

        A[0, 2] += 1
        A[1, 3] += 1

        A[2, 0] += (
            -2
            * (m**2)
            * (l**2)
            * np.cos(theta)
            * np.sin(theta)
            * (
                -M_t * gamma * theta_dot
                - (m**2) * (l**2) * np.sin(theta) * np.cos(theta) * (theta_dot**2)
                - c * m * l * np.cos(theta) * q_dot
                + M_t * m * g * l * np.sin(theta)
                + F * m * l * np.cos(theta)
            )
        ) / ((M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2)) ** 2) + (
            -(m**2) * (l**2) * (theta_dot**2) * np.cos(2 * theta)
            + c * m * l * q_dot * np.sin(theta)
            + M_t * m * g * l * np.cos(theta)
            - F * m * l * np.sin(theta)
        ) / (M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2))

        A[2, 2] += (
            -M_t * gamma
            - 2 * (m**2) * (l**2) * np.sin(theta) * np.cos(theta) * theta_dot
        ) / (M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2))

        A[2, 3] += (-c * m * l * np.cos(theta)) / (
            M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2)
        )

        A[3, 0] += (
            -2
            * (m**2)
            * (l**2)
            * np.cos(theta)
            * np.sin(theta)
            * (
                -m * l * np.cos(theta) * gamma * theta_dot
                - J_t * m * l * np.sin(theta) * (theta_dot**2)
                - J_t * c * q_dot
                + (m**2) * (l**2) * g * np.cos(theta) * np.sin(theta)
                + F * J_t
            )
        ) / ((M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2)) ** 2) + (
            (m**2) * (l**2) * g * np.cos(2 * theta)
            + m * l * np.sin(theta) * gamma * theta_dot
            - J_t * m * l * np.cos(theta) * (theta_dot**2)
        ) / (M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2))

        A[3, 2] += (
            -m * l * np.cos(theta) * gamma - 2 * J_t * m * l * np.sin(theta) * theta_dot
        ) / (M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2))

        A[3, 3] += (-J_t * c) / (M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2))

        B[2, 0] += (m * l * np.cos(theta)) / (
            M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2)
        )

        B[3, 0] += (J_t) / (M_t * J_t - (m**2) * (l**2) * (np.cos(theta) ** 2))

        A = np.eye(4) + self.Ts * A
        B = self.Ts * B

        return A, B

    def next_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        For the given state and control, returns the next state with the smallest simulation step
        Inputs:
            x: 2D array of shape (n, 1)
            u: 2D array of shape (m, 1)
        Returns:
            x_next: 2D array of shape (n, 1)
        """

        x_next = np.zeros_like(x)

        theta = x[0, 0]
        q = x[1, 0]
        theta_dot = x[2, 0]
        q_dot = x[3, 0]
        F = u[0, 0]

        den = 1 / (
            self.M_t * self.J_t - (self.m**2) * (self.l**2) * (np.cos(theta) ** 2)
        )

        num1 = (
            -self.M_t * self.gamma * theta_dot
            - (self.m**2) * (self.l**2) * np.sin(theta) * np.cos(theta) * (theta_dot**2)
            - self.c * self.m * self.l * np.cos(theta) * q_dot
            + self.M_t * self.m * self.g * self.l * np.sin(theta)
            + F * self.m * self.l * np.cos(theta)
        )
        num2 = (
            -self.m * self.l * np.cos(theta) * self.gamma * theta_dot
            - self.J_t * self.m * self.l * np.sin(theta) * (theta_dot**2)
            - self.J_t * self.c * q_dot
            + (self.m**2) * (self.l**2) * self.g * np.cos(theta) * np.sin(theta)
            + F * self.J_t
        )

        f1 = den * num1
        f2 = den * num2

        x_next[0, 0] = x[0, 0] + self.Ts * x[2, 0]
        x_next[1, 0] = x[1, 0] + self.Ts * x[3, 0]
        x_next[2, 0] = x[2, 0] + self.Ts * f1
        x_next[3, 0] = x[3, 0] + self.Ts * f2

        return x_next

    def calculate_cost(self, X: np.ndarray, U: np.ndarray) -> Tuple[float]:
        """
        For the given state and control return cost
        Inputs:
            X: 3D array of shape (T+1, n, 1)
            U: 3D array of shape (T, m, 1)
        Returns:
            cost: scalar
        """

        control_cost = 0
        state_cost = 0

        for t in range(U.shape[0]):
            control_cost += 1 / 2 * np.dot(U[t], self.R @ U[t])
            state_cost += 1 / 2 * np.dot(X[t + 1].T, self.Q @ X[t + 1])

        state_cost += 1 / 2 * np.dot(X[0].T, self.Q @ X[0])

        total_cost = control_cost + state_cost

        return total_cost[0, 0], control_cost[0, 0], state_cost[0, 0]

    def rollout(
        self,
        x0: np.ndarray,
        controller_type: ControllerType,
        X_hat: np.ndarray = None,
        U_hat: np.ndarray = None,
    ) -> Tuple[np.ndarray]:
        """
        For the given initial state and control, returns the trajectory of the system
        Inputs:
            x0: 2D array of shape (n, 1)
            controller_type: ControllerType
            X_hat: 3D array of shape (T+1, n, 1)
            U_hat: 3D array of shape (T, m, 1)
        Returns:
            X: 3D array of shape (T+1, n, 1)
            U: 3D array of shape (T, m, 1)
        """

        T = int(self.total_time / self.Ts)
        if np.any(X_hat):
            X = X_hat
        else:
            X = np.zeros((T + 1, x0.shape[0], 1))
        if np.any(U_hat):
            U = U_hat
        else:
            U = np.zeros((T, 1, 1))

        X[0] = x0

        if controller_type == ControllerType.lqr:
            controller = LQR_Controller(
                self.approx_A_B, self.next_step, self.calculate_cost, self.calculate_ls
            )

            A, B = self.approx_A_B(np.array([[0], [0], [0], [0]]), np.array([[0]]))
            controller.A = np.tile(A, (T, 1, 1))
            controller.B = np.tile(B, (T, 1, 1))
            controller.Q = np.tile(self.Q, (T, 1, 1))
            controller.R = np.tile(self.R, (T, 1, 1))
            controller.N = np.zeros((T, 1, 1))
        elif controller_type == ControllerType.ilqr:
            controller = iLQR_Controller(
                self.approx_A_B, self.next_step, self.calculate_cost, self.calculate_ls
            )
            controller.Q = np.tile(self.Q, (T, 1, 1))
            controller.R = np.tile(self.R, (T, 1, 1))
        else:
            raise ValueError("Invalid controller type")

        pbar = tqdm(range(T))
        pbar.set_description(
            f"Simulating {controller_type.value} controller for theta_0 = {x0[0, 0] * 180 / np.pi:.1f}"
        )
        for t in pbar:
            if controller_type == ControllerType.lqr:
                U[t] = controller.calculate_control(
                    X[t],
                    t,
                )[0]
            else:
                if t == 0:
                    X_hat[t:] = np.copy(X[t:])
                    U_hat[t:] = np.copy(U[t:])
                    U[t:] = controller.calculate_control(
                        X_hat[t:],
                        U_hat[t:],
                    )

            X[t + 1] = self.next_step(X[t], U[t])

        return X, U
