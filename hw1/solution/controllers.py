import numpy as np

# Minimum number of iterations for iLQR algorithm
MIN_ITER = 50


class LQR_Controller:
    def __init__(self, approx_A_B, next_step, calculate_cost, calculate_ls):
        """Initialize the LQR controller
        Inputs:
            approx_A_B: function
            next_step: function
            calculate_cost: function
            calculate_ls: function
        """

        self.approx_A_B = approx_A_B
        self.next_step = next_step
        self.calculate_cost = calculate_cost
        self.calculate_ls = calculate_ls

    def calculate_control(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """
        x_t: 2D array of shape (n, 1)
        t: int
        """

        assert len(x_t.shape) == 2, "x0 must be a 2D array"
        assert (
            x_t.shape[0] == self.A.shape[1]
        ), "x0 must have the same number of rows as A"
        assert x_t.shape[1] == 1, "x0 must be a column vector"

        U = np.zeros((len(self.A) - t, self.B.shape[2], 1))

        P = self.Q[-1]
        for i in range(len(self.A) - t):
            L = -np.linalg.inv(
                self.R[-i - 1] + self.B[-i - 1].T @ P @ self.B[-i - 1]
            ) @ (self.B[-i - 1].T @ P @ self.A[-i - 1] + self.N[-i - 1].T)
            U[-i - 1] = L @ x_t

            if i == len(self.A) - t - 1:
                break
            P = (
                self.Q[-i - 2]
                + self.A[-i - 1].T @ P @ self.A[-i - 1]
                + (self.A[-i - 1].T @ P @ self.B[-i - 1] + self.N[-i - 1]) @ L
            )

        return U


class iLQR_Controller(LQR_Controller):
    def __init__(self, approx_A_B, next_step, calculate_cost, calculate_ls):
        super().__init__(approx_A_B, next_step, calculate_cost, calculate_ls)

    def backward_pass(self, X_hat, U_hat):
        """Implement the backward pass of the iLQR algorithm
        Inputs:
            X_hat: 3D array of shape (N, n, 1)
            U_hat: 3D array of shape (N, m, 1)
        Returns:
            K: 3D array of shape (N, m, n)
            d: 3D array of shape (N, m, 1)
        """

        # l_x = Q @ x
        # l_u = R @ u
        # l_xx = Q
        # l_uu = R
        # l_ux = np.zeros((u.shape[0], x.shape[0]))

        len_x = X_hat.shape[1]
        len_u = U_hat.shape[1]
        N = X_hat.shape[0]
        S = np.zeros((len_x, len_x))
        s = np.zeros((len_x, 1))

        K = np.zeros((N, len_u, len_x))
        d = np.zeros((N, len_u, 1))

        s = self.Q[-1] @ X_hat[-1]
        S = self.Q[-1]

        for i in range(N - 2, -1, -1):
            A, B = self.approx_A_B(X_hat[i], U_hat[i])
            l_x, l_u, l_xx, l_uu, l_ux = self.calculate_ls(
                X_hat[i], U_hat[i], self.Q[i], self.R[i]
            )
            Q_x = l_x + A.T @ s
            Q_u = l_u + B.T @ s
            Q_xx = l_xx + A.T @ S @ A
            Q_uu = l_uu + B.T @ S @ B
            Q_ux = l_ux + B.T @ S @ A

            d[i] = -np.linalg.inv(Q_uu) @ Q_u
            K[i] = -np.linalg.inv(Q_uu) @ Q_ux

            s = Q_x + K[i].T @ Q_uu @ d[i] + K[i].T @ Q_u + Q_ux.T @ d[i]
            S = Q_xx + K[i].T @ Q_uu @ K[i] + K[i].T @ Q_ux + Q_ux.T @ K[i]

        return K, d

    def forward_pass(self, X_hat, U_hat, K, d, a=1):
        """Implement the forward pass of the iLQR algorithm
        Inputs:
            X_hat: 3D array of shape (N, n, 1)
            U_hat: 3D array of shape (N, m, 1)
            K: 3D array of shape (N, m, n)
            d: 3D array of shape (N, m, 1)
            a: scalar
        Returns:
            X: 3D array of shape (N, n, 1)
            U: 3D array of shape (N, m, 1)
        """
        len_x = X_hat.shape[1]
        N = X_hat.shape[0]

        U = np.zeros_like(U_hat)
        delta_x = np.zeros((len_x, 1))
        X = np.zeros_like(X_hat)
        X[0] = X_hat[0]

        for i in range(N - 1):
            delta_x = X[i] - X_hat[i]
            U[i] = U_hat[i] + K[i] @ delta_x + a * d[i]
            X[i + 1] = self.next_step(X[i], U[i])

        return X, U

    def calculate_control(self, X_hat, U_hat):
        """Calculate the control for the given state and control trajectory
        Inputs:
            X_hat: 3D array of shape (N, n, 1)
            U_hat: 3D array of shape (N, m, 1)
        Returns:
            U_hat: 3D array of shape (N, m, 1)
        """
        delta_l = 1
        losses = [self.calculate_cost(X_hat, U_hat)[0]]
        i = 0
        while delta_l >= 1e-3 or i <= MIN_ITER:
            K, d = self.backward_pass(X_hat, U_hat)

            X_hat, U_hat = self.forward_pass(X_hat, U_hat, K, d)

            loss_new = self.calculate_cost(X_hat, U_hat)[0]

            losses.append(loss_new)

            if i != 0:
                delta_l = abs(loss_new - loss) / (loss + 1e-12)

            loss = loss_new

            i += 1

        return U_hat
