# Main function to simulate cartpole environment and run iLQR
# To run this file please first install the requirements.txt file
# pip install -r requirements.txt
# Then run the file using the command
# python main.py
# This will generate the plots for the cartpole system
# The plots will be saved in the same directory as the main.py file


from cartpole import CartPole, ControllerType
from swarm_visualizer.utility.general_utils import set_plot_properties, save_fig
from swarm_visualizer.lineplot import plot_overlaid_ts
from swarm_visualizer.barplot import plot_grouped_barplot

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def simulate_cartpole(theta_0: float = 10, save_loc: str = "./theta_10.png") -> None:
    """
    Simulates the cartpole system for the given initial angle
    Inputs:
        theta_0: Angle in degrees
        save_loc: str
    Returns:
        None
    """

    # Setting up the environment for the given cartpole problem

    # Total time and time step for simulations
    total_time, Ts = 10, 0.1

    # Weight matrices for the cost function
    Q = np.diag([1, 1, 1, 1]) * 100
    R = np.diag([1]) * 0.01

    # Creating the cartpole object
    cartpole = CartPole(total_time, Ts, Q, R)

    # Time vector for simulation
    t = np.arange(0, total_time + Ts, Ts)

    # Reference trajectory for the cartpole system
    X_ref = np.zeros((len(t), 4, 1))
    U_ref = np.zeros((len(t) - 1, 1, 1))

    # State variable x = [theta, q,  theta_dot, q_dot]
    x_0 = np.array([[theta_0 / 180 * np.pi], [0], [0], [0]])

    # Simulate the cartpole system for the LQR controller
    X_LQR, U_LQR = cartpole.rollout(x_0, ControllerType.lqr)

    # Calculate the cost for the LQR controller
    LQR_total_cost, LQR_control_cost, LQR_state_cost = cartpole.calculate_cost(
        X_LQR, U_LQR
    )
    print("LQR Total Cost: ", LQR_total_cost)

    # Simulate the cartpole system for the iLQR controller
    X_hat = np.zeros_like(X_LQR)
    X_hat[0] = x_0
    U_hat = np.zeros_like(U_LQR)
    X_iLQR, U_iLQR = cartpole.rollout(x_0, ControllerType.ilqr, X_hat, U_hat)

    # Calculate the cost for the iLQR controller
    iLQR_total_cost, iLQR_control_cost, iLQR_state_cost = cartpole.calculate_cost(
        X_iLQR, U_iLQR
    )

    # Setting the plot properties before any plotting function
    set_plot_properties(legend_font_size=20, xtick_label_size=20, ytick_label_size=20)

    fig, ax = plt.subplots(3, 2, figsize=(20, 20))

    plt.plot()

    cart_position = {
        "Reference": {
            "xvec": t,
            "ts_vector": X_ref[:, 0, 0],
            "lw": 3,
            "color": "tab:green",
            "linestyle": "-",
        },
        "LQR": {
            "xvec": t,
            "ts_vector": X_LQR[:, 0, 0],
            "lw": 3,
            "color": "tab:blue",
            "linestyle": "-.",
        },
        "iLQR": {
            "xvec": t,
            "ts_vector": X_iLQR[:, 0, 0],
            "lw": 3,
            "color": "tab:orange",
            "linestyle": "--",
        },
    }

    plot_overlaid_ts(
        normalized_ts_dict=cart_position,
        ylabel="Pendulum Angle $(\\theta)$",
        xlabel=None,
        ax=ax[0, 0],
    )

    pendulum_angle = {
        "Reference": {
            "xvec": t,
            "ts_vector": X_ref[:, 1, 0],
            "lw": 3,
            "color": "tab:green",
            "linestyle": "-",
        },
        "LQR": {
            "xvec": t,
            "ts_vector": X_LQR[:, 1, 0],
            "lw": 3,
            "color": "tab:blue",
            "linestyle": "-.",
        },
        "iLQR": {
            "xvec": t,
            "ts_vector": X_iLQR[:, 1, 0],
            "lw": 3,
            "color": "tab:orange",
            "linestyle": "--",
        },
    }

    plot_overlaid_ts(
        normalized_ts_dict=pendulum_angle,
        ylabel="Cart Position $(q)$",
        xlabel=None,
        ax=ax[1, 0],
    )

    cart_velocity = {
        "Reference": {
            "xvec": t,
            "ts_vector": X_ref[:, 2, 0],
            "lw": 3,
            "color": "tab:green",
            "linestyle": "-",
        },
        "LQR": {
            "xvec": t,
            "ts_vector": X_LQR[:, 2, 0],
            "lw": 3,
            "color": "tab:blue",
            "linestyle": "-.",
        },
        "iLQR": {
            "xvec": t,
            "ts_vector": X_iLQR[:, 2, 0],
            "lw": 3,
            "color": "tab:orange",
            "linestyle": "--",
        },
    }

    plot_overlaid_ts(
        normalized_ts_dict=cart_velocity,
        ylabel="Pendulum Angular Velocity $(\\dot{\\theta})$",
        xlabel="Time $t$",
        ax=ax[2, 0],
    )

    pendulum_velocity = {
        "Reference": {
            "xvec": t,
            "ts_vector": X_ref[:, 3, 0],
            "lw": 3,
            "color": "tab:green",
            "linestyle": "-",
        },
        "LQR": {
            "xvec": t,
            "ts_vector": X_LQR[:, 3, 0],
            "lw": 3,
            "color": "tab:blue",
            "linestyle": "-.",
        },
        "iLQR": {
            "xvec": t,
            "ts_vector": X_iLQR[:, 3, 0],
            "lw": 3,
            "color": "tab:orange",
            "linestyle": "--",
        },
    }

    plot_overlaid_ts(
        normalized_ts_dict=pendulum_velocity,
        ylabel="Cart Velocity $(\\dot{q})$",
        xlabel=None,
        ax=ax[0, 1],
    )

    Force = {
        "Reference": {
            "xvec": t[:-1],
            "ts_vector": U_ref[:, 0, 0],
            "lw": 3,
            "color": "tab:green",
            "linestyle": "-",
        },
        "LQR": {
            "xvec": t[:-1],
            "ts_vector": U_LQR[:, 0, 0],
            "lw": 3,
            "color": "tab:blue",
            "linestyle": "-.",
        },
        "iLQR": {
            "xvec": t[:-1],
            "ts_vector": U_iLQR[:, 0, 0],
            "lw": 3,
            "color": "tab:orange",
            "linestyle": "--",
        },
    }

    plot_overlaid_ts(
        normalized_ts_dict=Force,
        ylabel="Force $(F)$",
        xlabel="Time $t$",
        ax=ax[1, 1],
    )

    df = pd.DataFrame(
        {
            "LQR": [LQR_total_cost, LQR_control_cost, LQR_state_cost],
            "iLQR": [iLQR_total_cost, iLQR_control_cost, iLQR_state_cost],
            "Cost Type": ["Total Cost", "Control Cost", "State Cost"],
        },
    )

    print("Differences for theta = %.1f" % theta_0)
    print("Total Cost Difference: ", LQR_total_cost - iLQR_total_cost)
    print("Control Cost Difference: ", LQR_control_cost - iLQR_control_cost)
    print("State Cost Difference: ", LQR_state_cost - iLQR_state_cost)
    print(
        "Proportional Difference: ", (LQR_total_cost - iLQR_total_cost) / LQR_total_cost
    )

    plot_grouped_barplot(
        df=df, x_var="Cost Type", y_var=["LQR", "iLQR"], ax=ax[2, 1], y_label="Cost"
    )

    # Rotating the xtick labels
    for tick in ax[2, 1].get_xticklabels():
        tick.set_rotation(0)

    fig.suptitle(
        "Cartpole System for $\\theta_0 = %.1f^\\circ$" % theta_0,
        fontsize=50,
        fontweight="bold",
    )
    fig.tight_layout()

    save_fig(fig, save_loc=save_loc)


if __name__ == "__main__":
    # Simulate the cartpole system for theta_0 = 10 and theta_0 = 30

    current_loc = os.path.dirname(os.path.realpath(__file__))

    simulate_cartpole(theta_0=10, save_loc=os.path.join(current_loc, "theta_10.pdf"))
    simulate_cartpole(theta_0=30, save_loc=os.path.join(current_loc, "theta_30.pdf"))
