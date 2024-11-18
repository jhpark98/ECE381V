import numpy as np

from Environment import Environment
from PolicyIteration import PolicyIteration
from ValueIteration import ValueIteration
from visualization import gen_heatmap_value, gen_heatmap_policy

if __name__ == '__main__':

    # define grid-world
    N = 20
    grid_size = (N,N)
    goal = np.array((6,17))
    STOCHASTIC = 0.8
    env = Environment(grid_size, goal, STOCHASTIC)
    
    gamma = 0.99
    epsilon = 0.01 # convergence limit
    
    # ''' simulation '''
    # print("\n\n\n\n")
    # print(" ------ ------ Value Iteration ----- -----")
    # VI = ValueIteration(env, gamma)
    # VI.run(epsilon)

    # ''' visualization '''    
    # value_table, policy_table = VI.get_results()
    # print("plotting a heatmap of the optimal value function ...")
    # gen_heatmap_value(value_table, policy_table, plot=False, save=True, name="vi_value.png", title="Value Iteration")
    # print("plotting a heatmap of the optimal policy ...")
    # gen_heatmap_policy(policy_table, plot=False, save=True, name="vi_policy.png", title="Value Iteration")

    ''' simulation '''
    print("\n\n\n\n\n\n\n")
    print(" ------ ------ Policy Iteration ----- -----")
    PI = PolicyIteration(env, gamma)
    PI.run(epsilon)

    ''' visualization '''    
    value_table, policy_table = PI.get_results()
    print("plotting a heatmap of the optimal value function ...")
    gen_heatmap_value(value_table, policy_table, plot=False, save=True, name="pi_value.png", title="Policy Iteration")
    print("plotting a heatmap of the optimal policy ...")
    gen_heatmap_policy(policy_table, plot=False, save=True, name="pi_policy.png", title="Policy Iteration")