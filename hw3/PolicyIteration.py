import numpy as np

class PolicyIteration():
    """ Use Policy Iteration to compute optimal policy and value function
    
    until policy converges to optimal policy
        0. random policy
        1. policy evaluation
        2. policy improvement
        3. repeat 1 & 2

    """
    def __init__(self, env, gamma):
        '''
            action = idx of action_lst (up, down, left, right)
        '''
        
        self.env = env

        self.discount_factor = gamma

        self.value_table = np.zeros((self.env.WIDTH, self.env.HEIGHT))
        self.policy_table = np.full((self.env.WIDTH, self.env.HEIGHT, 4), 0.25)

        # Initialize the previous tables with infinity values
        self.prev_value_table = np.zeros((self.env.WIDTH, self.env.HEIGHT))
        self.prev_policy_table = np.full((self.env.WIDTH, self.env.HEIGHT, 4), 0.25)
        
        self.policy_initial = True

        self.iter_value = 0
        self.iter_policy = 0

    ''' Helper Function '''

    def get_value(self, state):
        ''' get corresponding value from the value table '''
        return self.value_table[state[0]][state[1]]
    
    # def get_action_stochastic(self, state, action):
    #     ''' Select an action stochastically based on the agent's dynamics. '''
    #     return action if random.random() < 0.8 else random.choice([a for a in self.actions if a != action])

    def check_value_convergence(self, threshold=1e-1):
        ''' True if converged '''
        delta = np.linalg.norm(self.value_table - self.prev_value_table)
        return (delta < threshold)

    def check_policy_convergence(self):
        ''' True if converged '''
        delta = np.sum(np.argmax(self.policy_table, axis=2) - np.argmax(self.prev_policy_table, axis=2))
        return (delta == 0.0)
    
    def possible_next_states(self, state):
        next_state_lst = [self.env.state_after_action(state, action) for action in self.env.possible_actions]
        return next_state_lst
        
    
    ''' Main Function '''

    # def update_value_table(self):
    #     ''' Policy Evaluation
    #         with fixed current policy, update value table once with simplified Bellman updates until convergence
    #     '''

    #     next_value_table = np.zeros((self.env.WIDTH, self.env.HEIGHT)) # initialize

    #     # compute Bellman optimality equation for all states
    #     for state in self.env.get_all_states():

    #         if tuple(state) == tuple(self.env.goal):
    #             # value for terminal state = 0
    #             next_value_table[state[0]][state[1]] = 0.0
    #             continue # terminal state reached. no action needed.
    #         value_lst = []
    #         for action in self.env.possible_actions: # (0,1,2,3)
    #             next_state_lst = self.possible_next_states(state)
    #             reward = np.array([self.env.get_reward(next_state) for next_state in next_state_lst]) # (4,)
    #             next_value = np.array([self.get_value(next_state) for next_state in next_state_lst]) # (4,)
    #             value_lst.append(np.sum(self.env.TP[action] * self.policy_table[state[0]][state[1]] * (reward + self.discount_factor * next_value)))
    #         next_value_table[state[0]][state[1]] = max(value_lst)
        
    #     self.prev_value_table = self.value_table # update old value
    #     self.value_table = next_value_table      # update new value
        
    def update_value_table(self):
        ''' Policy Evaluation
            Update value table using the current policy until convergence
        '''
        next_value_table = np.zeros((self.env.WIDTH, self.env.HEIGHT))  # initialize

        # compute value function under current policy
        for state in self.env.get_all_states():
            if tuple(state) == tuple(self.env.goal):
                # value for terminal state = 0
                next_value_table[state[0]][state[1]] = 0.0
                continue  # terminal state reached. no action needed.

            value = 0.0
            for action in self.env.possible_actions:  # (0,1,2,3)
                pi_s_a = self.policy_table[state[0]][state[1]][action]
                if pi_s_a == 0:
                    continue  # skip actions not in the policy
                # Get possible next states and their probabilities
                next_state_lst = self.possible_next_states(state)
                reward = np.array([self.env.get_reward(next_state) for next_state in next_state_lst])  # (4,)
                next_value = np.array([self.get_value(next_state) for next_state in next_state_lst])  # (4,)
                # Expected value over next states
                expected_value = np.sum(self.env.TP[action] * (reward + self.discount_factor * next_value))
                value += pi_s_a * expected_value  # weighted by policy probability

            next_value_table[state[0]][state[1]] = value

        self.prev_value_table = self.value_table  # update old value
        self.value_table = next_value_table       # update new value

    def update_policy_table(self):
        ''' Policy Improvement
            update policy table using value table
        '''
        
        next_policy_table = np.zeros((self.env.WIDTH, self.env.HEIGHT, 4))
        # next_policy_table = self.policy_table
        
        for state in self.env.get_all_states():
            if tuple(state) == tuple(self.env.goal):
                # terminal state reached. no action needed.
                next_policy_table[state[0]][state[1]] = np.zeros(4)
                continue

            value_lst = []
            for action in self.env.possible_actions: # (0,1,2,3)
                next_state_lst = self.possible_next_states(state)
                reward = np.array([self.env.get_reward(next_state) for next_state in next_state_lst]) # (4,)
                next_value = np.array([self.get_value(next_state) for next_state in next_state_lst]) # (4,)
                value_lst.append(np.sum(self.env.TP[action] * (reward + self.discount_factor * next_value)))
            next_policy_table[state[0]][state[1]] = np.eye(4, dtype=int)[np.argmax(value_lst)]

        self.prev_policy_table = self.policy_table # update old policy
        self.policy_table = next_policy_table      # update new policy

        # return next_policy_table
    
    def run(self, epsilon):
        ''' policy iteration algo. to run the value iteration algorithm until convergence is made '''
        converged_policy = False
        converged_value = False
        while not converged_policy:
            while not converged_value:
                self.update_value_table()
                converged_value = self.check_value_convergence(epsilon)
                self.iter_value += 1
            self.update_policy_table()
            converged_policy = self.check_policy_convergence()
            converged_value = False
        
        # print("----- ----- Summary of Policy Iteration ----- -----")
        print(f"{self.iter_value} sweeps over the state space required for convergence.")

    def simulate(self, start):
        ''' at each state, follow the optimal policy until the goal state is reached '''
        
        cur_state = start
        goal = self.env.goal

        trajectory = []
        trajectory.append(cur_state)
        while cur_state != tuple(goal):
            action = np.argmax(self.policy_table[cur_state[0]][cur_state[1]])
            next_state = self.env.next_state(cur_state, action) # stochastic
            cur_state = tuple(next_state) # update
            trajectory.append(cur_state)
        
        
        print(f"Goal {goal} is reached.")
        print("Finish simulation ...")
        
        return trajectory
        
    def get_results(self):
        return self.value_table, self.policy_table