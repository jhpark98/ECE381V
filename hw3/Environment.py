import random 
import numpy as np

class Environment:
    """ Grid World

    The grid world is responsible for returning states, actions, and rewards.
    
    """
    def __init__(self, grid_size, goal, STOCHASTIC=0.8):
        
        self.WIDTH = grid_size[0]
        self.HEIGHT = grid_size[1]
        self.goal = goal # tuple
        
        # state, action, reward
        self.all_states = np.array(np.meshgrid(range(self.HEIGHT), range(self.WIDTH))).T.reshape(-1, 2)
        
        self.STOCHASTIC = STOCHASTIC
        self.possible_actions = [0,1,2,3] # [up,down,left,right].
        UNDERSIRED = (1 - self.STOCHASTIC) / len(self.possible_actions) # e.g. if STOCHASTIC = 0.8, UNDESIRED = 0.05
        DESIRED = 1 - (len(self.possible_actions)-1) * UNDERSIRED
        self.actions_xy = np.array([[1, 0],[-1, 0],[0, -1],[0, 1]]) # [up,down,left,right]
        self.TP = np.where(np.eye(4, dtype=bool), DESIRED, UNDERSIRED) # transition probability for each action
        
        self.reward = np.zeros((self.HEIGHT, self.WIDTH)) 
        self.reward[self.goal[0]][self.goal[1]] = 1.0

    def check_boundary(self, state):
        ''' return (next) valid state after checking boundary '''
        x = (0 if state[0] < 0 else self.WIDTH - 1 if state[0] > self.WIDTH - 1 else state[0])
        y = (0 if state[1] < 0 else self.HEIGHT - 1 if state[1] > self.HEIGHT - 1 else state[1])
        return np.array((x, y))

    def next_state(self, state, action):
        self.possible_actions = [0, 1, 2, 3]
        probabilities = self.TP[action]
        stochastic_action = random.choices(self.possible_actions, probabilities)[0]
        
        return self.check_boundary(state + self.actions_xy[stochastic_action])

    def state_after_action(self, state, action):
        """Given state and action pair, return the next state

        Args:
            state (np.array)
            action (np.array)

        Returns:
            np.array: next state
        """
        return self.check_boundary(state + self.actions_xy[action])

    # def get_reward(self, state, action):
    #     """Given state and action pair, return the reward value

    #     Args:
    #         state (np.array)
    #         action (np.array)

    #     Returns:
    #         float: reward value
    #     """
    #     next_state = self.state_after_action(state, action)
    #     return self.reward[next_state[0]+self.actions_xy[action][0]][next_state[1]+self.actions_xy[action][1]]
    
    def get_reward(self, next_state):
        """Given state and action pair, return the reward value"""
        return self.reward[next_state[0]][next_state[1]]

    def get_all_states(self):
        return self.all_states

    