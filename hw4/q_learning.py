import numpy as np
from tqdm import tqdm
import imageio


class QLearningAgent:
    def __init__(self, env, N):
        # Number of grid points for each state dimension
        self.Discrete_State_Space_Size = env.observation_space.shape[0] * [N]

        # Q Table
        self.q_table = np.random.uniform(
            low=-2, high=0, size=(self.Discrete_State_Space_Size + [env.action_space.n])
        )

        # Environment
        self.env = env

        # Grid size for each state
        self.window_size = (
            env.observation_space.high - env.observation_space.low
        ) / self.Discrete_State_Space_Size

    def get_action(self, state, epsilon):
        # TODO: Implement epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        # END TODO
        return action

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        # TODO: Implement Q-Learning update rule
        self.q_table[state][action] += alpha * (reward + gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])
        # END TODO
        return None

    def get_q_table(self):
        return self.q_table

    def get_policy(self):
        return np.argmax(self.q_table, axis=2)

    def get_value_function(self):
        return np.max(self.q_table, axis=2)

    def get_discrete_state(self, state):
        # TODO: Implement discretization of state
        x_min, x_max = -1.2, 0.6
        x_dot_min, x_dot_max = -0.07, 0.07
        N = 40
        discrete_state = (state - np.array([x_min, x_dot_min])) / (np.array([x_max - x_min, x_dot_max - x_dot_min])) * N
        # END TODO
        return tuple(discrete_state.astype(int))

    def train(
        self,
        num_episodes,
        min_epsilon,
        max_epsilon,
        env,
        alpha,
        gamma,
        log_every,
    ):
        episode_bar = tqdm(range(num_episodes))

        epsilons = np.linspace(max_epsilon, min_epsilon, num_episodes)
        epsilons = np.maximum(epsilons, 0)

        # For stats
        self.ep_rewards = []
        self.aggr_ep_rewards = {"Episode": [], "Avg": [], "Max": [], "Min": []}

        for episode in episode_bar:
            terminated, truncated = False, False
            discrete_state = self.get_discrete_state(env.reset()[0])

            episode_reward = 0

            while not (terminated or truncated):
                action = self.get_action(discrete_state, epsilons[episode])

                new_state, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward

                new_discrete_state = self.get_discrete_state(new_state)

                if not (terminated or truncated):
                    self.update_q_table(
                        discrete_state,
                        action,
                        reward,
                        new_discrete_state,
                        alpha,
                        gamma,
                    )

                elif new_state[0] >= env.goal_position:
                    self.q_table[discrete_state + (action,)] = 0

                discrete_state = new_discrete_state

            self.ep_rewards.append(episode_reward)

            if episode % log_every == 0 and episode > 0:
                average_reward = sum(self.ep_rewards[-log_every:]) / log_every
                self.aggr_ep_rewards["Episode"].append(episode)
                self.aggr_ep_rewards["Avg"].append(average_reward)
                self.aggr_ep_rewards["Max"].append(max(self.ep_rewards[-log_every:]))
                self.aggr_ep_rewards["Min"].append(min(self.ep_rewards[-log_every:]))

                episode_bar.set_description(
                    f"Episode: {episode:>5d}, Avg reward: {average_reward:>4.1f}, "
                    f"current epsilon: {epsilons[episode]:.2f}"
                )

        return None

    def visualize(self, env, save_path=None):
        frames = []

        terminated, truncated = False, False
        discrete_state = self.get_discrete_state(env.reset(seed=42)[0])

        frames.append(env.render())

        while not (terminated or truncated):
            action = np.argmax(self.q_table[discrete_state])
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_discrete_state = self.get_discrete_state(new_state)
            discrete_state = new_discrete_state
            frames.append(env.render())

        save_video(frames, save_path)
        return None


def save_video(frames, save_loc):
    with imageio.get_writer(save_loc, fps=30) as video:
        for frame in frames:
            video.append_data(frame)
