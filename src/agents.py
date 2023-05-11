# Python base libraries
import os
# Visualization libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning libraries
import numpy as np
import tensorflow as tf

from abc import abstractclassmethod
from tqdm import tqdm
from collections import deque
from matplotlib_inline.backend_inline import set_matplotlib_formats
from typing import Tuple, List  
from gymnasium import Env

# Show all figures in svg format
set_matplotlib_formats('svg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('animation', html='jshtml')
mpl.rcParams['animation.embed_limit'] = 2**128

class Agent:
    def __init__(self, env) -> None:
        self.env = env

    @abstractclassmethod
    def act(self):
        """ Defines how the agent takes actions """
        

    @abstractclassmethod
    def evaluate(self):
        """ Defines how the agent is to be evaluated"""

    def plot_performance(
            self, 
            rewards: List, 
            title: str = "Frequency of wins, draws and loses", 
            xticks: List = ["Loses", "Draws", "Wins"],
            color: List = ["royalblue"], 
            figsize: Tuple = (16, 9), 
            fontfamily: str = "serif", 
            fontweight: str = "bold"
            ) -> None:
        """
        Plot performance of agent. It is a barplot with the frequency of wins, draws and loses.

        Args:
        -------
        rewards: list
            Rewards obtained by agent.
        title: str
            Title of plot. Default: "Frequency of wins, draws and loses".
        xticks: list
            X-axis ticks. Default: ["Loses", "Draws", "Wins"].
        color: list
            Color of bars. Default: ["royalblue"].
        figsize: tuple
            Figure size. Default: (16, 9).
        fontfamily: str
            Font family. Default: "serif".
        fontweight: str
            Font weight. Default: "bold".
        
        Returns:
        --------
        None
        """
        # Create dictionary with the number of times each reward was obtained
        unique, counts = np.unique(rewards, return_counts=True)
        stats = dict(zip(xticks, counts/len(rewards)))

        # Create figure and axes
        fig, (ax) = plt.subplots(1, 1, figsize=figsize)

        # Plot barplot with the number of times each reward was obtained
        ax.bar(stats.keys(), counts/len(rewards), color=color, edgecolor="black")

        # X-axis and Y-axis labels
        # i) X-axis
        plt.setp(ax.get_xticklines(), visible=False)
        plt.xticks(list(stats.keys()), fontsize=15, fontweight=fontweight, fontfamily=fontfamily)
        # ii) Y-axis
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)

        # Add title
        plt.title(title, fontsize=25, fontweight=fontweight, fontfamily=fontfamily)

        # Add text to each bar
        for key, value in stats.items():
            ax.text(key, value/2, f"{np.round(100*value, 1)}%", ha="center", color="#FFFFFF", fontsize=15, fontweight=fontweight, fontfamily=fontfamily)

        # Remove box from plot
        sns.despine(top=True, right=True, left=True, bottom=False)
        plt.show()

class RandomAgent(Agent):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.eval_rewards = []

    def act(self):
        action = self.env.action_space.sample()
        return action

    def evaluate(self, n_episodes: int = 1000):
        for episode in tqdm(range(n_episodes)):
            obs = self.env.reset()[0]
            done = False
            while not done:
                action = self.act()
                obs, reward, done, info, _ = self.env.step(action)

            if reward == 0:
                reward += 0.5
            
            elif reward == -1:
                reward = -100
                
            self.eval_rewards.append(reward)
        
        return self.eval_rewards
    
class QAgent():
    """
    Defines an agent that can be used to solve the Frozen Lake environment
    """
    def __init__(self,
                 env: Env,
                 num_states: int = 3,
                 num_actions: int = 2,
                 gamma: float = 0.8) -> None:
        """
        Initialize the agent

        Parameters
        ----------
        env : gym environment
            The environment to solve
        holes : list, optional
            The list of holes in the environment, by default None
        frozen : list, optional
            The list of frozen tiles in the environment, by default None
        
        Returns
        -------
        None
            The agent is initialized
        """
        # Initialize the environment, num_states and num_actions
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        # Initialize the Q-table
        self.q_table = np.zeros(
            [self.num_states,
            self.num_actions]
        )
        # Initialize the list of holes, frozen tiles, rewards and wins
        self.gamma = gamma
        self.train_rewards = []
        self.eval_rewards = []

    def act(self, state: Tuple, epsilon: float):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def train(self, alpha, gamma, epsilon, episodes=1000, modify_rewards=False):
        """
        Train the agent using the Q-learning algorithm

        Parameters
        ----------
        alpha : float
            The learning rate
        gamma : float
            The discount factor
        epsilon : float
            The exploration rate
        episodes : int, optional
            The number of episodes to train for, by default 1000
        modify_rewards : bool, optional
            Whether to modify the rewards, by default False

        Returns
        -------
        None
            The agent is trained
        """
        # For each episode
        for i in tqdm(range(episodes)):
            epsilon = max(1 - i / 500, 0.01)
            state = self.env.reset()[0]
            done = False
            
            # While the episode is not done
            while not done:
                action = self.act() # Exploit learned values

                next_state, reward, done, _, _ = self.env.step(action)

                # Modify the rewards
                if reward == 0:
                    reward += 0.5

                elif reward == -1:
                    reward = -100
                
                self.train_rewards.append(reward)
                
                # Update the Q-table
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value

                # Update the state
                state = next_state
                
                # Save the images
                self.env.render()

            return self.train_rewards
    
    def evaluate(self,
                 n_episodes: int = 100):
        """
        Evaluate the performance of the agent

        Parameters
        ----------
        episodes : int, optional
            The number of episodes to evaluate for, by default 100
        save_images : bool, optional
            Whether to save the images, by default False
        
        Returns
        -------
        total_epochs : int
            The total number of epochs
        total_penalties : int
            The total number of penalties
        """
        for episode in tqdm(range(n_episodes)):
            obs = self.env.reset()[0]
            done = False
            while not done:
                action = self.act()
                obs, reward, done, info, _ = self.env.step(action)

            if reward == 0:
                reward += 0.5
            
            elif reward == -1:
                reward = -100
                
            self.eval_rewards.append(reward)
        
        return self.eval_rewards

    def plot_q_training_curve(self, window=10):
        """
        Plot the training curve

        Parameters
        ----------
        window : int, optional
            The window size, by default 10

        Returns
        -------
        None
            The training curve is plotted
        """
        avg_rewards = np.array([
            np.mean(self.train_rewards[i-window:i])  
                if i >= window
                else np.mean(self.train_rewards[:i])
                for i in range(1, len(self.train_rewards))
        ])
        
        plt.figure(figsize=(12,8))
        plt.plot(avg_rewards, label='Mean Q Rewards')
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.show()
        return
        
class DQNAgent(Agent):
    def __init__(self,
                 env: Env,
                 num_states: int = 3,
                 num_actions: int = 2,
                 batch_size: int = 200,
                 gamma: float = 0.8) -> None:
        
        super().__init__(env)
        np.random.seed(42)
        tf.random.set_seed(42)
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory = deque(maxlen=2000)
        self.train_rewards = []
        self.eval_rewards = []
        self.frames = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.loss_fn = tf.keras.losses.mean_squared_error
        self.model = self.create_model()
    
    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation="elu", input_shape=(self.num_states,)),
            tf.keras.layers.Dense(64, activation="elu"),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(self.num_actions)
            ])
        
        return model
       
    def act(self, state: Tuple, epsilon: float):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            Q_values = self.model.predict([state], verbose=0)[0]
            return np.argmax(Q_values)
        
    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def play_one_step(self, state: Tuple, epsilon: float):
        action = self.act(state, epsilon)
        next_state, reward, done, info, _ = self.env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info
    
    def training_step(self):
        states, actions, rewards, next_states, dones = self.sample_experiences()
        next_Q_values = self.model.predict(next_states, verbose=0)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                        (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.num_actions)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self, n_episodes: int = 1000):
        for episode in tqdm(range(n_episodes)):
            epsilon = max(1 - episode / 500, 0.01)
            obs, _ = self.env.reset()
            done = False

            while not done:
                obs, reward, done, info = self.play_one_step(obs, epsilon)
                
            if reward == 0:
                reward += 0.5

            elif reward == -1:
                reward = -100
                
            self.train_rewards.append(reward)

            if episode > self.batch_size:
                self.training_step()

        return self.train_rewards
    
    def evaluate(self, n_episodes: int = 1000, save_images: bool = False):
        for episode in tqdm(range(n_episodes)):
            obs = self.env.reset()[0]
            done = False
            while not done:
                action = self.act(obs, epsilon=0)
                obs, reward, done, info, _ = self.env.step(action)
                
                if save_images:
                    img = self.env.render()
                    self.frames.append(img)

            if reward == 0:
                reward += 0.5
            
            elif reward == -1:
                reward = -100
                
            self.eval_rewards.append(reward)
        
        return self.eval_rewards
    
    def save(self, path):
        self.model.save(path)
    
    def plot_environment(self, figsize=(5,4)):
        """
        Given a gym environment, plot the environment

        Parameters
        ----------
        env : gym environment
            The environment to plot
        figsize : tuple, optional
            The size of the figure to plot, by default (5,4)
        
        Returns
        -------
        None
            The plot is shown
        """
        plt.figure(figsize=figsize)
        img = self.env.render()
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        return

    def update_scene(self, num, frames, patch):
        """
        Update the scene for the animation

        Parameters
        ----------
        num : int
            The current frame number
        frames : list
            The list of frames to plot
        patch : matplotlib patch
            The patch to update

        Returns
        -------
        matplotlib patch
            The updated patch
        """
        patch.set_data(frames[num])
        return patch

    def plot_animation(self, repeat=False, interval=20):
        """
        Plot an animation of the frames

        Parameters
        ----------
        repeat : bool, optional
            Whether to repeat the animation, by default False
        interval : int, optional
            The interval between frames, by default 20
        
        Returns
        -------
        matplotlib animation
            The animation
        """
        fig = plt.figure()
        patch = plt.imshow(self.frames[0])
        plt.axis('off')
        # Call the animator
        anim = animation.FuncAnimation(
            fig, self.update_scene, fargs=(self.frames, patch),
            frames=len(self.frames), repeat=repeat, interval=interval, cache_frame_data=False
        )
        plt.close()
        return anim

    def save_gif(self, path, anim, fps):
        """
        Save the animation as a gif

        Parameters
        ----------
        path : str
            The path to save the gif
        anim : matplotlib animation
            The animation to save

        Returns
        -------
        None
            The gif is saved
        """
        writergif = animation.PillowWriter(fps=fps) 
        anim.save(path, writer=writergif)
        return


class PolicyGradientAgent(Agent):
    """
    Policy gradient agent.

    Args:
    -------
    env: gym.Env
        Gym environment.
    num_states: int
        Number of states.
    num_actions: int
        Number of actions.
    gamma: float
        Discount factor. Default: 0.8.
    seed: int
        Seed for random number generator. Default: 13.

    Examples:
    ---------
    >>> from src.agents import PolicyGradientAgent
    >>> import gym
    >>> env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='rgb_array')
    >>> env.reset()
    >>> agent = PolicyGradientAgent(env=env, num_states=3, num_actions=2)
    >>> historic_rewards, mean_rewards = agent.train(n_iterations=100, n_episodes_per_update=32, n_max_steps=100)
    """

    def __init__(self, 
                 env: Env,
                 num_states: int = 3,
                 num_actions: int = 2, 
                 gamma: float = 0.8, 
                 seed: int = 13) -> None:
        """
        Initialize agent.

        Args:
        -------
        env: gym.Env
            Gym environment.
        num_states: int
            Number of states.
        num_actions: int
            Number of actions.
        gamma: float
            Discount factor. Default: 0.8.
        seed: int
            Seed for random number generator. Default: 13.

        Returns:
        --------
        None
        """
        # Call parent constructor
        super().__init__(env)
        # Set seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        # Loss function - Binary crossentropy
        self.loss_fn = tf.keras.losses.binary_crossentropy
        # Number of states and actions
        self.num_states = num_states
        self.num_actions = num_actions
        # Discount factor
        self.gamma = gamma
        # Build model
        self.model = self._build_model()
        # Initialize lists to store rewards
        # i) Historic rewards (i.e., rewards for each episode)
        self.historic_rewards = []
        # ii) Mean rewards (i.e., mean rewards for each iteration, which includes multiple episodes)
        self.mean_rewards = []


    def train(self, n_iterations: int, n_episodes_per_update: int, n_max_steps: int = 100) -> Tuple[List, List]:
        """
        Train agent.

        Args:
        -------
        n_iterations: int
            Number of iterations.
        n_episodes_per_update: int
            Number of episodes in each iteration.
        n_max_steps: int
            Maximum number of steps per episode. Default: 100.

        Returns:
        --------
        self.historic_rewards: list
            Historic rewards (i.e., rewards for each episode).
        self.mean_rewards: list
            Mean rewards (i.e., mean rewards for each iteration, which includes multiple episodes).
        """
        # Iterate over number of iterations
        for iteration in range(n_iterations):
            # Play multiple episodes
            all_rewards, all_grads, end_rewards = self._play_multiple_episodes(n_episodes_per_update, n_max_steps)
            # Compute mean reward for current iteration
            mean_reward = sum(map(sum, all_rewards)) / n_episodes_per_update
            # Print current iteration and mean reward
            print("\rIteration: {}/{}, mean reward: {:.1f}  ".format(iteration + 1, n_iterations, mean_reward), end="")

            # Store final rewards in historic rewards
            for reward in end_rewards:
                self.historic_rewards.append(reward)
            # Store mean reward in mean rewards
            self.mean_rewards.append(mean_reward)

            # Get mean gradients
            all_final_rewards = self._discount_and_normalize_rewards(all_rewards)
            all_mean_grads = []
            for var_index in range(len(self.model.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_grads[episode_index][step][var_index]
                    for episode_index, final_rewards in enumerate(all_final_rewards)
                        for step, final_reward in enumerate(final_rewards)], axis=0)
                all_mean_grads.append(mean_grads)

            # Apply gradients
            self.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))
        
        return self.historic_rewards, self.mean_rewards


    def evaluate(self, n_episodes: int, n_max_steps: int = 100) -> List:
        """
        Evaluate agent once it has been trained.

        Args:
        -------
        n_episodes: int
            Number of episodes.
        n_max_steps: int
            Maximum number of steps per episode. Default: 100.

        Returns:
        --------
        final_rewards: list
            Final rewards (i.e., rewards for each episode, +1 if win, -1 if lose, 0 if draw).
        """
        # List where to store all final rewards obtained (one per episode, +1 if win, -1 if lose, 0 if draw)
        final_rewards = []

        # Iterate over episodes
        for episode in range(n_episodes):
            # Reset environment
            obs = self.env.reset()[0]
            # Iterate over steps within the episode
            for step in range(n_max_steps):
                # Play one step and get next state, reward, if episode is done and gradients
                obs, reward, done, grads = self._play_one_step(obs)
                # If episode is done, append final reward and break inner loop
                if done:
                    final_rewards.append(reward)
                    break
        
        return final_rewards


    def plot_performance(
            self, 
            rewards: List, 
            title: str = "Frequency of wins, draws and loses", 
            xticks: List = ["Loses", "Draws", "Wins"],
            color: List = ["royalblue"], 
            figsize: tuple = (16, 9), 
            fontfamily: str = "serif", 
            fontweight: str = "bold"
            ) -> None:
        """
        Plot performance of agent. It is a barplot with the frequency of wins, draws and loses.

        Args:
        -------
        rewards: list
            Rewards obtained by agent.
        title: str
            Title of plot. Default: "Frequency of wins, draws and loses".
        xticks: list
            X-axis ticks. Default: ["Loses", "Draws", "Wins"].
        color: list
            Color of bars. Default: ["royalblue"].
        figsize: tuple
            Figure size. Default: (16, 9).
        fontfamily: str
            Font family. Default: "serif".
        fontweight: str
            Font weight. Default: "bold".
        
        Returns:
        --------
        None
        """
        # Create dictionary with the number of times each reward was obtained
        unique, counts = np.unique(rewards, return_counts=True)
        stats = dict(zip(xticks, counts/len(rewards)))

        # Create figure and axes
        fig, (ax) = plt.subplots(1, 1, figsize=figsize)

        # Plot barplot with the number of times each reward was obtained
        ax.bar(stats.keys(), counts/len(rewards), color=color, edgecolor="black")

        # X-axis and Y-axis labels
        # i) X-axis
        plt.setp(ax.get_xticklines(), visible=False)
        plt.xticks(list(stats.keys()), fontsize=15, fontweight=fontweight, fontfamily=fontfamily)
        # ii) Y-axis
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)

        # Add title
        plt.title(title, fontsize=25, fontweight=fontweight, fontfamily=fontfamily)

        # Add text to each bar
        for key, value in stats.items():
            ax.text(key, value/2, f"{np.round(100*value, 1)}%", ha="center", color="#FFFFFF", fontsize=15, fontweight=fontweight, fontfamily=fontfamily)

        # Remove box from plot
        sns.despine(top=True, right=True, left=True, bottom=False)
        plt.show()


    def save(self, model_dir: str, model_name: str) -> None:
        """
        Save model.

        Args:
        -------
        model_dir: str
            Directory where to save model.
        model_name: str
            Name of model in .h5 format.
            
        Returns:
        --------
        None
        """
        # Create directory where to save model, if it does not exist
        os.makedirs(model_dir, exist_ok=True)
        # Save model
        model_path = os.path.join(model_dir, model_name)
        self.model.save(model_path)


    def _build_model(self) -> tf.keras.models.Sequential:
        """
        Build model.

        Args:
        -------
        None

        Returns:
        --------
        model: tf.keras.models.Sequential
        """
        # Build model using TensorFlow's Keras API
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation="elu", input_shape=(self.num_states, )),
            tf.keras.layers.Dense(64, activation="elu"),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
        # Return model
        return model


    def _play_one_step(self, obs: np.array) -> Tuple[np.array, float, bool, List]:
        """
        Play one step.

        Args:
        -------
        obs: np.array
            Current state.

        Returns:
        --------
        obs: np.array
            Next state.
        reward: float
            Reward obtained.
        done: bool
            Whether the episode is done or not.
        grads: list
            Gradients of the model.
        """
        with tf.GradientTape() as tape:
            # Probability of not requesting additional card
            stop_proba = self.model(tf.constant([obs], dtype=tf.float32))
            # Action is 1 if random number is greater than stop_proba, 0 otherwise
            action = (tf.random.uniform([1, 1]) > stop_proba)
            # y_target is 1 if action is 0 and vice versa
            y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
            # Loss is the difference between y_target and stop_proba
            loss = tf.reduce_mean(self.loss_fn(y_target, stop_proba))
        # Compute gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Take action, the environment changes
        obs, reward, done, info, _ = self.env.step(int(action[0, 0].numpy()))
        
        return obs, reward, done, grads


    def _play_multiple_episodes(self, n_episodes: int, n_max_steps: int = 100) -> Tuple[List, List, List]:
        """
        Play multiple episodes.

        Args:
        -------
        n_episodes: int
            Number of episodes.
        n_max_steps: int
            Maximum number of steps per episode. Default: 100.

        Returns:
        --------
        all_rewards: list
            Rewards for each episode.
        all_grads: list
            Gradients of the model for each episode.
        final_rewards: list
            Final rewards (i.e., rewards for each episode, +1 if win, -1 if lose, 0 if draw).
        """
        # List where to store all rewards obtained (one list per episode)
        all_rewards = []
        # List where to store all final rewards obtained (one per episode, +1 if win, -1 if lose, 0 if draw)
        final_rewards = []
        # List where to store all gradients obtained (one list per episode)
        all_grads = []

        # Iterate over episodes
        for episode in range(n_episodes):
            # List where to store rewards obtained in the current episode (one per step)
            current_rewards = []
            # List where to store gradients obtained in the current episode (one per step)
            current_grads = []
            # Reset environment
            obs = self.env.reset()[0]
            # Iterate over steps within the episode
            for step in range(n_max_steps):
                # Play one step and get next state, reward, if episode is done and gradients
                obs, reward, done, grads = self._play_one_step(obs)
                # Add reward and gradients to the current lists
                current_rewards.append(reward)
                current_grads.append(grads)
                # If episode is done, append final reward and break inner loop
                if done:
                    final_rewards.append(reward)
                    break
            # Append current rewards and gradients to the global lists
            all_rewards.append(current_rewards)
            all_grads.append(current_grads)

        return all_rewards, all_grads, final_rewards


    def _discount_rewards(self, rewards: List) -> np.array:
        """
        Discount rewards.

        Args:
        -------
        rewards: list
            Rewards.
            
        Returns:
        --------
        discounted: np.array
            Discounted rewards.
        """
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * self.gamma

        return discounted


    def _discount_and_normalize_rewards(self, all_rewards: List) -> List:
        """
        Discount and normalize rewards.

        Args:
        -------
        all_rewards: list
            Rewards for each episode.

        Returns:
        --------
        Discounted and normalized rewards in list format.
        """
        all_discounted_rewards = [self._discount_rewards(rewards) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        # Reward normalization
        # i) Mean
        reward_mean = flat_rewards.mean()
        # ii) Standard deviation
        reward_std = flat_rewards.std()
        # iii) Normalization
        discounted_normalized_rewards = [
            (discounted_rewards - reward_mean) / reward_std 
            for discounted_rewards in all_discounted_rewards
            ]
        
        return discounted_normalized_rewards