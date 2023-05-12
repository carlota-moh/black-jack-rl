# Python base libraries
import os
# Visualization libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, 
    Dense
)

from abc import abstractclassmethod
from tqdm import tqdm
from collections import deque
from matplotlib_inline.backend_inline import set_matplotlib_formats
from typing import Tuple, List  
from gymnasium import Env
from collections import defaultdict
import random

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
        return

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
    
    
class QAgent(Agent):
    def __init__(self, env:Env, learning_rate, discount_factor, exploration_rate):
        super().__init__(env)
        self.env=env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.Q = defaultdict(lambda: np.random.uniform(low=-0.001, high=0.001, size=(2,)))
        self.is_trained=False
        self.train_rewards=[]
        self.eval_rewards=[]

    def get_action(self, state):
        # Elegir una acción epsilon-greedy
        if random.uniform(0, 1) < self.exploration_rate:
            # Acción aleatoria
            action = random.randint(0, 1)
        else:
            # Acción greedy
            action = np.argmax(self.Q[state])

        return action

    def update_q(self, state, action, reward, next_state):
        # Actualizar la matriz Q
        old_value = self.Q[state][action]
        next_max = np.max(self.Q[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.Q[state][action] = new_value
    # Entrenar al agente
    def train(self, episodes):
        # Loop de entrenamiento
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                # Elegir una acción y tomarla
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)

                # Actualizar la matriz Q
                self.update_q(state, action, reward, next_state)

                # Actualizar el estado actual
                state = next_state

            self.train_rewards.append(reward)
        self.is_trained=True

        

    def evaluate(self, episodes):
        assert self.is_trained is True, "You need to train the agent before evaluating it!"
        # Contadores de victorias, derrotas y empates
        wins = 0
        losses = 0
        draws = 0

        # Loop de juego
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                # Elegir una acción greedy
                action = np.argmax(self.Q[state])

                # Tomar la acción y obtener la información del entorno
                next_state, reward, done, info = self.env.step(action)

                # Actualizar el estado actual
                state = next_state

            # Actualizar los contadores de victorias, derrotas y empates
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1
            self.eval_rewards.append(reward)
            
        print("Wins: ", (wins/episodes)*100, ' %')
        print("Looses: ", (losses/episodes)*100, ' %')
        print("Draws: ", (draws/episodes)*100, ' %')

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
    def __init__(
        self,
        env: Env,
        num_states: int = 3,
        num_actions: int = 2,
        batch_size: int = 200,
        gamma: float = 0.8
    ) -> None:
        """
        Initialize the agent

        Parameters
        ----------
        env : Env
            The environment
        num_states : int, optional
            The number of states, by default 3
        num_actions : int, optional
            The number of actions, by default 2
        batch_size : int, optional
            The batch size, by default 200
        gamma : float, optional
            The discount factor, by default 0.8
        
        Returns
        -------
        None
            The agent is initialized
        """
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
        """
        Create the model
        
        Returns
        -------
        tf.keras.models.Sequential
            The model
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation="elu", input_shape=(self.num_states,)),
            tf.keras.layers.Dense(64, activation="elu"),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(self.num_actions)
        ])
        
        return model
       
    def act(self, state: Tuple, epsilon: float):
        """
        Performs an action

        Parameters
        ----------
        state : Tuple
            The state
        epsilon : float
            The epsilon
        
        Returns
        -------
        int
            The action
        """
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            Q_values = self.model.predict([state], verbose=0)[0]
            return np.argmax(Q_values)
        
    def sample_experiences(self):
        """
        Sample experiences from the replay memory

        Returns
        -------
        Tuple
            The experiences
        """
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def play_one_step(self, state: Tuple, epsilon: float):
        """
        Play one step.

        Parameters
        ----------
        state : Tuple
            The state
        epsilon : float
            The epsilon
        
        Returns
        -------
        obs: np.array
            Next state.
        reward: float
            Reward obtained.
        done: bool
            Whether the episode is done or not.
        info: list
            Gradients of the model.
        """
        action = self.act(state, epsilon)
        next_state, reward, done, info, _ = self.env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info
    
    def training_step(self):
        """
        Perform one training step.

        Returns
        -------
        None
            The model is updated in place.
        """
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
        """
        Train the agent

        Parameters
        ----------
        n_episodes : int, optional
            The number of episodes. 1000 by default

        Returns
        -------
        list
            The rewards
        """
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
        """
        Evaluate the agent

        Parameters
        ----------
        n_episodes : int, optional
            The number of episodes. 1000 by default
        save_images : bool, optional
            Whether to save the images or not. False by default
        
        Returns
        -------
        list
            The rewards
        """
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
        """
        Save the model

        Parameters
        ----------
        path : str
            The path to save the model
        """
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
    
class A2CAgent(Agent):
    """
    Actor-Critic agent.

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
    >>> from src.agents import A2CAgent
    >>> import gym
    >>> env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='rgb_array')
    >>> env.reset()
    >>> agent = A2CAgent(num_states=3, num_actions=2, env=env)
    >>> historic_rewards, mean_rewards = agent.train(n_iterations=100, n_episodes_per_update=32, n_max_steps=100)
    """

    def __init__(self, 
        env: Env,
        num_states: int = 3,
        num_actions: int = 2, 
        gamma: float = 0.8, 
        seed: int = 42
    ) -> None:
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
        self.train_rewards = []
        self.eval_rewards = []

    def _build_model(self) -> tf.keras.models.Sequential:
        """
        Build model.

        Args:
        -------
        None

        Returns:
        --------
        model: tf.keras.models.Sequential
            Actor-critic model.
        """
        # Define input layer
        inputs = Input(shape=(self.num_states,))

        # Define shared hidden layers
        hidden1 = Dense(32, activation='relu')(inputs)
        hidden2 = Dense(32, activation='relu')(hidden1)

        # Define separate output layers
        output1 = Dense(self.num_actions)(hidden2)
        output2 = Dense(self.num_actions, activation='softmax')(hidden2)

        model = tf.keras.Model(inputs=inputs, outputs=[output2, output1])
        return model

    def _play_one_step(self, state):
        """
        Play one step.

        Args:
        -------
        obs: np.array
            Current state.

        Returns:
        --------
        next_state: np.array
            Next state.
        reward: float
            Reward.
        done: bool
            Whether episode is done.
        info: dict
            Additional information.
        action: int
            Action.
        """
        # Get the action probabilities from the policy network
        action_probs, _ = self.model.predict(np.array([state]),verbose=0)
        # Choose an action based on the action probabilities
        action = np.random.choice(self.num_actions, p=action_probs[0])
        # Take the chosen action and observe the next state and reward
        next_state, reward, done, info, _ = self.env.step(action)
        # Convert next state to list
        next_state = list(next_state)
        # Modify reward
        if reward == 0:
            reward = 0.5
        elif reward == -1:
            reward = -100
        return next_state, reward, done, info, action

    def train(self, n_episodes: int, n_max_steps: int = 10):
        """
        Train agent.

        Args:
        -------
        n_episodes: int
            Number of episodes.
        n_max_steps: int
            Maximum number of steps per episode. Default: 10.

        Returns:
        --------
        self.train_rewards: list
            Rewards for the training process
        """
        states = deque(maxlen=n_max_steps)
        actions = deque(maxlen=n_max_steps)
        rewards = deque(maxlen=n_max_steps)
        next_states = deque(maxlen=n_max_steps)
        dones = deque(maxlen=n_max_steps)
        total_rewards = []
        
        # Loop over iterations
        for episode in range(n_episodes):
            # Reset environment
            state = list(self.env.reset()[0])
            # Iterate over steps within one episode
            for step in range(n_max_steps):
                # Play one step
                next_state, reward, done, info, action = self._play_one_step(state)
                # Store the current state, action, and reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_rewards.append(reward)
                next_states.append(next_state)
                dones.append(1 if done else 0)
                state = next_state
                # End the episode if the environment is done
                if done:
                    done=False
                    break
            
            # Update the model's weights
            self._training_step(states, actions, rewards, next_states, dones)

        # Save the train rewards after converting the deque to list
        self.train_rewards = total_rewards
        return self.train_rewards

    def _training_step(self, states: deque, actions: deque, rewards: deque, next_states: deque, dones) -> None:
        """
        Perform one training step.

        Parameters:
        -----------
        states: deque
            Deque of states.
        actions: deque
            Deque of actions.
        rewards: deque
            Deque of rewards.
        next_states: deque
            Deque of next states.
        dones: deque
            Deque of done flags.

        Returns:
        --------
        None
            The model is updated in place.
        """
        # Convert the lists of states, actions, and discounted rewards to tensors
        states_ = tf.convert_to_tensor(states)
        actions_ = tf.convert_to_tensor(actions)
        rewards_ = tf.convert_to_tensor(rewards)
        next_states_ = tf.convert_to_tensor(next_states)
        
        with tf.GradientTape(persistent=True) as tape:
            # Compute the discounted rewards
            action_probs,q_values1 = self.model(states_)
            q_value1 = tf.cast(tf.gather(q_values1,actions_,axis=1,batch_dims=1),tf.float64)
            _,q_value2 = self.model(next_states_)
            q_value2 = tf.cast(tf.reduce_max(q_value2),tf.float64)
            # Compute the action probabilities
            target_value = tf.cast(rewards_,tf.float64) + (tf.cast(1,tf.float64)-tf.cast(dones,tf.float64))*self.gamma*q_value2
            value_loss = tf.reduce_mean(tf.math.pow(target_value-q_value1,2))
            advantage = target_value-q_value1
            action_probs = tf.cast(tf.math.log(tf.gather(action_probs,actions,axis=1,batch_dims=1)),tf.float64)
            # Compute the loss
            loss = -tf.reduce_mean(action_probs * advantage) + value_loss

        # Compute the gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return

    def evaluate(self, n_episodes: int, n_max_steps: int = 10) -> List:
        """
        Evaluate agent once it has been trained.

        Args:
        -------
        n_episodes: int
            Number of episodes.
        n_max_steps: int
            Maximum number of steps per episode. Default: 10.

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
                obs, reward, done, grads, _ = self._play_one_step(obs)
                # If episode is done, append final reward and break inner loop
                if done:
                    final_rewards.append(reward)
                    break
        
        return final_rewards

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