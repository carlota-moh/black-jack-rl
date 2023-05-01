from abc import abstractclassmethod
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import numpy as np
import random
from collections import deque
import tensorflow as tf

class Agent:
    def __init__(self, env) -> None:
        self.env = env

    @abstractclassmethod
    def act(self):
        """ Defines how the agent takes actions """
        pass
    @abstractclassmethod
    def evaluate(self):
        """ Defines how the agent is to be evaluated"""
    
    def plot_performance(self, rewards):
        plt.figure(figsize=(8, 4))
        plt.plot(rewards)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of rewards", fontsize=14)
        plt.show()

    def plot_wins(self, rewards):
        reward_counts = [rewards.count(i) for i in [-100, 0.5, 1]]
        plt.figure(figsize=(8, 4))
        plt.bar(x=["lose", 'draw', 'win'],height=reward_counts)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of rewards", fontsize=14)
        plt.show()


class RandomAgent(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.eval_rewards = []

    def act(self):
        action = self.env.action_space.sample()
        return action

    def evaluate(self, n_episodes=1000):
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
        
class DQNAgent(Agent):
    def __init__(self, env, num_states, num_actions, batch_size=200, gamma=0.8) -> None:
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
       
    def act(self, state, epsilon):
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
    
    def play_one_step(self, state, epsilon):
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

    def train(self, n_episodes=1000):
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
    
    def evaluate(self, n_episodes=1000):
        for episode in tqdm(range(n_episodes)):
            obs = self.env.reset()[0]
            done = False
            while not done:
                action = self.act(obs, epsilon=0)
                obs, reward, done, info, _ = self.env.step(action)

            if reward == 0:
                reward += 0.5
            
            elif reward == -1:
                reward = -100
                
            self.eval_rewards.append(reward)
        
        return self.eval_rewards
    
    def save(self, path):
        self.model.save(path)


    