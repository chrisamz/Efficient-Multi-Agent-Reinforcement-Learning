# multi_agent_rl_algorithms.py

"""
Multi-Agent Reinforcement Learning Algorithms for Efficient Multi-Agent Reinforcement Learning

This module contains functions for developing and training reinforcement learning algorithms for multi-agent systems.

Techniques Used:
- Q-learning
- Deep Q-Networks (DQN)
- Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

Libraries/Tools:
- tensorflow
- keras
- numpy
- gym
- pettingzoo

"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from pettingzoo.mpe import simple_spread_v2

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Initialize the DQNAgent class.
        
        :param state_size: int, size of the state space
        :param action_size: int, size of the action space
        :param learning_rate: float, learning rate for the optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the neural network model for the agent.
        
        :return: Model, compiled Keras model
        """
        inputs = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        """
        Choose an action based on the current state.
        
        :param state: np.array, current state
        :return: int, chosen action
        """
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        """
        Train the agent on a single step.
        
        :param state: np.array, current state
        :param action: int, action taken
        :param reward: float, reward received
        :param next_state: np.array, next state
        :param done: bool, whether the episode is done
        """
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        target = reward
        if not done:
            target = reward + 0.99 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

class MADDPGAgent:
    def __init__(self, state_size, action_size, num_agents, actor_lr=0.001, critic_lr=0.002):
        """
        Initialize the MADDPGAgent class.
        
        :param state_size: int, size of the state space
        :param action_size: int, size of the action space
        :param num_agents: int, number of agents
        :param actor_lr: float, learning rate for the actor network
        :param critic_lr: float, learning rate for the critic network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actors = [self._build_actor() for _ in range(num_agents)]
        self.critics = [self._build_critic() for _ in range(num_agents)]
        self.target_actors = [self._build_actor() for _ in range(num_agents)]
        self.target_critics = [self._build_critic() for _ in range(num_agents)]

    def _build_actor(self):
        """
        Build the actor network.
        
        :return: Model, compiled Keras model
        """
        inputs = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(self.action_size, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.actor_lr), loss='mse')
        return model

    def _build_critic(self):
        """
        Build the critic network.
        
        :return: Model, compiled Keras model
        """
        inputs = Input(shape=(self.state_size * self.num_agents + self.action_size * self.num_agents,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mse')
        return model

    def act(self, states):
        """
        Choose actions for each agent based on the current states.
        
        :param states: list of np.array, current states for each agent
        :return: list of int, chosen actions for each agent
        """
        actions = []
        for i, state in enumerate(states):
            state = np.reshape(state, [1, self.state_size])
            action_prob = self.actors[i].predict(state)
            action = np.argmax(action_prob[0])
            actions.append(action)
        return actions

    def train(self, experiences):
        """
        Train the agents on a batch of experiences.
        
        :param experiences: list of tuples, (states, actions, rewards, next_states, dones)
        """
        for i, experience in enumerate(experiences):
            states, actions, rewards, next_states, dones = experience
            state = np.reshape(states[i], [1, self.state_size])
            next_state = np.reshape(next_states[i], [1, self.state_size])
            action = np.zeros([1, self.action_size])
            action[0][actions[i]] = 1

            target_action = self.target_actors[i].predict(next_state)
            target_action = np.argmax(target_action, axis=1)
            target_action = np.reshape(target_action, [1, self.action_size])
            target_input = np.concatenate([next_state.flatten(), target_action.flatten()])
            target_input = np.reshape(target_input, [1, -1])
            target_q = rewards[i] + 0.99 * self.target_critics[i].predict(target_input)

            critic_input = np.concatenate([state.flatten(), action.flatten()])
            critic_input = np.reshape(critic_input, [1, -1])
            self.critics[i].fit(critic_input, target_q, epochs=1, verbose=0)

            actor_input = state
            actions_pred = self.actors[i].predict(actor_input)
            action_onehot = np.zeros(self.action_size)
            action_onehot[np.argmax(actions_pred)] = 1
            actor_input = np.concatenate([state.flatten(), action_onehot])
            actor_input = np.reshape(actor_input, [1, -1])
            action_gradient = np.array(self.critics[i].predict(actor_input))
            self.actors[i].fit(state, actions_pred, sample_weight=action_gradient.flatten(), epochs=1, verbose=0)

if __name__ == "__main__":
    env = simple_spread_v2.env()
    num_agents = env.num_agents
    state_size = env.observation_space(env.agents[0]).shape[0]
    action_size = env.action_space(env.agents[0]).n

    episodes = 1000
    max_steps = 200

    agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
    for episode in range(episodes):
        observations = env.reset()
        for step in range(max_steps):
            actions = [agent.act(observation) for agent, observation in zip(agents, observations)]
            next_observations, rewards, dones, _ = env.step(actions)
            for i, agent in enumerate(agents):
                agent.train(observations[i], actions[i], rewards[i], next_observations[i], dones[i])
            observations = next_observations
            if all(dones):
                break
        print(f"Episode {episode+1}/{episodes} completed.")
    print("Training completed.")
