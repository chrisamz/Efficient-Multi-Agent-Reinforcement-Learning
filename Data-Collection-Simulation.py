# data_collection_simulation.py

"""
Data Collection and Simulation Module for Efficient Multi-Agent Reinforcement Learning

This module contains functions for setting up environments and simulations to generate data for training and evaluating multi-agent systems.

Techniques Used:
- Environment setup
- Data logging
- Simulation

Libraries/Tools:
- numpy
- gym
- pettingzoo

"""

import os
import numpy as np
import gym
from pettingzoo.mpe import simple_spread_v2
from gym import spaces
import pandas as pd

class DataCollectionSimulation:
    def __init__(self, env_name='simple_spread_v2', episodes=1000, max_steps=200, save_dir='data/'):
        """
        Initialize the DataCollectionSimulation class.
        
        :param env_name: str, name of the multi-agent environment
        :param episodes: int, number of episodes to simulate
        :param max_steps: int, maximum steps per episode
        :param save_dir: str, directory to save the collected data
        """
        self.env_name = env_name
        self.episodes = episodes
        self.max_steps = max_steps
        self.save_dir = save_dir
        self.env = simple_spread_v2.env()

    def reset(self):
        """
        Reset the environment to its initial state.
        
        :return: list, initial observations
        """
        self.env.reset()
        observations = [self.env.observe(agent) for agent in self.env.agents]
        return observations

    def step(self, actions):
        """
        Take a step in the environment with the given actions.
        
        :param actions: list, actions for each agent
        :return: tuple, (observations, rewards, dones, infos)
        """
        for agent, action in zip(self.env.agents, actions):
            self.env.step(action)
        observations = [self.env.observe(agent) for agent in self.env.agents]
        rewards = [self.env.rewards[agent] for agent in self.env.agents]
        dones = [self.env.dones[agent] for agent in self.env.agents]
        infos = [self.env.infos[agent] for agent in self.env.agents]
        return observations, rewards, dones, infos

    def run_simulation(self):
        """
        Run the simulation to generate data.
        
        :return: DataFrame, collected data
        """
        data = []
        for episode in range(self.episodes):
            observations = self.reset()
            for step in range(self.max_steps):
                actions = [self.env.action_space(agent).sample() for agent in self.env.agents]
                next_observations, rewards, dones, infos = self.step(actions)
                for agent_idx, agent in enumerate(self.env.agents):
                    data.append({
                        'episode': episode,
                        'step': step,
                        'agent': agent,
                        'observation': observations[agent_idx],
                        'action': actions[agent_idx],
                        'reward': rewards[agent_idx],
                        'next_observation': next_observations[agent_idx],
                        'done': dones[agent_idx],
                        'info': infos[agent_idx]
                    })
                observations = next_observations
                if all(dones):
                    break
        data_df = pd.DataFrame(data)
        return data_df

    def save_data(self, data_df):
        """
        Save the collected data to a CSV file.
        
        :param data_df: DataFrame, collected data
        """
        os.makedirs(self.save_dir, exist_ok=True)
        data_filepath = os.path.join(self.save_dir, 'simulation_data.csv')
        data_df.to_csv(data_filepath, index=False)
        print(f"Data saved to {data_filepath}")

if __name__ == "__main__":
    simulation = DataCollectionSimulation(env_name='simple_spread_v2', episodes=1000, max_steps=200, save_dir='data/')

    # Run the simulation
    data_df = simulation.run_simulation()

    # Save the collected data
    simulation.save_data(data_df)
    print("Data collection and simulation completed.")
