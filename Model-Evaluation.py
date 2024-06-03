# model_evaluation.py

"""
Model Evaluation Module for Efficient Multi-Agent Reinforcement Learning

This module contains functions for evaluating the performance of multi-agent reinforcement learning algorithms.

Techniques Used:
- Average reward
- Success rate
- Cooperation metrics

Libraries/Tools:
- numpy
- pandas
- matplotlib
- gym
- pettingzoo

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v2

class ModelEvaluation:
    def __init__(self, env_name='simple_spread_v2', episodes=100, max_steps=200, model_dir='models/', results_dir='results/'):
        """
        Initialize the ModelEvaluation class.
        
        :param env_name: str, name of the multi-agent environment
        :param episodes: int, number of episodes to evaluate
        :param max_steps: int, maximum steps per episode
        :param model_dir: str, directory where models are saved
        :param results_dir: str, directory to save evaluation results
        """
        self.env_name = env_name
        self.episodes = episodes
        self.max_steps = max_steps
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.env = simple_spread_v2.env()
        self.num_agents = self.env.num_agents
        self.state_size = self.env.observation_space(self.env.agents[0]).shape[0]
        self.action_size = self.env.action_space(self.env.agents[0]).n
        self.agents = self.load_agents()

    def load_agents(self):
        """
        Load trained agents from the model directory.
        
        :return: list of agents
        """
        agents = []
        for agent_idx in range(self.num_agents):
            model_path = os.path.join(self.model_dir, f'agent_{agent_idx}.h5')
            agent = self.load_model(model_path)
            agents.append(agent)
        return agents

    def load_model(self, model_path):
        """
        Load a trained model from a given path.
        
        :param model_path: str, path to the model
        :return: loaded model
        """
        model = tf.keras.models.load_model(model_path)
        return model

    def evaluate(self):
        """
        Evaluate the performance of the trained models.
        
        :return: DataFrame, evaluation results
        """
        results = []
        for episode in range(self.episodes):
            observations = self.env.reset()
            episode_rewards = np.zeros(self.num_agents)
            for step in range(self.max_steps):
                actions = [np.argmax(agent.predict(np.reshape(observation, [1, self.state_size]))[0]) for agent, observation in zip(self.agents, observations)]
                next_observations, rewards, dones, _ = self.env.step(actions)
                episode_rewards += rewards
                observations = next_observations
                if all(dones):
                    break
            results.append({
                'episode': episode,
                'rewards': episode_rewards
            })
        results_df = pd.DataFrame(results)
        return results_df

    def calculate_metrics(self, results_df):
        """
        Calculate evaluation metrics from the results DataFrame.
        
        :param results_df: DataFrame, evaluation results
        :return: dict, calculated metrics
        """
        average_rewards = results_df['rewards'].mean()
        success_rate = (results_df['rewards'].sum(axis=1) > 0).mean()
        metrics = {
            'average_rewards': average_rewards,
            'success_rate': success_rate
        }
        return metrics

    def save_results(self, results_df, metrics):
        """
        Save the evaluation results and metrics to the results directory.
        
        :param results_df: DataFrame, evaluation results
        :param metrics: dict, calculated metrics
        """
        os.makedirs(self.results_dir, exist_ok=True)
        results_path = os.path.join(self.results_dir, 'evaluation_results.csv')
        metrics_path = os.path.join(self.results_dir, 'evaluation_metrics.txt')
        results_df.to_csv(results_path, index=False)
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"Results saved to {self.results_dir}")

    def plot_results(self, results_df):
        """
        Plot the evaluation results.
        
        :param results_df: DataFrame, evaluation results
        """
        plt.figure(figsize=(10, 6))
        for agent_idx in range(self.num_agents):
            plt.plot(results_df['episode'], results_df['rewards'].apply(lambda x: x[agent_idx]), label=f'Agent {agent_idx}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Evaluation Results')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, 'evaluation_results.png'))
        plt.show()

if __name__ == "__main__":
    evaluator = ModelEvaluation(env_name='simple_spread_v2', episodes=100, max_steps=200, model_dir='models/', results_dir='results/')

    # Evaluate the models
    results_df = evaluator.evaluate()

    # Calculate metrics
    metrics = evaluator.calculate_metrics(results_df)

    # Save results and metrics
    evaluator.save_results(results_df, metrics)

    # Plot results
    evaluator.plot_results(results_df)
    print("Model evaluation completed and results saved.")
