# deployment.py

"""
Deployment Module for Efficient Multi-Agent Reinforcement Learning

This module contains functions for deploying trained multi-agent reinforcement learning models for real-world applications or further simulation testing.

Techniques Used:
- Model loading
- Real-time decision making

Libraries/Tools:
- numpy
- tensorflow
- flask
- gunicorn
- Docker

"""

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from pettingzoo.mpe import simple_spread_v2

app = Flask(__name__)

# Load the trained models
model_dir = 'models/'
num_agents = len([name for name in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, name))])
state_size = simple_spread_v2.env().observation_space(simple_spread_v2.env().agents[0]).shape[0]
action_size = simple_spread_v2.env().action_space(simple_spread_v2.env().agents[0]).n
agents = [tf.keras.models.load_model(os.path.join(model_dir, f'agent_{i}.h5')) for i in range(num_agents)]

def get_action(observation, agent):
    """
    Get the action for a given observation using the agent's model.
    
    :param observation: np.array, current state
    :param agent: trained model of the agent
    :return: int, chosen action
    """
    observation = np.reshape(observation, [1, state_size])
    q_values = agent.predict(observation)
    return np.argmax(q_values[0])

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to get actions for the current states of all agents.
    
    :return: JSON response with the actions for all agents
    """
    data = request.json
    if 'observations' not in data:
        return jsonify({'error': 'No observations provided'}), 400
    
    observations = data['observations']
    actions = [get_action(observation, agent) for observation, agent in zip(observations, agents)]
    
    return jsonify({'actions': actions})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
