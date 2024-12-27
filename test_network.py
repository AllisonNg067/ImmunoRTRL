import pandas as pd
from differential_equations_hypoxia_advanced import radioimmuno_response_model
import numpy as np
import matplotlib.pyplot as plt
from TMEClass import TME
from DeepRL import ReplayBuffer, QNetwork
import optuna
import collections
import random
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from deer.learning_algos.q_net_keras import MyQNetwork
from tensorflow.keras import backend as K  # Correct import for Keras backend
import keras.backend as K
import tensorflow as tf
from deer.agent import NeuralAgent
import deer.experiment.base_controllers as bc
from sklearn.model_selection import KFold

params = pd.read_csv('new_hypoxia_parameters.csv').values.tolist()
initial_cell_count = 100000
param = params[0]
param[0] = initial_cell_count
reward_type = 'killed'
action_type = 'RT'

def test_network(network, reward_type, action_type, env, sample_size=2):
    # Reset the environment to testing mode
    env.reset(1)

    # Manually run the testing loop
    env.mode = 1  # Set mode to testing
    total_reward = 0
    doses = []
    final_rewards = []
    counter = 1
    #agent.run(n_epochs = 1, epoch_length = 26)
    for patient in range(30, 30 + sample_size):
        env = TME(reward_type, 'DQN', action_type, params[patient], range(10, 36), [11, 13], [10, 14], None, 1)
        schedule = []
        for _ in range(26):  # Run for 26 steps
            state = env.observe()
        #state_input = np.expand_dims(state, axis=0)  # Add batch dimension
            q_values = network.q_vals.predict(state)
            print(q_values)
            action_index = np.argmax(q_values)  # Select the action with the highest Q-value
            schedule.append(env.action_space[action_index])
            state, reward, done = env.act(action_index)
            total_reward += reward
            if done:
                final_rewards.append(reward)
                doses.append(schedule)
                plt.plot(schedule)
                plt.xlabel('RT Fraction Number')
                plt.ylabel('Dose (Gy)')
                plt.title('Optimal Dose per Fraction')
                plt.savefig('optimal dose per fraction ' + action_type + reward_type + ' ' + str(counter) + '.png')
                plt.show()
                plt.close()
                break

    print(f'Total reward during testing: {total_reward}')
    print(doses)
    print(final_rewards)

def setupAgentNetwork(env, hyperparams, double_Q):
  network = QNetwork(environment=env, batch_size=hyperparams['batch_size'], double_Q = double_Q)
  agent = NeuralAgent(env, network, replay_memory_size=hyperparams['buffer_capacity'], batch_size=hyperparams['batch_size'])
  agent.setDiscountFactor(0.95)
  agent.attach(bc.EpsilonController(initial_e=1, e_decays=hyperparams['epsilon_decays'], e_min=hyperparams['min_epsilon']))
  agent.attach(bc.LearningRateController(0.001))
  agent.attach(bc.InterleavedTestEpochController(epoch_length=26))
  if double_Q:
    # Initialize the target network
      target_network = QNetwork(environment=env, batch_size=hyperparams['batch_size'], double_Q=True)
      target_network.q_vals.set_weights(network.q_vals.get_weights())  # Copy initial weights from the online network
      return agent, network, target_network
  else:
      return agent, network

# Define your environment class (assuming TME is already defined)
env = TME(reward_type, 'DQN', action_type, params[40], range(10, 36), [11, 13], [10, 14], None, 1)
import json

# Read from a file
with open('optimal_hyperparameters.json', 'r') as file:
    hyperparams = json.load(file)

# Initialize the agent with the environment and the network
agent, network = setupAgentNetwork(env, hyperparams, False)
# Load the trained weights
network.q_vals.load_weights('dqn_killed.weights.keras')

test_network(network, reward_type, action_type, env)
