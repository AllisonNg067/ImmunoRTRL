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

def trainNetwork(network, reward_type, action_type, env, sample_size, double_Q, update_target_frequency=1, num_epochs=1):
  # List to store Q-values for visualization
  q_values_list = []
  errors_list = []
  relative_errors_list = []

  # Training loop with Q-value visualization
  env.mode = (-1,)  # Set mode to training
  for epoch in range(num_epochs):  # Run for 5 epochs
      for patient in range(sample_size):
          env = TME(reward_type, 'DQN', action_type, params[patient], range(10, 36), [11, 13], [10, 14], None, (-1,))
          for step in range(26):  # Run for 26 steps per epoch
              print(step)
              state = env.observe()
              state_input = np.expand_dims(state, axis=0)  # Add batch dimension
              q_values = network.q_vals.predict(state)
              q_values_list.append(np.max(q_values))  # Store the maximum Q-value for visualization

              action_index = np.argmax(q_values)  # Select the action with the highest Q-value
              next_state, reward, done = env.act(action_index)

              # Compute the target Q-value using the target network
              next_state_input = np.expand_dims(next_state, axis=0)
              if double_Q:
                  next_q_values = target_network.q_vals.predict(next_state_input)
              else:
                  next_q_values = network.q_vals.predict(next_state_input)
              target_q_values = q_values.copy()
              if done:
                  target_q_values[0, action_index] = reward
              else:
                  target_q_values[0, action_index] = reward + agent.discountFactor() * np.max(next_q_values)

              # Calculate Q-value error
              q_value_error = np.abs(target_q_values - q_values)
              non_zero_error = q_value_error[q_value_error != 0]
              if non_zero_error.size > 0:
                  errors_list.append(non_zero_error[0])  # Store the non-zero Q-value error

              #print(np.abs(target_q_values - q_values)[np.abs(target_q_values - q_values) != 0])
              if np.all(np.abs(target_q_values - q_values) == 0.0):
                  errors_list.append(0)
                  relative_errors_list.append(0)
              else:
                  errors_list.append(np.abs(target_q_values - q_values)[np.abs(target_q_values - q_values) != 0][0])
                  if np.abs(target_q_values).all() != 0:
                      relative_errors_list.append((np.abs(target_q_values - q_values)/(np.abs(target_q_values) + 10 **-10))[np.abs(target_q_values - q_values) != 0][0])
                  if relative_errors_list[-1] > 10**2:
                      print(target_q_values)
                      print(q_values)
              # Update the Q-network
              network.q_vals.train_on_batch(state, target_q_values)

              # Replace inf values with 0 in relative errors list
              relative_errors_list = [0 if np.isinf(x) else x for x in relative_errors_list]

              if done:
                  env.reset((-1,))
                  break

              if double_Q and step % update_target_frequency == 0:
                # Periodically update the target network
                  target_network.q_vals.set_weights(network.q_vals.get_weights())
  if double_Q:
      network.q_vals.save('ddqn_killed.weights.keras')
      print("Trained Q-network weights saved successfully.")
  else:
      network.q_vals.save('dqn_killed.weights.keras')
      print("Trained Q-network weights saved successfully.")
  # Plot the Q-values
  plt.plot(q_values_list)
  plt.xlabel('Training Steps')
  plt.ylabel('Max Q-value')
  plt.title('Max Q-values over Training Steps')
  plt.savefig('max q values.png')
  plt.show()
  print(errors_list)
  # Plot the Q-values
  plt.plot(errors_list)
  plt.xlabel('Training Steps')
  plt.ylabel('Error')
  plt.title('Error in Q-Values over Training Steps')
  plt.savefig('error.png')
  plt.show()
  plt.plot(relative_errors_list)
  plt.xlabel('Training Steps')
  plt.ylabel('Relative Error')
  plt.title('Relative Error in Q-Values over Training Steps')
  plt.savefig('relative error.png')
  plt.show()
reward_type = 'killed'
# Initialize the environment
action_type = 'RT'
env = TME(reward_type, 'DQN', action_type, param, range(10, 36), [11, 13], [10, 14], None, (-1,))
env.reset((-1,))

import json

# Read from a file
with open('optimal_hyperparameters.json', 'r') as file:
    hyperparams = json.load(file)

#print(data)
# Initialize the agent
agent, network = setupAgentNetwork(env, hyperparams, False)
sample_size = 4
trainNetwork(network, reward_type, action_type, env, sample_size, False, num_epochs=1)
