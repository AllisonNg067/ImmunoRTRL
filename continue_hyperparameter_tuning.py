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
import joblib
#initialising parameters
free = [1,1,0]
LQL = 0
activate_vd = 0
use_Markov = True
t_f1 = 0

delta_t = 0.05
# t_treat_c4 = np.zeros(3)
# t_treat_p1 = np.zeros(3)
c4 = 0
p1 = 0
PD_fractions = 1
CTLA4_fractions = 1
         #print('errors', errorMerged)
all_res_list = []
IT = (True, True)
RT_fractions = 1

file_name = 'hypoxia RT ' + str(RT_fractions) + ' PD ' + str(PD_fractions) + ' CTLA4 ' + str(CTLA4_fractions) + ' a.csv'

# paramNew = list(param)

# Generate a list of seeds

# Assuming 'params' is your list of parameters
# Load from CSV
params = pd.read_csv('new_hypoxia_parameters.csv').values.tolist()
initial_cell_count = 100000
param = params[0]
param[0] = initial_cell_count
D = [4,5]
t_rad = [10, 11]
t_treat_c4 = [10, 19]
t_treat_p1 = [10, 11]
t_f2 = max(max(t_rad[-1], t_treat_c4[-1]), t_treat_p1[-1]) + 30
t_f2 = t_rad[-1] - delta_t

def train_and_evaluate(q_network, reward_type, action_type, mode, num_episodes=1, max_steps_per_episode=26, sample_size=5, n_splits=5):
    if n_splits > sample_size:
        raise ValueError("n_splits must be less than or equal to sample size")

    total_rewards = []
    epsilon = q_network._initial_epsilon
    kf = KFold(n_splits=n_splits)
    epsilon_decays = q_network._epsilon_decays
    min_epsilon = q_network._min_epsilon
    epsilon_decay_rate =( min_epsilon/epsilon) ** (1/epsilon_decays)
    
    for train_index, test_index in kf.split(range(sample_size)):
        #print('train index', train_index)
        #print('test index', test_index)
        # Train on the training set
        for patient in train_index:
            np.random.seed(patient)
            param[0] = int(np.random.normal(loc=100000, scale=100.0, size=None))
            q_network.environment = TME(reward_type, 'DQN', action_type, param, range(10, 36), [10, 19], [10, 11], None, (-1,))
            print(q_network.environment.observe())
            epsilon = q_network._initial_epsilon
            for episode in range(num_episodes):
                state = q_network.environment.reset(mode)
                episode_reward = 0

                for step in range(max_steps_per_episode):
                    if np.random.rand() < epsilon:
                        action = np.random.randint(len(q_network.environment.action_space))
                    else:
                        state_input = np.expand_dims(state, axis=0)  # Shape (1, 3)
                        state_input = np.expand_dims(state_input, axis=1)  # Shape (1, 1, 3)
                        q_values = q_network.q_vals.predict(state_input)
                        action = np.argmax(q_values)

                    next_state, reward, done = q_network.environment.step(action)
                    q_network.store_transition(state, action, reward, next_state, done)
                    q_network.train(state, action, reward, next_state, done)

                    state = next_state
                    episode_reward += reward
                    epsilon = max(epsilon_decay_rate * epsilon, min_epsilon)

                    if done:
                        break

                print(f"Training - Patient {patient + 1}/{sample_size}, Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

        # Evaluate on the validation set
        for patient in test_index:
            np.random.seed(patient)
            param[0] = int(np.random.normal(loc=100000, scale=100.0, size=None))
            q_network.environment = TME(reward_type, 'DQN', action_type, param, range(10, 36), [10, 19], [10, 11], None, (-1,))
            print(q_network.environment.observe())
            for episode in range(num_episodes):
                state = q_network.environment.reset(mode)
                episode_reward = 0

                for step in range(max_steps_per_episode):
                    state_input = np.expand_dims(state, axis=0)  # Shape (1, 3)
                    state_input = np.expand_dims(state_input, axis=1)  # Shape (1, 1, 3)
                    q_values = q_network.q_vals.predict(state_input)
                    action = np.argmax(q_values)

                    next_state, reward, done = q_network.environment.step(action)
                    state = next_state
                    episode_reward += reward

                    if done:
                        break

                total_rewards.append(episode_reward)
                print(f"Validation - Patient {patient + 1}/{sample_size}, Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

import numpy as np
# Define the objective function for Optuna
def objective(trial, reward_type, action_type, double_Q):
    momentum = trial.suggest_float("momentum", 0.0, 0.9)
    clip_norm = trial.suggest_float("clip_norm", 0.1, 1.0)
    epsilon_decays = trial.suggest_int("epsilon_decays", 100, 200)
    min_epsilon = trial.suggest_float("min_epsilon", 0.001, 0.2)
    #list_of_buffer_capacities = list(np.linspace(2000, 10000, 81).astype(int))
    buffer_capacity = trial.suggest_categorical("buffer_capacity", [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000])
    batch_size = trial.suggest_categorical("batch_size", [64,128])
    environment = TME(reward_type, 'DQN', action_type, params[0], range(10, 36), [10, 19], [10, 11], None, (-1,))
    q_network = QNetwork(environment, double_Q = double_Q)
    q_network.set_hyperparameters(momentum=momentum, clip_norm=clip_norm, initial_epsilon=1, epsilon_decays=epsilon_decays, min_epsilon=min_epsilon, buffer_capacity=buffer_capacity, batch_size=batch_size)

    evaluation_metric = train_and_evaluate(q_network, reward_type, action_type, -1)
    return evaluation_metric

def setupAgentNetwork(env, hyperparams, double_Q):
  network = QNetwork(environment=env, batch_size=hyperparams['batch_size'], double_Q = double_Q)
  agent = NeuralAgent(env, network, replay_memory_size=hyperparams['buffer_capacity'], batch_size=hyperparams['batch_size'])
  agent.setDiscountFactor(0.95)
  agent.attach(bc.EpsilonController(initial_e=1, e_decays=hyperparams['epsilon_decays'], e_min=hyperparams['min_epsilon']))
  agent.attach(bc.LearningRateController(0.0001))
  agent.attach(bc.InterleavedTestEpochController(epoch_length=26))
  if double_Q:
    # Initialize the target network
      target_network = QNetwork(environment=env, batch_size=hyperparams['batch_size'], double_Q=True)
      target_network.q_vals.set_weights(network.q_vals.get_weights())  # Copy initial weights from the online network
      return agent, network, target_network
  else:
      return agent, network

# Create a study and optimize the objective function
reward_type = 'dose'
action_type = 'RT'
double_Q = False
study_dqn = joblib.load("study_dqn_dose.pkl")
study_dqn.optimize(lambda trial: objective(trial, reward_type, action_type, double_Q), n_trials=6)

print("Best hyperparameters: ", study_dqn.best_params)
import json

# Sample dictionary

# Save to a file
with open('optimal_hyperparameters_dqn_dose.json', 'w') as file:
    json.dump(study_dqn.best_params, file)
import joblib
# Save the study to a file
joblib.dump(study_dqn, 'study_dqn_dose.pkl')
print("Study saved to study_ddqn_killed.pkl")
