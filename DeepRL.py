#importing relevant modules
import optuna
import collections
import numpy as np
import random
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from deer.learning_algos.q_net_keras import MyQNetwork
from tensorflow.keras import backend as K  # Correct import for Keras backend
class ReplayBuffer:
    def __init__(self, capacity):
        #the buffer is a double ended queue with fixed capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        #add to the buffer (state, action, reward, next_state, done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        #take a random sample of tuples from buffer of a given batch size
        batch = random.sample(self.buffer, batch_size)
        #collects all the states, actions, rewards, next states, and dones in the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def size(self):
        #return the length of the buffer
        return len(self.buffer)

class QNetwork(MyQNetwork):
    def __init__(self, environment, *args, **kwargs):
        super(QNetwork, self).__init__(environment, *args, **kwargs)
        self.environment = environment

        self.gamma = 0.95  # Discount factor
        self._compile()

    def _compile(self):
        #initialise the optimiser to be an Adam optimiser
        optimizer = Adam(beta_1=self._momentum, beta_2=self._rho, epsilon=self._rms_epsilon, clipnorm=self._clip_norm, learning_rate = 0.0001)
        self.q_vals.compile(optimizer=optimizer, loss='mse')

    def set_hyperparameters(self, momentum, clip_norm, initial_epsilon, epsilon_decays, min_epsilon, buffer_capacity, batch_size):
        #setting values of hyperparameters
        self._momentum = momentum
        self._clip_norm = clip_norm
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size  # Adjust batch size as needed
        self._initial_epsilon = initial_epsilon
        self._epsilon_decays = epsilon_decays
        self._min_epsilon = min_epsilon
        self._compile()

    def _resetQHat(self):
        for i, (param, next_param) in enumerate(zip(self.params, self.next_params)):
            #loop over pairs of parameters and target parameters
            #target parameters are modified to match current parameters
            next_param.assign(param)
        self._compile()  # recompile to take into account new optimizer parameters that may have changed since

    def store_transition(self, state, action, reward, next_state, done):
        #stores a transition to the replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self, state, action, reward, next_state, done):
        if self.replay_buffer.size() < self.batch_size:
            return  # Not enough samples to train
        #sample experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to numpy arrays and reshape to match the expected input shape
        states = np.array(states).reshape(self.batch_size, 1, -1)
        next_states = np.array(next_states).reshape(self.batch_size, 1, -1)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Predict Q-values for current states
        q_values = self.q_vals.predict(states)

        # Predict Q-values for next states
        next_q_values = self.q_vals.predict(next_states)

        # Update Q-values
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q_values[i])
            q_values[i][actions[i]] = target

        # Train the Q-network
        self.q_vals.fit(states, q_values, epochs=1, verbose=0)
