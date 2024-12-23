import pandas as pd
from differential_equations_hypoxia_advanced import radioimmuno_response_model
import numpy as np
import matplotlib.pyplot as plt
class TME():
    def __init__(self, reward_type, action_type, treatment_to_optimise, parameters, t_rad_plan, t_treat_c4, t_treat_p1, agent, mode):
        #initialising the class
        self.reward_type = reward_type
        self.action_type = action_type
        self.parameters = parameters
        self.t_rad_plan = t_rad_plan
        self.t_treat_c4 = t_treat_c4
        self.t_treat_p1 = t_treat_p1
        self.treatment_to_optimise = treatment_to_optimise
        if self.treatment_to_optimise == 'RT':
            self.total_fractions = len(self.t_rad_plan)
        elif self.treatment_to_optimise == 'anti-PD-1':
            self.total_fractions = len(self.t_treat_p1)
        else:
            self.total_fractions = len(self.t_treat_c4)
        #initialise dose per fraction of IO
        self.parameters[24] = 0.2 / len(self.t_treat_c4)
        self.parameters[37] = 0.6 / len(self.t_treat_p1)
        #initialise dose vector
        if self.treatment_to_optimise == 'RT':
            self.D = [1 for x in range(len(self.t_rad_plan))]
            self.action_space = np.array(range(10, 30)) / 10
        else:
            # optimum RT dose fractionation obtained should be used
            self.D = [2 for x in range(len(self.t_rad_plan))]
            if self.treatment_to_optimise == 'anti-PD-1':
                pass
        self.cumulative_dose = 0
        self.dose_fractions = 0
        self.state = np.array([100000, 0, 0])  # C_N, C_H, D_tot
        self.action_space = np.array(range(10, 30)) / 10  # Doses from 1.0 to 2.9 Gy
        self.epsilon = 0.1  # Initial epsilon value for exploration
        self.mode = mode  # -1 for training, 1 for testing
        self.agent = agent  # Store the agent instance

    def reset(self, mode):
        #reset the agent to starting state
        self.state = np.array([100000, 0, 0])
        if self.treatment_to_optimise == 'RT':
            #initialise RT dose
            self.D = [1 for x in range(len(self.t_rad_plan))]
        else:
            # optimum RT dose fractionation obtained should be used
            self.D = [2 for x in range(len(self.t_rad_plan))]
        self.cumulative_dose = 0
        self.dose_fractions = 0
        self.mode = mode
        return self.state

    def inputDimensions(self):
        #dimensions of the state vector
        return [(1, 3)]

    def nActions(self):
        return len(self.action_space)  # Number of possible actions

    # DO NOT REMOVE act() or step() FUNCTIONS - IT BREAKS THE SYSTEM
    def act(self, action_index):
        # select dose to be administered
        action = self.action_space[action_index]
        if self.treatment_to_optimise == 'RT':
            #if RT is the treatment to be optimised,
            self.D[self.dose_fractions] = action
            if self.dose_fractions == self.total_fractions - 1:
                # if it is the final dose fraction, monitor tumour for 31 days post final fraction
                self.t_f2 = self.t_rad_plan[-1] + 31
            else:
                # otherwise, monitor the tumour up to just before next dose fraction
                self.t_f2 = self.t_rad_plan[self.dose_fractions + 1] + 0.95
        self.cumulative_dose += action

        self.dose_fractions += 1
        # model tumour microenvironment
        _, _, _, _, self.time, self.C_tot, self.C, _, _, _, self.C_N, self.C_H, *_ = radioimmuno_response_model(self.parameters,
                                                                                                                0.05,
                                                                                                                [1, 1,
                                                                                                                 0], 0,
                                                                                                                self.t_f2,
                                                                                                                self.D,
                                                                                                                self.t_rad_plan,
                                                                                                                self.t_treat_c4,
                                                                                                                self.t_treat_p1,
                                                                                                                0, 0,
                                                                                                                True)
        # calculate reward, state, and whether state is terminal following treatment
        self.reward = self.reward_function()
        self.state = np.array([self.C_N[0][-1], self.C_H[0][-1], self.cumulative_dose]).reshape(1, 3)
        self.done = self.inTerminalState()
        return self.state, self.reward, self.done

    def step(self, action_index):
        # select dose to be administered
        action = self.action_space[action_index]
        if self.treatment_to_optimise == 'RT':
            # if RT is the treatment to be optimised,
            self.D[self.dose_fractions] = action
            if self.dose_fractions == self.total_fractions - 1:
                # if it is the final dose fraction, monitor tumour for 31 days post final fraction
                self.t_f2 = self.t_rad_plan[-1] + 31
            else:
                # otherwise, monitor the tumour up to just before next dose fraction
                self.t_f2 = self.t_rad_plan[self.dose_fractions + 1] + 0.95
        self.cumulative_dose += action
        self.dose_fractions += 1
        # model tumour microenvironment
        _, _, _, _, self.time, self.C_tot, self.C, _, _, _, self.C_N, self.C_H, *_ = radioimmuno_response_model(self.parameters,
                                                                                                                0.05,
                                                                                                                [1, 1,
                                                                                                                 0], 0,
                                                                                                                self.t_f2,
                                                                                                                self.D,
                                                                                                                self.t_rad_plan,
                                                                                                                self.t_treat_c4,
                                                                                                                self.t_treat_p1,
                                                                                                                0, 0,
                                                                                                                True)
        # calculate reward, state, and whether state is terminal following treatment
        self.reward = self.reward_function()
        self.state = np.array([self.C_N[0][-1], self.C_H[0][-1], self.cumulative_dose])
        self.done = self.inTerminalState()
        return self.state, self.reward, self.done

    def visualise(self, total=False):
        # visualises total tumour cell count if total is True, otherwise only active cell count is visualised
        if total:
            plt.plot(self.time, self.C_tot[0])
        else:
            plt.plot(self.time, self.C[0])
        plt.show()

    def reward_function(self):
        if self.treatment_to_optimise == 'RT':
            #checks if RT is the treatment to be optimised
            if self.dose_fractions == len(self.t_rad_plan):
            # if this is the final treatment fraction, obtain indices of tumour volumes within 31 days following last treatment
                indices = np.where((self.time >= self.t_rad_plan[self.dose_fractions - 1]) & (self.time <= self.t_rad_plan[-1] + 31))[0]
            else:
            # otherwise obtain indices of tumour volumes between current treatment and next treatment
                indices = np.where((self.time >= self.t_rad_plan[self.dose_fractions - 1]) & (
                        self.time <= self.t_rad_plan[self.dose_fractions] - 0.05))
        # extract all active tumour cell counts according to indices
        self.C_trimmed = self.C[0][indices]
        if self.treatment_to_optimise == 'RT':
            if self.dose_fractions == 1:
            # if this is the first dose fraction, calculate normalisation factor so the first reward due to cell killing is -1 (dose reward function)
                self.kill_normalisation = 1 / np.min(self.C_trimmed)
            if self.reward_type == 'killed':
            # reward only accounts for active tumour cell count. Lower active tumour cell counts are favourable
                reward = -1 * np.min(self.C_trimmed)
            elif self.reward_type == 'dose':
            # reward accounts for total dose and active tumour cell count. Lower active tumour cell counts and total dose are favourable
                reward = -1 * self.kill_normalisation * np.min(self.C_trimmed) - self.cumulative_dose / (
                        3 * self.dose_fractions)
        return reward

    def inTerminalState(self):
        if self.C[0][-1] == 0:
            # state is terminal if all active tumour cells are killed
            return True
        elif self.dose_fractions == self.total_fractions:
            # or the number of dose fractions is the length of t_rad_plan (number of dose fractions fixed to be below a certain number)
            return True
        else:
            return False

    def observe(self):
        return np.array([self.state.reshape(1, 3)])  # Reshape to (1, 3) and wrap in an array

    def observationType(self, index):
        return np.float32  # or the appropriate dtype for your observations

    def end(self):
        print("End of epoch. Performing cleanup if necessary.")
