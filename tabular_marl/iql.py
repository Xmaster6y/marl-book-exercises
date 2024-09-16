from collections import defaultdict
import random
from typing import List, DefaultDict

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class IQL:
    """Agent using the Independent Q-Learning algorithm"""

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        **kwargs,
    ):
        """Constructor of IQL

        Initializes variables for independent Q-learning agents

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr n_acts (List[int]): number of actions for each agent
        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents
        """
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        # access value of Q_i(o, a) with self.q_tables[i][str((o, a))] (str conversion for hashable obs)
        self.q_tables: List[DefaultDict] = [
            defaultdict(lambda: 0) for _ in range(self.num_agents)
        ]

    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **IMPLEMENT THIS FUNCTION**

        :param obss (List): list of observations for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []
        for i, obs in enumerate(obss):
            if random.random() < self.epsilon:
                actions.append(random.randint(0, self.n_acts[i] - 1))
            else:
                # Get the best action for the agent
                best_action = None
                best_q_value = float("-inf")
                for act in range(self.n_acts[i]):
                    q_value = self.q_tables[i][str((obs, act))]
                    if q_value > best_q_value:
                        best_action = act
                        best_q_value = q_value
                actions.append(best_action)
        return actions

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """Updates the Q-tables based on agents' experience

        **IMPLEMENT THIS FUNCTION**

        :param obss (List[np.ndarray]): list of observations for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray]): list of observations after taking the action for each agent
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (List[float]): updated Q-values for current actions of each agent
        """
        for i, (obs, act, reward, n_obs) in enumerate(zip(obss, actions, rewards, n_obss)):
            # Get the best action for the agent
            best_q_value = float("-inf")
            for a in range(self.n_acts[i]):
                q_value = self.q_tables[i][str((n_obs, a))]
                if q_value > best_q_value:
                    best_q_value = q_value
            # Update the Q-value for the action
            target = reward + self.gamma * best_q_value
            self.q_tables[i][str((obs, act))] += self.learning_rate * (
                target - self.q_tables[i][str((obs, act))]
            )
        return [self.q_tables[i][str((obs, act))] for i, (obs, act) in enumerate(zip(obss, actions))]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        epsilon_min = 0.0
        epsilon_max = 1
        self.epsilon = epsilon_min + 0.5 *(epsilon_max - epsilon_min) * (1 + np.cos(np.pi * timestep / max_timestep))
        lr_min = 0.01
        lr_max = 0.05
        self.learning_rate = lr_min + 0.5 *(lr_max - lr_min) * (1 + np.cos(np.pi * timestep / max_timestep))
