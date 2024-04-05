import pfrl
import torch
from torch import nn
import gym
import numpy as np
import matplotlib.pyplot as plt
from pfrl.nn import ConcatObsAndAction
from pfrl.policies import SoftmaxCategoricalHead
from pfrl import experiments, explorers, replay_buffers, utils
from torch.nn import LeakyReLU, Linear, BatchNorm1d, Dropout
from example import ddpg

class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        hidden_layer_n = 100
        self.l1 = torch.nn.Linear(obs_size, hidden_layer_n)
        self.l2 = torch.nn.Linear(hidden_layer_n, hidden_layer_n)
        self.l3 = torch.nn.Linear(hidden_layer_n, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)

def policy(obs, env): 
    obs_size = env.observation_space[0].shape[0] # map_aoba01 has 18 nodes, current-position(18dimensions)+current-goal(18dimensions)=36
    n_actions = env.action_space[0].n # map_aoba01 has 18 nodes and each node corresponds to one action

    actor_func = nn.Sequential(
        Linear(obs_size, 256),  # Increase number of units
        BatchNorm1d(256),       # Batch normalization
        LeakyReLU(),            # Leaky ReLU activation
        Linear(256, 128),       # Increase number of units
        BatchNorm1d(128),       # Batch normalization
        LeakyReLU(),            # Leaky ReLU activation
        Linear(128, 64), # Output layer
        BatchNorm1d(64),       # Batch normalization
        LeakyReLU(),
        Linear(64, n_actions),        
        SoftmaxCategoricalHead()
    )

    critic_func = nn.Sequential(
        ConcatObsAndAction(),
        Linear(obs_size + 1, 256), # Increase number of units
        BatchNorm1d(256),          # Batch normalization
        LeakyReLU(),               # Leaky ReLU activation
        Linear(256, 128),          # Increase number of units
        BatchNorm1d(128),          # Batch normalization
        LeakyReLU(),               # Leaky ReLU activation
        Linear(128, 64),          # Increase number of units
        BatchNorm1d(64),          # Batch normalization
        LeakyReLU(),               # Leaky ReLU activation
        Linear(64, 1)             # Output layer
    )
    # actor_func = QFunction(obs_size, n_actions)
    print(f"obs_size:{env.observation_space[0].shape[0]}")
    print(f"n_actions:{env.action_space[0].n}")
    opt_a = torch.optim.Adam(actor_func.parameters(), lr=1e-2, eps=1e-2)
    opt_c = torch.optim.Adam(critic_func.parameters(), lr=1e-2, eps=1e-2)
    # Set the discount factor that discounts future rewards.
    gamma = 0.9

    start_epsilon = 0.7  # Initial exploration rate
    end_epsilon = 0.05    # Minimum exploration rate
    decay = 0.998       # Decay factor

    # Initialize the explorer
    explorer = pfrl.explorers.ExponentialDecayEpsilonGreedy(
        start_epsilon=start_epsilon,
        end_epsilon=end_epsilon,
        decay=decay,
        random_action_func=env.action_space[0].sample
    )

    # DQN uses Experience Replay.
    rbuf = replay_buffers.ReplayBuffer(10**6)
    phi = lambda x: np.array(x, dtype=np.float32)
    # Set the device id to use GPU. To use CPU only, set it to -1.
    gpu = -1
    # Now create an agent for each drone to make a distribute controll.
    agent = ddpg.DDPG(
        actor_func,
        critic_func,
        opt_a,
        opt_c,
        rbuf,
        gamma=0.995,
        explorer=explorer,
        replay_start_size=10,
        target_update_method="soft",
        target_update_interval=1,
        update_interval=1,
        soft_update_tau=5e-3,
        n_times_update=1,
        phi=phi,
        minibatch_size=5,
        gpu=-1,
    ) 
    # agent_array = [agent for age in range(env.n_agents)]
    actions = []
    for age in range(env.n_agents):
        if age < 3:
            agent.load(f"./models/sample_model{age}")
            action = agent.act(obs[age])
            actions.append(action)
    return actions
