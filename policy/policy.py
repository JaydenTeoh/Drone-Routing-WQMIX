import gym
import random
import os
import numpy as np
import tensorflow as tf
from typing import List
from maddpg.agents.maddpg import MADDPGAgent, AbstractAgent
from maddpg.common.util import space_n_to_shape_n

### submission information ####
TEAM_NAME = "JaydenTeoh" 
#TEAM_NAME must be the same as the name registered on the DRP website 
#(or the team name if participating as a team).
##############################

def get_agents(env, n_agents, lr, batch_size,
               buff_size, num_units, num_layers, gamma, tau, priori_replay, alpha, beta,
               num_episodes, max_episode_len) -> List[AbstractAgent]:
    """
    This function generates the agents for the environment. The parameters are meant to be filled
    by sacred, and are therefore documented in the configuration function train_config.

    :returns List[AbstractAgent] returns a list of instantiated agents
    """
    agents = []
    for agent_idx in range(n_agents):
        agent = MADDPGAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                            buff_size, lr, num_layers,
                            num_units, gamma, tau, priori_replay, alpha=alpha,
                            max_step=num_episodes * max_episode_len, initial_beta=beta)
        agents.append(agent)
    return agents


def policy(obs, env): #Random Policy 
    num_episodes = 5000
    batch_size = 64
    max_episode_len = 100
    update_rate = 16

    agents = get_agents(env, env.n_agents, lr=1e-3, 
                    batch_size=batch_size, buff_size=1e6, 
                    num_units=128, num_layers=2, gamma=0.95, tau=0.01, priori_replay=True,
                    alpha=0.6, beta=0.5, num_episodes=num_episodes, max_episode_len=max_episode_len)
    print(f"obs_size:{env.observation_space[0].shape[0]}")
    print(f"n_actions:{env.action_space[0].n}")
    actions = []
    map_name = env.map_name
    model_dir = os.path.join("./logs/", map_name + '_drones' + str(env.n_agents) + "/", "models/")
    for age in range(env.n_agents):
        if age < 3:
            agent_dir = model_dir + f"agent_{age}/"
            agents[age].load(agent_dir)
            action = agents[age].target_action(obs[age].reshape(1,-1))
            actions.append(action[0])
    hard_actions = [np.argmax(act) for act in actions]
    print(hard_actions)
    return hard_actions