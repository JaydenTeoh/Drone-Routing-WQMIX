import gym
import random
import os
import numpy as np
import yaml
import torch
from algo.agents.wqmix import WQMIX_Agents
from argparse import Namespace
import time

### submission information ####
TEAM_NAME = "JaydenTeoh" 
#TEAM_NAME must be the same as the name registered on the DRP website 
#(or the team name if participating as a team).
##############################

def process_obs(obs):
    curr_pos = obs[:len(obs)//2]
    target_pos = np.argmax(obs[len(obs)//2:]) / (len(obs)//2) # get normalized goal
    curr_pos = np.append(curr_pos, target_pos)
    return curr_pos

def get_actions(agents, obs_n, avail_actions, *rnn_hidden):
    rnn_hidden_policy = rnn_hidden
    rnn_hidden_next, actions_n = agents.act(obs_n, *rnn_hidden_policy,
                                            avail_actions=avail_actions,
                                            test_mode=True) # for deterministic actions
    actions_n = actions_n.flatten() # need to flatten actions to 1D
    return actions_n, rnn_hidden_next

def get_avail_actions(env):
    avail_actions = []
    for i in range(env.n_agents):
        avail_actions.append(env.get_avail_agent_actions(i, env.n_actions)[0])
    avail_actions = np.array(avail_actions)

    return avail_actions

def load_config(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Namespace(**config_dict)
    return config

def policy(obs, env, rnn_hidden): #Random Policy 
    env_name_drones = env.map_name + f"_drones{env.n_agents}"
    args = load_config(f"./configs/{env_name_drones}.yaml")
    args.model_dir = "./models/wqmix/" + env_name_drones
    args.model_dir_load = args.model_dir + "/benchmark_model.pth"
    args.model_dir_save = args.model_dir
    args.log_dir = ""
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    agents = WQMIX_Agents(env, args, args.device)
    agents.load_model(args.model_dir_load)

    # print(f"Testing on {env.map_name} with {env.n_agent} drones")

    if rnn_hidden == None: # first env step
        rnn_hidden = agents.policy.representation.init_hidden(env.n_agents)

    obs_n = np.array([process_obs(o) for o in obs]) # process obs to reduce unnecessary dimensions
    available_actions = get_avail_actions(env) # get available actions for each agent at each node
    actions_n, new_rnn_hidden = get_actions(agents, obs_n, available_actions, *rnn_hidden)

    return actions_n, new_rnn_hidden