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

# args = Namespace(agent="OWQMIX", seed=101, logger="tensorboard", test_mode=False, device="cpu",
#                    continuous_action=False,
#                    policy="Mixing_Q_network",
#                    representation="Basic_RNN",
#                    hidden_dim_mixing_net=64,  # hidden units of mixing network
#                    hidden_dim_hyper_net=128,  # hidden units of hyper network
#                    hidden_dim_ff_mix_net=256, # hidden units of mixing network
#                    fc_hidden_sizes=[ 128, 64, 32, ],
#                    qtran_net_hidden_dim=64, lambda_opt=1.0, lambda_nopt=1.0,
#                    buffer_size=5000, batch_size=64, learning_rate=3e-4,
#                    gamma=0.995,  # discount factor
#                    alpha=0.1,
#                    double_q=True,  # use double q learning
#                    representation_hidden_size=[128, 64, 32, ],
#                    q_hidden_size=[128, ],  # the units for each hidden layer
#                    activation="LeakyReLU", use_recurrent=True, rnn="GRU",
#                    recurrent_hidden_size=64, N_recurrent_layers=2, dropout=0.1,
#                    start_greedy=1.0, end_greedy=0.05, decay_step_greedy=20000,
#                    running_steps=3000000, train_per_step=False, start_training=1000, 
#                    sync_frequency=100, training_frequency=1, use_grad_clip=True, grad_clip_norm=0.5)

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
    args = load_config(f"./algo/configs/{env_name_drones}.yaml")
    args.model_dir = "./models/wqmix/" + env_name_drones
    args.model_dir_load = args.model_dir + "/final_train_model.pth"
    args.model_dir_save = args.model_dir
    args.log_dir = ""
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    agents = WQMIX_Agents(env, args, args.device)
    agents.load_model(args.model_dir_load)

    print(f"obs_size:{env.observation_space[0].shape[0]}")
    print(f"n_actions:{env.action_space[0].n}")

    if rnn_hidden == None: # first env step
        rnn_hidden = agents.policy.representation.init_hidden(env.n_agents)

    obs_n = np.array([process_obs(o) for o in obs]) # process obs to reduce unnecessary dimensions
    available_actions = get_avail_actions(env) # get available actions for each agent at each node
    actions_n, new_rnn_hidden = get_actions(agents, obs_n, available_actions, *rnn_hidden)

    return actions_n, new_rnn_hidden