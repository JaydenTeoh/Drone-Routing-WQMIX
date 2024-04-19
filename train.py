import torch
import gym
import random
import numpy as np
import time
import math
import os
import matplotlib.pyplot as plt
import yaml
import argparse
from copy import deepcopy
from algo.agents.wqmix import WQMIX_Agents
from configs.benchmark import BENCHMARK_ENV_CONFIG
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter

def process_obs(obs):
    curr_pos = obs[:len(obs)//2]
    target_pos = torch.argmax(torch.tensor(obs[len(obs)//2:])) / (len(obs)//2) # get normalized goal
    curr_pos = torch.cat((torch.tensor(curr_pos), target_pos.view(1)))
    return curr_pos

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_start_goal(map_name, drone_num, few_shot_chance=0.2):
    """
    Randomly selects a start and goal configuration for a given map name and drone number.
    
    If zero-shot training, set few_shot_chance to 0.0 => always pick random goal and start
    """
    if random.random() > few_shot_chance:
        return [], []
    
    key = (map_name, drone_num)
    if key in BENCHMARK_ENV_CONFIG:
        positions = BENCHMARK_ENV_CONFIG[key]
        if positions:
            config = random.choice(positions)
            return config['start'], config['goal']
    
    return [], []

class Runner():
    def __init__(self, args, env, reward_list):
        # Set random seeds
        set_seed(args.seed)

        # Prepare directories
        self.args = args
        self.args.agent_name = args.agent

        self.env = env
        self.few_shot = args.few_shot
        self.eval_interval = self.args.eval_interval
        self.best_performance = None
        self.reward_list = reward_list
        
        # logging
        folder_name = f"seed_{args.seed}_" + time.asctime().replace(" ", "").replace(":", "_")
        self.args.model_dir = f"./models/wqmix/" + self.env.map_name + f"_drones{self.env.n_agents}"
        self.args.model_dir_save = self.args.model_dir_load = self.args.model_dir
        self.args.log_dir = f"./models/wqmix/" + self.env.map_name + f"_drones{self.env.n_agents}/logs/"
        if (not os.path.exists(self.args.model_dir_save)) and (not self.args.test_mode):
            os.makedirs(self.args.model_dir_save)

        if self.args.logger == "tensorboard":
            log_dir = os.path.join(self.args.log_dir, folder_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)

        self.episode_length = self.env.time_limit
        self.running_steps = args.running_steps
        self.training_frequency = args.training_frequency
        self.current_step = 0
        self.env_step = 0
        self.current_episode = 0
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.agents = WQMIX_Agents(self.env, args, args.device)


    def log_infos(self, info: dict, x_index: int):
        if x_index <= self.running_steps:
            for k, v in info.items():
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

    def get_actions(self, obs_n, avail_actions, *rnn_hidden, test_mode=False):
        rnn_hidden_policy = rnn_hidden
        rnn_hidden_next, actions_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                     avail_actions=avail_actions, test_mode=test_mode)
        return {'actions_n': actions_n, 'rnn_hidden': rnn_hidden_next}
    
    def get_avail_actions(self):
        avail_actions = []
        for i in range(self.env.n_agents):
            avail_actions.append(self.env.get_avail_agent_actions(i, self.env.n_actions)[0])
        avail_actions = torch.tensor(avail_actions, dtype=torch.int32)

        return avail_actions

    def benchmark(self, n_test_runs):
        key = (self.env.map_name, self.env.n_agents)
        if key in BENCHMARK_ENV_CONFIG:
            positions = BENCHMARK_ENV_CONFIG[key]
        else:
            return
        
        episode_scores = []
        won_count = 0
        for pos in positions:
            for i in range(n_test_runs):
                self.env.ee_env.input_start_ori_array, self.env.ee_env.input_goal_array = pos['start'], pos['goal']
                run_score, won = self.run_episode(test_mode=True)
                episode_scores.append(run_score)
                if won:
                    won_count += 1

        mean_loss = np.array(episode_scores).mean()
        results_info = {"Test-Results/Mean-Episode-Rewards": mean_loss,
                        "Test-Results/Win-Rate": won_count / len(episode_scores)}
        self.log_infos(results_info, self.current_step)

        if self.best_performance is None or mean_loss <= self.best_performance:
            # print("Best benchmark performance, saving model at eps ", self.current_step, "| Win rate: ", won_count / len(episode_scores))
            self.best_performance = mean_loss
            self.agents.save_model("benchmark_model.pth")

    def run_episode(self, test_mode):
        if not test_mode:
            if self.few_shot:
                self.env.ee_env.input_start_ori_array, self.env.ee_env.input_goal_array = get_start_goal(map_name=self.env.map_name,
                                                                                                        drone_num=self.env.n_agents,
                                                                                                        few_shot_chance=0.2)
            else:
                self.env.ee_env.input_start_ori_array, self.env.ee_env.input_goal_array = [], []

        obs_n = self.env.reset()
        obs_n = np.array([process_obs(o) for o in obs_n])
        done = False
        filled = np.zeros([self.episode_length, 1], np.int32)
        rnn_hidden = self.agents.policy.representation.init_hidden(self.agents.n_agents)
        env_step = 0
        episode_score = 0
        if test_mode:
            goal_step = [None] * self.agents.n_agents
            goal_drones = 0

        won_episode = False

        while not done:
            avail_actions = self.get_avail_actions()
            actions_dict = self.get_actions(obs_n, avail_actions, *rnn_hidden, test_mode=test_mode)
            actions_dict['actions_n'] = actions_dict['actions_n'].flatten()
            next_obs_n, rew_n, terminated_n, infos = self.env.step(actions_dict['actions_n'])
            next_obs_n = np.array([process_obs(o) for o in next_obs_n])
            rnn_hidden = actions_dict['rnn_hidden']
            if torch.any(torch.isnan(rnn_hidden[0])):
                print("NAN RNN VALUE", rnn_hidden)
                return

            filled[env_step] = 1
            done = all(terminated_n)

            if not test_mode:
                transition = (obs_n, actions_dict, obs_n.reshape(-1), rew_n, done, avail_actions)
                self.agents.memory.store_transitions(env_step, *transition)
                episode_score += np.mean(rew_n)
            else: # benchmark scoring
                for i in range(self.agents.n_agents):
                    if rew_n[i] == self.reward_list["goal"]:  # goal
                        goal_drones += 1
                        goal_step[i] = infos["step"]
                    elif rew_n[i] == self.reward_list["collision"] * self.env.speed:  # collision
                        if goal_step[i] == None:
                            goal_step[i] = 100
            
            if done and not test_mode:
                filled[env_step, 0] = 0
                avail_actions = self.get_avail_actions()

                terminal_data = (next_obs_n, next_obs_n.reshape(-1), avail_actions, filled)
                self.agents.memory.finish_path(env_step + 1, *terminal_data)
                # print(f"r:{rew_n},done:{terminated_n},info:{infos}")  
            
            env_step += 1
            obs_n = deepcopy(next_obs_n)
        
        if not test_mode:
            self.agents.memory.store_episodes()  # store episode data
            train_info = self.agents.train(self.current_step)  # train
            for k in train_info.keys():
                if math.isnan(train_info[k]):
                    print("NAN VALUE " + k, train_info[k]) # just to verify that there's no nan losses
                    break
            episode_info = {"Train_Episode_Score": episode_score}
            self.log_infos(episode_info, self.current_step)
            self.log_infos(train_info, self.current_step)
        else: # benchmark scoring
            for i in range(self.agents.n_agents):
                if goal_step[i] == None:
                    goal_step[i] = 100
            episode_score = sum(goal_step)
            won_episode = goal_drones == self.agents.n_agents
            return episode_score, won_episode
        
        return episode_score
        

    def train(self, n_episodes):
        for _ in range(n_episodes):
            if self.current_step % self.eval_interval == 0:
                self.benchmark(n_test_runs=20)

            self.current_step += 1
            self.run_episode(test_mode=False)


    def run(self):
        self.train(self.agents.n_episodes)
        print("Finish training.")
        self.agents.save_model("final_train_model.pth")

    def finish(self):
        self.env.close()
        self.writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='QMIX for DRP env')
    parser.add_argument('--config', type=str, default='./qmix/drp_config.yaml', help='Path to YAML config for training QMIX on DRP env')
    parser.add_argument('--drone_num', type=int, default=3, help='Number of drones')
    parser.add_argument('--map_name', type=str, default='map_3x3', help='Name of the map')
    args = parser.parse_args()
    return args

def load_config(args):
    with open(args.config, 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Namespace(**config_dict)
    return config


'''
Problem information 
map_name            |number of drone| number of problem
--------------------|---------------|-------------------
map_3x3             |  2            | 3
map_3x3             |  3            | 4
map_3x3             |  4            | 3
map_aoba01          |  4            | 3
map_aoba01          |  6            | 4
map_aoba01          |  8            | 3
map_shibuya         |  8            | 3
map_shibuya         |  10           | 4
map_shibuya         |  12           | 3
'''

if __name__ == "__main__":
    # make env
    args = parse_arguments()
    args.config = f"./configs/{args.map_name}_drones{args.drone_num}.yaml"
    config = load_config(args)

    reward_list = {
        "goal": 100,
        "collision": -20,
        "wait": -0.2,
        "move": -0.1,
    }

    env = gym.make(
        "drp_env:drp-" + str(args.drone_num) + "agent_" + args.map_name + "-v2",
        state_repre_flag="onehot_fov",
        reward_list=reward_list,
    )

    config.few_shot = True # whether to include benchmarks in training envs
    runner = Runner(config, env, reward_list)
    runner.run()
    runner.finish()
