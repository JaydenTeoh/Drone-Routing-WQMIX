import torch
import time
import torch.nn.functional as F
import torch.nn as nn
from algo.utils.popart import PopArt
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
from argparse import Namespace
import os
import numpy as np


class Learner(ABC):
    def __init__(self,
                 policy: torch.nn.Module,
                 optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./"):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_dir = model_dir
        self.iterations = 0

    def save_model(self, model_path):
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, path, seed=1):
        file_names = os.listdir(path)
        for f in file_names:
            '''Change directory to the specified seed (if exists)'''
            if f"seed_{seed}" in f:
                path = os.path.join(path, f)
                break
        model_names = os.listdir(path)
        if os.path.exists(path + "/obs_rms.npy"):
            model_names.remove("obs_rms.npy")
        model_names.sort()
        model_path = os.path.join(path, model_names[-1])
        self.policy.load_state_dict(torch.load(model_path, map_location={
            "cuda:0": self.device,
            "cuda:1": self.device,
            "cuda:2": self.device,
            "cuda:3": self.device,
            "cuda:4": self.device,
            "cuda:5": self.device,
            "cuda:6": self.device,
            "cuda:7": self.device
        }))

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError


class LearnerMAS(ABC):
    def __init__(self,
                 config: Namespace,
                 policy: torch.nn.Module,
                 optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./"):
        self.value_normalizer = None
        self.args = config
        self.n_agents = self.args.n_agents
        self.dim_obs = self.args.dim_obs
        self.dim_act = self.args.dim_act
        self.dim_id = self.n_agents
        self.device = torch.device("cuda" if (torch.cuda.is_available() and self.args.device == "gpu") else "cpu")

        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_dir = model_dir
        self.running_steps = config.running_steps
        self.iterations = 0

    def onehot_action(self, actions_int, num_actions):
        return F.one_hot(actions_int.long(), num_classes=num_actions)

    def save_model(self, model_path):
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, path, seed=1):
        self.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    def update_recurrent(self, *args):
        pass

    def act(self, *args, **kwargs):
        pass

    def get_hidden_states(self, *args):
        pass

    def lr_decay(self, *args):
        pass

class WQMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.alpha = config.alpha
        self.gamma = gamma

        self.use_popart = config.use_popart
        if self.use_popart:
            self.popart = PopArt(1) # denormalize values and normalize

        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(WQMIX_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

    def update(self, sample):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        state_next = torch.Tensor(sample['state_next']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
        terminals = torch.Tensor(sample['terminals']).all(dim=1, keepdims=True).float().to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        batch_size = actions.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        # calculate Q_tot
        _, action_max, q_eval = self.policy(obs, IDs)
        action_max = action_max.unsqueeze(-1)
        q_eval_a = q_eval.gather(-1, actions.long().reshape(batch_size, self.n_agents, 1))
        q_tot_eval = self.policy.Q_tot(q_eval_a * agent_mask, state)

        # calculate centralized Q
        q_eval_centralized = self.policy.q_centralized(obs, IDs).gather(-1, action_max.long())
        q_tot_centralized = self.policy.q_feedforward(q_eval_centralized * agent_mask, state)

        # calculate y_i
        if self.args.double_q:
            _, action_next_greedy, _ = self.policy(obs_next, IDs)
            action_next_greedy = action_next_greedy.unsqueeze(-1)
        else:
            q_next_eval = self.policy.target_Q(obs_next, IDs)
            action_next_greedy = q_next_eval.argmax(dim=-1, keepdim=True)
        q_eval_next_centralized = self.policy.target_q_centralized(obs_next, IDs).gather(-1, action_next_greedy)
        q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized * agent_mask, state_next)

        target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized
        td_error = q_tot_eval - target_value.detach()

        # calculate weights
        ones = torch.ones_like(td_error)
        w = ones * self.alpha
        if self.args.agent == "CWQMIX":
            condition_1 = ((action_max == actions.reshape([-1, self.n_agents, 1])) * agent_mask).all(dim=1)
            condition_2 = target_value > q_tot_centralized
            conditions = condition_1 | condition_2
            w = torch.where(conditions, ones, w)
        elif self.args.agent == "OWQMIX":
            condition = td_error < 0
            w = torch.where(condition, ones, w)
        else:
            AttributeError("You have assigned an unexpected WQMIX learner!")

        # calculate losses and train
        loss_central = self.mse_loss(q_tot_centralized, target_value.detach())
        loss_qmix = (w.detach() * (td_error ** 2)).mean()
        loss = loss_qmix + loss_central
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Qmix": loss_qmix.item(),
            "loss_central": loss_central.item(),
            "loss": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        return np.abs(td_error.cpu().detach().numpy()), info

    def update_recurrent(self, sample):
        """
        Update the parameters of the model with recurrent neural networks.
        """
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).mean(dim=1, keepdims=False).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().to(self.device)
        avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
        filled = torch.Tensor(sample['filled']).float().to(self.device)
        batch_size = actions.shape[0]
        episode_length = actions.shape[2]
        IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
            self.device)

        # calculate Q_tot
        rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
        _, actions_greedy, q_eval = self.policy(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                *rnn_hidden,
                                                avail_actions=avail_actions.reshape(-1, episode_length + 1, self.dim_act))
        q_eval = q_eval[:, :-1].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
        actions_greedy = actions_greedy.reshape(batch_size, self.n_agents, episode_length + 1, 1).detach()
        q_eval_a = q_eval.gather(-1, actions.long().reshape(batch_size, self.n_agents, episode_length, 1))
        q_eval_a = q_eval_a.transpose(1, 2).reshape(-1, self.n_agents, 1)
        q_tot_eval = self.policy.Q_tot(q_eval_a, state[:, :-1])

        # calculate centralized Q
        q_eval_centralized = self.policy.q_centralized(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                       IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                       *rnn_hidden)
        q_eval_centralized = q_eval_centralized[:, :-1].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
        q_eval_centralized_a = q_eval_centralized.gather(-1, actions_greedy[:, :, :-1].long())
        q_eval_centralized_a = q_eval_centralized_a.transpose(1, 2).reshape(-1, self.n_agents, 1)
        q_tot_centralized = self.policy.q_feedforward(q_eval_centralized_a, state[:, :-1])

        # calculate y_i
        target_rnn_hidden = self.policy.target_representation.init_hidden(batch_size * self.n_agents)
        if self.args.double_q:
            action_next_greedy = actions_greedy[:, :, 1:]
        else:
            _, q_next = self.policy.target_Q(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                             IDs.reshape(-1, episode_length + 1, self.n_agents),
                                             *target_rnn_hidden)
            q_next = q_next[:, 1:].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
            q_next[avail_actions[:, :, 1:] == 0] = -9999999
            action_next_greedy = q_next.argmax(dim=-1, keepdim=True)
        q_eval_next_centralized = self.policy.target_q_centralized(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                                   IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                                   *target_rnn_hidden)
        q_eval_next_centralized = q_eval_next_centralized[:, 1:].reshape(batch_size, self.n_agents, episode_length,
                                                                      self.dim_act)
        q_eval_next_centralized_a = q_eval_next_centralized.gather(-1, action_next_greedy)
        q_eval_next_centralized_a = q_eval_next_centralized_a.transpose(1, 2).reshape(-1, self.n_agents, 1)
        q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized_a, state[:, 1:])

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)
        filled = filled.reshape(-1, 1)
        
        if self.use_popart:
            print("target value WITHOUT normarlize", rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized)
            target_value = rewards + (1 - terminals) * self.args.gamma * self.popart.denormalize(q_tot_next_centralized) 
            target_value = self.popart(target_value) # normalize values
            print("target value with normarlize", target_value)
        else:
            target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized

        td_error = q_tot_eval - target_value.detach()
        td_error *= filled

        # calculate weights
        ones = torch.ones_like(td_error)
        w = ones * self.alpha
        if self.args.agent == "CWQMIX":
            actions_greedy = actions_greedy[:, :, :-1]
            condition_1 = (actions_greedy == actions.reshape([-1, self.n_agents, episode_length, 1])).all(dim=1)
            condition_1 = condition_1.reshape(-1, 1)
            condition_2 = target_value > q_tot_centralized
            conditions = condition_1 | condition_2
            w = torch.where(conditions, ones, w)
        elif self.args.agent == "OWQMIX":
            condition = td_error < 0
            w = torch.where(condition, ones, w)
        else:
            AttributeError("You have assigned an unexpected WQMIX learner!")

        # calculate losses and train
        error_central = (q_tot_centralized - target_value.detach()) * filled
        # using mse loss
        loss_central = (error_central ** 2).sum() / filled.sum()
        loss_qmix = (w.detach() * (td_error ** 2)).sum() / filled.sum()
        loss = loss_qmix + loss_central
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Qmix": loss_qmix.item(),
            "loss_central": loss_central.item(),
            "loss": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        return np.abs(td_error.cpu().detach().numpy()), info
