import numpy as np
from abc import ABC, abstractmethod
from algo.utils.segtree import *
import random


class BaseBuffer(ABC):
    """
    Basic buffer for MARL algorithms.
    """

    def __init__(self, *args):
        self.n_agents, self.state_space, self.obs_space, self.act_space, self.rew_space, self.done_space, self.n_envs, self.buffer_size = args
        self.ptr = 0  # last data pointer
        self.size = 0  # current buffer size

    @property
    def full(self):
        return self.size >= self.n_size

    @abstractmethod
    def store(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def clear(self, *args):
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args):
        raise NotImplementedError

    def store_transitions(self, *args, **kwargs):
        return

    def store_episodes(self, *args, **kwargs):
        return

    def finish_path(self, *args, **kwargs):
        return
    
    @abstractmethod
    def update_priorities(self, *args):
        raise NotImplementedError


class MARL_OffPolicyBuffer(BaseBuffer):
    """
    Replay buffer for off-policy MARL algorithms.

    Args:
        n_agents: number of agents.
        state_space: global state space, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        act_space: action space for one agent (suppose same actions space for group agents).
        rew_space: reward space.
        done_space: terminal variable space.
        n_envs: number of parallel environments.
        buffer_size: buffer size for one environment.
        batch_size: batch size of transition data for a sample.
        **kwargs: other arguments.
    """

    def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space,
                 n_envs, buffer_size, batch_size, **kwargs):
        super(MARL_OffPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                                                   n_envs, buffer_size)
        self.n_size = buffer_size // n_envs
        self.batch_size = batch_size
        if self.state_space is not None:
            self.store_global_state = True
        else:
            self.store_global_state = False
        self.data = {}
        self.clear()
        self.keys = self.data.keys()

    def clear(self):
        self.data = {
            'obs': np.zeros((self.n_size, self.n_agents) + self.obs_space).astype(np.float32),
            'actions': np.zeros((self.n_size, self.n_agents) + self.act_space).astype(np.float32),
            'obs_next': np.zeros((self.n_size, self.n_agents) + self.obs_space).astype(np.float32),
            'rewards': np.zeros((self.n_size, self.n_agents)).astype(np.float32),
            'terminals': np.zeros((self.n_size, self.n_agents)).astype(np.bool_),
            'agent_mask': np.ones((self.n_size, self.n_agents)).astype(np.bool_)
        }
        if self.state_space is not None:
            self.data.update({'state': np.zeros((self.n_size, ) + self.state_space).astype(np.float32),
                              'state_next': np.zeros((self.n_size, ) + self.state_space).astype(np.float32)})
        self.ptr, self.size = 0, 0

    def store(self, step_data):
        for k in self.keys:
            self.data[k][self.ptr, :] = step_data[k]
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = np.min([self.size + 1, self.n_size])

    def sample(self):
        step_choices = np.random.choice(self.size, self.batch_size)
        samples = {k: self.data[k][step_choices] for k in self.keys}
        return samples
    

class MARL_OffPolicyBuffer_RNN(MARL_OffPolicyBuffer):
    """
    Replay buffer for off-policy MARL algorithms with DRQN trick.

    Args:
        n_agents: number of agents.
        state_space: global state space, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        act_space: action space for one agent (suppose same actions space for group agents).
        rew_space: reward space.
        done_space: terminal variable space.
        n_envs: number of parallel environments.
        buffer_size: buffer size for one environment.
        batch_size: batch size of episodes for a sample.
        kwargs: other arguments.
    """

    def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space,
                 n_envs, buffer_size, batch_size, **kwargs):
        self.max_eps_len = kwargs['max_episode_length']
        self.dim_act = kwargs['dim_act']
        super(MARL_OffPolicyBuffer_RNN, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                       done_space, n_envs, buffer_size, batch_size)

        self.episode_data = {}
        self.clear_episodes()

    def clear(self):
        self.data = {
            'obs': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len + 1) + self.obs_space, np.float32),
            'actions': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.act_space, np.float32),
            'rewards': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool_),
            'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool_),
            'filled': np.zeros((self.buffer_size, self.max_eps_len, 1)).astype(np.bool_)
        }
        if self.state_space is not None:
            self.data.update({'state': np.zeros(
                (self.buffer_size, self.max_eps_len + 1) + self.state_space).astype(np.float32)})
        self.ptr, self.size = 0, 0

    def clear_episodes(self):
        self.episode_data = {
            'obs': np.zeros((self.n_agents, self.max_eps_len + 1) + self.obs_space, dtype=np.float32),
            'actions': np.zeros((self.n_agents, self.max_eps_len) + self.act_space, dtype=np.float32),
            'rewards': np.zeros((self.n_agents, self.max_eps_len) + self.rew_space, dtype=np.float32),
            'terminals': np.zeros((self.max_eps_len,) + self.done_space).astype(np.bool_),
            'avail_actions': np.ones((self.n_agents, self.max_eps_len + 1, self.dim_act)).astype(np.bool_),
            'filled': np.zeros((self.max_eps_len, 1), dtype=np.bool_),
        }
        if self.state_space is not None:
            self.episode_data.update({
                'state': np.zeros((self.max_eps_len + 1, ) + self.state_space, dtype=np.float32),
            })

    def store_transitions(self, t_envs, *transition_data):
        obs_n, actions_dict, state, rewards, terminated, avail_actions = transition_data
        self.episode_data['obs'][:, t_envs] = obs_n
        self.episode_data['actions'][:, t_envs] = actions_dict['actions_n']
        self.episode_data['rewards'][:, t_envs] = rewards
        self.episode_data['terminals'][t_envs] = terminated
        self.episode_data['avail_actions'][:, t_envs] = avail_actions
        if self.state_space is not None:
            self.episode_data['state'][t_envs] = state

    def store_episodes(self):
        for k in self.keys:
            self.data[k][self.ptr] = self.episode_data[k].copy()
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])
        self.clear_episodes()

    def finish_path(self, next_t, *terminal_data):
        obs_next, state_next, available_actions, filled = terminal_data
        self.episode_data['obs'][:, next_t] = obs_next
        self.episode_data['state'][next_t] = state_next
        self.episode_data['avail_actions'][:, next_t] = available_actions
        self.episode_data['filled'] = filled

    def sample(self):
        sample_choices = np.random.choice(self.size, self.batch_size)
        samples = {k: self.data[k][sample_choices] for k in self.keys}
        return samples


class Per_MARL_OffPolicyBuffer_RNN(MARL_OffPolicyBuffer):
    """
    Prioritized Experience Replay buffer for off-policy MARL algorithms with DRQN trick.

    Args:
        n_agents: number of agents.
        state_space: global state space, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        act_space: action space for one agent (suppose same actions space for group agents).
        rew_space: reward space.
        done_space: terminal variable space.
        n_envs: number of parallel environments.
        buffer_size: buffer size for one environment.
        batch_size: batch size of episodes for a sample.
        kwargs: other arguments.
    """

    def __init__(self, per_nu, per_alpha, n_agents, state_space, obs_space, act_space, rew_space, done_space,
                 n_envs, buffer_size, batch_size, **kwargs):
        self.max_eps_len = kwargs['max_episode_length']
        self.dim_act = kwargs['dim_act']
        super(Per_MARL_OffPolicyBuffer_RNN, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                       done_space, n_envs, buffer_size, batch_size)

        it_capacity = 1
        while it_capacity < self.buffer_size:
            it_capacity *= 2

        self.episode_data = {}
        self.clear_episodes()


        # PER configs
        self.per_nu = per_nu
        self.per_alpha = per_alpha
        self.per_eps = 1e-8
        self._it_sum = []
        self._it_min = []
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def clear(self):
        self.data = {
            'obs': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len + 1) + self.obs_space, np.float32),
            'actions': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.act_space, np.float32),
            'rewards': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool_),
            'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool_),
            'filled': np.zeros((self.buffer_size, self.max_eps_len, 1)).astype(np.bool_)
        }
        if self.state_space is not None:
            self.data.update({'state': np.zeros(
                (self.buffer_size, self.max_eps_len + 1) + self.state_space).astype(np.float32)})
        self.ptr, self.size = 0, 0
        self._it_sum = []
        self._it_min = []

    def clear_episodes(self):
        self.episode_data = {
            'obs': np.zeros((self.n_agents, self.max_eps_len + 1) + self.obs_space, dtype=np.float32),
            'actions': np.zeros((self.n_agents, self.max_eps_len) + self.act_space, dtype=np.float32),
            'rewards': np.zeros((self.n_agents, self.max_eps_len) + self.rew_space, dtype=np.float32),
            'terminals': np.zeros((self.max_eps_len,) + self.done_space).astype(np.bool_),
            'avail_actions': np.ones((self.n_agents, self.max_eps_len + 1, self.dim_act)).astype(np.bool_),
            'filled': np.zeros((self.max_eps_len, 1), dtype=np.bool_),
        }
        if self.state_space is not None:
            self.episode_data.update({
                'state': np.zeros((self.max_eps_len + 1, ) + self.state_space, dtype=np.float32),
            })

    def store_transitions(self, t_envs, *transition_data):
        obs_n, actions_dict, state, rewards, terminated, avail_actions = transition_data
        self.episode_data['obs'][:, t_envs] = obs_n
        self.episode_data['actions'][:, t_envs] = actions_dict['actions_n']
        self.episode_data['rewards'][:, t_envs] = rewards
        self.episode_data['terminals'][t_envs] = terminated
        self.episode_data['avail_actions'][:, t_envs] = avail_actions
        if self.state_space is not None:
            self.episode_data['state'][t_envs] = state

    def store_episodes(self):
        for k in self.keys:
            self.data[k][self.ptr] = self.episode_data[k].copy()
        self._it_sum[self.ptr] = self._max_priority ** self.per_alpha
        self._it_min[self.ptr] = self._max_priority ** self.per_alpha

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])
        self.clear_episodes()

    def finish_path(self, next_t, *terminal_data):
        obs_next, state_next, available_actions, filled = terminal_data
        self.episode_data['obs'][:, next_t] = obs_next
        self.episode_data['state'][next_t] = state_next
        self.episode_data['avail_actions'][:, next_t] = available_actions
        self.episode_data['filled'] = filled

    def _sample_proportional(self):
        res = []
        p_total = self._it_sum.sum(0, self.size - 1)
        every_range_len = p_total / self.batch_size
        for i in range(self.batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(int(idx))
        return res


    def sample(self, beta):
        step_choices = np.zeros(self.batch_size)
        # weights = np.zeros(self.batch_size)

        assert beta > 0

        idxes = self._sample_proportional()
        weights_ = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = p_min * self.size ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = p_sample * self.size ** (-beta)
            weights_.append(weight / max_weight)
        step_choices = idxes
        # weights = np.array(weights_)
        samples = {k: self.data[k][step_choices] for k in self.keys}
        return samples, step_choices

    def update_priorities(self, idxes, td_errors):
        for idx, error in zip(idxes, td_errors):
            assert 0 <= idx < self.size
            priority = error[0]

            if priority == 0:
                priority += self.per_eps

            self._it_sum[idx] = priority ** self.per_alpha
            self._it_min[idx] = priority ** self.per_alpha

            self._max_priority = max(self._max_priority, priority)
