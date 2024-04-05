from typing import Optional, Sequence, Tuple, Type, Union, Callable
from drp_env.drp_env import DrpEnv
from algo.policies.networks import *
from algo.policies.mixers import *
from algo.utils.buffers import *
from algo.policies.learners import *
from algo.agents.agents_marl import MARLAgents
from algo.utils.util import *
import torch.nn as nn
import torch

class WQMIX_Agents(MARLAgents):
    """The implementation of QTRAN agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
        device: the calculating device of the model, such as CPU or GPU.
    """
    def __init__(self, env: DrpEnv, config: Namespace, device: Optional[Union[int, str, torch.device]] = None):
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.per_beta_start = config.per_beta_start
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        self.dim_act = env.action_space[0].n
        self.dim_obs = env.observation_space[0].shape[0] // 2 + 1
        self.n_agents = env.n_agents
        self.obs_shape = (self.dim_obs,)
        self.dim_state = (self.dim_obs * env.n_agents,)
        self.act_shape = ()
        self.rew_shape = ()
        self.done_shape = (1,)
        self.n_agents = env.n_agents

        # env params
        config.dim_obs, config.dim_act = self.dim_obs, self.dim_act
        config.obs_shape, config.act_shape = (self.dim_obs,), ()
        config.rew_shape = config.done_shape = (1,)
        config.action_space = env.action_space
        config.state_space = (self.dim_obs * env.n_agents,)

        self.lr = config.learning_rate
        self.n_episodes = config.running_steps // env.time_limit # for learning rate scheduler
        self.start_training = config.start_training

        self.model_dir = config.model_dir
        self.use_recurrent = config.use_recurrent

        if config.use_global_state:
            config.dim_state, state_shape = self.dim_state[0], self.dim_state
        else:
            config.dim_state, state_shape = None, None

        if self.use_recurrent:
            hidden_sizes = {"fc_hidden_sizes": config.fc_hidden_sizes, "recurrent_hidden_size": config.recurrent_hidden_size}
            kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                            "dropout": config.dropout,
                            "rnn": config.rnn,
                            "hidden_sizes": hidden_sizes}
        else:
            kwargs_rnn = {}

        representation = Basic_RNN(input_shape=self.obs_shape, activation=ActivationFunctions[config.activation], 
                                   initialize=InitializeFunctions[config.initialize], 
                                   normalize=NormalizeFunctions[config.normalize], device=device, **kwargs_rnn)
        mixer = QMIX_mixer(config.dim_state, config.hidden_dim_mixing_net, config.hidden_dim_hyper_net,
                           self.n_agents, device)
        ff_mixer = QMIX_FF_mixer(config.dim_state, config.hidden_dim_ff_mix_net, self.n_agents, device)

        policy = Weighted_MixingQnetwork(env.action_space[0], self.n_agents, representation,
                                      mixer, ff_mixer, config.representation_hidden_size, activation=ActivationFunctions[config.activation], 
                                      initialize=InitializeFunctions[config.initialize], normalize=NormalizeFunctions[config.normalize], use_recurrent=config.use_recurrent, 
                                      device=device, **kwargs_rnn)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=self.n_episodes)
        
        # for PER
        self.beta_anneal = DecayThenFlatSchedule(self.per_beta_start, 1.0, self.n_episodes, decay="linear")

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        buffer = Per_MARL_OffPolicyBuffer_RNN if self.use_recurrent else MARL_OffPolicyBuffer
        if self.use_recurrent: # only implemented PER for recurrent
            input_buffer = (config.per_nu, config.per_alpha, self.n_agents, state_shape, self.obs_shape, self.act_shape, self.rew_shape,
                            self.done_shape, 1, config.buffer_size, config.batch_size)
        else:
            input_buffer = (self.n_agents, state_shape, self.obs_shape, self.act_shape, self.rew_shape,
                            self.done_shape, 1, config.buffer_size, config.batch_size)
        
        memory = buffer(*input_buffer, max_episode_length=env.time_limit, dim_act=self.dim_act)
        config.n_agents = self.n_agents

        learner = WQMIX_Learner(config, policy, optimizer, scheduler,
                                config.device, config.model_dir, self.gamma,
                                config.sync_frequency)
        super(WQMIX_Agents, self).__init__(config, policy, memory, learner, device,
                                           config.log_dir, config.model_dir)
        self.on_policy = False

    def act(self, obs_n, *rnn_hidden, batch_size=1, avail_actions=None, test_mode=False):
        batch_size = batch_size
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
        if self.use_recurrent:
            batch_agents = batch_size * self.n_agents
            if avail_actions is not None:
                avail_actions=avail_actions.reshape(batch_agents, 1, -1)
            hidden_state, greedy_actions, _ = self.policy(obs_in.view(batch_agents, 1, -1),
                                                          agents_id.view(batch_agents, 1, -1),
                                                          *rnn_hidden,
                                                          avail_actions)
            greedy_actions = greedy_actions.view(batch_size, self.n_agents)
        else:
            hidden_state, greedy_actions, _ = self.policy(obs_in, agents_id, avail_actions=avail_actions)
        greedy_actions = greedy_actions.cpu().detach().numpy()

        if test_mode:
            return hidden_state, greedy_actions
        else:
            if avail_actions is None:
                random_actions = np.random.choice(self.dim_act, self.n_agents)
            else:
                random_actions = torch.distributions.Categorical(torch.Tensor(avail_actions)).sample().numpy()
            if np.random.rand() < self.egreedy:
                return hidden_state, random_actions
            else:
                return hidden_state, greedy_actions[0]

    def train(self, i_step, n_epoch=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        beta = self.beta_anneal.eval(i_step)

        if i_step > self.start_training:
            for i_epoch in range(n_epoch):
                sample, idxes = self.memory.sample(beta)
                if self.use_recurrent:
                    td_error, info_train = self.learner.update_recurrent(sample)
                    self.memory.update_priorities(idxes, td_error) # update the priorities in PER buffer
                else:
                    td_error, info_train = self.learner.update(sample)
                    # have not implemented PER for non-RNN
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
