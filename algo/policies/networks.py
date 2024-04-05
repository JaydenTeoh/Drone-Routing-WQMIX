from typing import Optional, Sequence, Tuple, Type, Union, Callable
import torch.nn as nn
import numpy as np
import torch
import copy

from gym.spaces import Discrete
from algo.policies.mixers import *

ModuleType = Type[nn.Module]


def mlp_block(input_dim: int,
              output_dim: int,
              normalize: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
              device: Optional[Union[str, int, torch.device]] = None) -> Tuple[Sequence[ModuleType], Tuple[int]]:
    block = []
    linear = nn.Linear(input_dim, output_dim, device=device)
    if initialize is not None:
        initialize(linear.weight)
        nn.init.constant_(linear.bias, 0)
    block.append(linear)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        block.append(normalize(output_dim, device=device))
    return block, (output_dim,)

class BasicQhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], n_actions, None, None, None, device)[0])
        self.model = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
class Basic_MLP(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(Basic_MLP, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                         device=self.device)
            layers.extend(mlp)
            
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        tensor_observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': self.model(tensor_observation)}

def gru_block(input_dim: int,
              output_dim: int,
              num_layers: int = 1,
              dropout: float = 0,
              initialize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
              device: Optional[Union[str, int, torch.device]] = None) -> Tuple[nn.Module, int]:
    gru = nn.GRU(input_size=input_dim,
                 hidden_size=output_dim,
                 num_layers=num_layers,
                 batch_first=True,
                 dropout=dropout,
                 device=device)
    if initialize is not None:
        for weight_list in gru.all_weights:
            for weight in weight_list:
                if len(weight.shape) > 1:
                    initialize(weight)
                else:
                    nn.init.constant_(weight, 0)
    return gru, output_dim


def lstm_block(input_dim: int,
               output_dim: int,
               num_layers: int = 1,
               dropout: float = 0,
               initialize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
               device: Optional[Union[str, int, torch.device]] = None) -> Tuple[nn.Module, int]:
    lstm = nn.LSTM(input_size=input_dim,
                   hidden_size=output_dim,
                   num_layers=num_layers,
                   batch_first=True,
                   dropout=dropout,
                   device=device)
    if initialize is not None:
        for weight_list in lstm.all_weights:
            for weight in weight_list:
                if len(weight.shape) > 1:
                    initialize(weight)
                else:
                    nn.init.constant_(weight, 0)
    return lstm, output_dim


class Basic_RNN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 normalize: Optional[nn.Module] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Basic_RNN, self).__init__()
        self.input_shape = input_shape
        self.fc_hidden_sizes = hidden_sizes["fc_hidden_sizes"]
        self.recurrent_hidden_size = hidden_sizes["recurrent_hidden_size"]
        self.N_recurrent_layer = kwargs["N_recurrent_layers"]
        self.dropout = kwargs["dropout"]
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes["recurrent_hidden_size"],)}
        self.mlp, self.rnn, output_dim = self._create_network()
        if self.normalize is not None:
            self.use_normalize = True
            self.input_norm = self.normalize(input_shape, device=device)
            self.norm_rnn = self.normalize(output_dim, device=device)
        else:
            self.use_normalize = False

    def _create_network(self) -> Tuple[nn.Module, nn.Module, int]:
        layers = []
        input_shape = self.input_shape
        for h in self.fc_hidden_sizes:
            mlp_layer, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                               device=self.device)
            layers.extend(mlp_layer)
        if self.lstm:
            rnn_layer, input_shape = lstm_block(input_shape[0], self.recurrent_hidden_size, self.N_recurrent_layer,
                                                self.dropout, self.initialize, self.device)
        else:
            rnn_layer, input_shape = gru_block(input_shape[0], self.recurrent_hidden_size, self.N_recurrent_layer,
                                               self.dropout, self.initialize, self.device)
        return nn.Sequential(*layers), rnn_layer, input_shape

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor = None):
        mlp_output = self.mlp(self.input_norm(x)) if self.use_normalize else self.mlp(x)
        self.rnn.flatten_parameters()
        if self.lstm:
            output, (hn, cn) = self.rnn(mlp_output, (h, c))
            if self.use_normalize:
                output = self.norm_rnn(output)
            return {"state": output, "rnn_hidden": hn.detach(), "rnn_cell": cn.detach()}
        else:
            output, hn = self.rnn(mlp_output, h)
            if self.use_normalize:
                output = self.norm_rnn(output)
            return {"state": output, "rnn_hidden": hn.detach(), "rnn_cell": None}

    def init_hidden(self, batch):
        hidden_states = torch.zeros(size=(self.N_recurrent_layer, batch, self.recurrent_hidden_size)).to(self.device)
        cell_states = torch.zeros_like(hidden_states).to(self.device) if self.lstm else None
        return hidden_states, cell_states

    def init_hidden_item(self, i, *rnn_hidden):
        if self.lstm:
            rnn_hidden[0][:, i] = torch.zeros(size=(self.N_recurrent_layer, self.recurrent_hidden_size)).to(self.device)
            rnn_hidden[1][:, i] = torch.zeros(size=(self.N_recurrent_layer, self.recurrent_hidden_size)).to(self.device)
            return rnn_hidden
        else:
            rnn_hidden[0][:, i] = torch.zeros(size=(self.N_recurrent_layer, self.recurrent_hidden_size)).to(self.device)
            return rnn_hidden

    def get_hidden_item(self, i, *rnn_hidden):
        return (rnn_hidden[0][:, i], rnn_hidden[1][:, i]) if self.lstm else (rnn_hidden[0][:, i], None)


class Qtran_MixingQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 mixer: Optional[VDN_mixer] = None,
                 qtran_mixer: Optional[QTRAN_base] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Qtran_MixingQnetwork, self).__init__()
        self.n_actions = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.n_actions, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.qtran_net = qtran_mixer
        self.target_qtran_net = copy.deepcopy(qtran_mixer)
        self.q_tot = mixer

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                *rnn_hidden: torch.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions)
            evalQ_detach = evalQ.clone().detach()
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return rnn_hidden, outputs['state'], argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return rnn_hidden, outputs['state'], self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.qtran_net.parameters(), self.target_qtran_net.parameters()):
            tp.data.copy_(ep)


class MixingQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 mixer: Optional[VDN_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MixingQnetwork, self).__init__()
        self.n_actions = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.n_actions, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.eval_Qtot = mixer
        self.target_Qtot = copy.deepcopy(self.eval_Qtot)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                *rnn_hidden: torch.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, rnn_hidden[0])
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions)
            evalQ_detach = evalQ.clone().detach()
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(dim=-1, keepdim=False)

        return rnn_hidden, argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return rnn_hidden, self.target_Qhead(q_inputs)

    def Q_tot(self, q, states=None):
        return self.eval_Qtot(q, states)

    def target_Q_tot(self, q, states=None):
        return self.target_Qtot(q, states)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)


class Weighted_MixingQnetwork(MixingQnetwork):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 mixer: Optional[VDN_mixer] = None,
                 ff_mixer: Optional[QMIX_FF_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                      normalize, initialize, activation, device, **kwargs)
        self.eval_Qhead_centralized = copy.deepcopy(self.eval_Qhead)
        self.target_Qhead_centralized = copy.deepcopy(self.eval_Qhead_centralized)
        self.q_feedforward = ff_mixer
        self.target_q_feedforward = copy.deepcopy(self.q_feedforward)

    def q_centralized(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
        else:
            outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.eval_Qhead_centralized(q_inputs)

    def target_q_centralized(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
        else:
            outputs = self.target_representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_Qhead_centralized(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead_centralized.parameters(), self.target_Qhead_centralized.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.q_feedforward.parameters(), self.target_q_feedforward.parameters()):
            tp.data.copy_(ep)

