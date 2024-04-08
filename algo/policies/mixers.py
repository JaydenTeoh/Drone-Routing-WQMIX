import torch
import torch.nn as nn
import torch.nn.functional as F


class VDN_mixer(nn.Module):
    def __init__(self):
        super(VDN_mixer, self).__init__()

    def forward(self, values_n, states=None):
        return values_n.sum(dim=1)


class QMIX_mixer(nn.Module):
    def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device):
        super(QMIX_mixer, self).__init__()
        self.device = device
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_hypernet_hidden = dim_hypernet_hidden
        self.n_agents = n_agents
        # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
        # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.LeakyReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, self.dim_hidden * self.n_agents)).to(device)
        self.hyper_w_2 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.LeakyReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, self.dim_hidden)).to(device)

        self.hyper_b_1 = nn.Linear(self.dim_state, self.dim_hidden).to(device)
        self.hyper_b_2 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.LeakyReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, 1)).to(device)

    def forward(self, values_n, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.reshape(-1, 1, self.n_agents)
        # First layer
        w_1 = torch.abs(self.hyper_w_1(states))
        w_1 = w_1.view(-1, self.n_agents, self.dim_hidden)
        b_1 = self.hyper_b_1(states)
        b_1 = b_1.view(-1, 1, self.dim_hidden)
        hidden = F.elu(torch.bmm(agent_qs, w_1) + b_1)
        # Second layer
        w_2 = torch.abs(self.hyper_w_2(states))
        w_2 = w_2.view(-1, self.dim_hidden, 1)
        b_2 = self.hyper_b_2(states)
        b_2 = b_2.view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_2) + b_2
        # Reshape and return
        q_tot = y.view(-1, 1)
        return q_tot


class QMIX_FF_mixer(nn.Module):
    def __init__(self, dim_state, dim_hidden, n_agents, device):
        super(QMIX_FF_mixer, self).__init__()
        self.device = device
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_input = self.n_agents + self.dim_state
        self.ff_net = nn.Sequential(nn.Linear(self.dim_input, self.dim_hidden),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.dim_hidden, 1)).to(self.device)
        self.ff_net_bias = nn.Sequential(nn.Linear(self.dim_state, self.dim_hidden),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.dim_hidden, 1)).to(self.device)

    def forward(self, values_n, states):
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view([-1, self.n_agents])
        inputs = torch.cat([agent_qs, states], dim=-1).to(self.device)
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = y.view([-1, 1])
        return q_tot
