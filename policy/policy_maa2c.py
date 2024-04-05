from maa2c.maa2c import MAA2C
from torch import nn
import os

def policy(obs, env): #Random Policy 
    obs_size = env.observation_space[0].shape[0] # map_aoba01 has 18 nodes, current-position(18dimensions)+current-goal(18dimensions)=36
    n_actions = env.action_space[0].n # map_aoba01 has 18 nodes and each node corresponds to one action

    agent = MAA2C(env, env.n_agents, obs_size, n_actions,
            critic_parameter_sharing=True, batch_size=50,
            episodes_before_train=10 ,use_cuda=False)
    actions = []
    model_dir = os.path.join("./models/", env.map_name + '_drones' + str(env.n_agents) + "/")
    agent.load(model_dir)
    actions = agent.action(obs)
    return actions
    # return hard_actions