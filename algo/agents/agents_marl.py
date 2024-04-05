import os
import torch
import torch.nn as nn
from algo.utils.buffers import BaseBuffer
from algo.policies.learners import LearnerMAS
from typing import Optional, Sequence, Tuple, Type, Union, Callable
from argparse import Namespace

def create_directory(path):
    """Create an empty directory.
    Args:
        path: the path of the directory
    """
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1, len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"


class MARLAgents(object):
    """The class of basic agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
        policy: the neural network modules of the agent.
        memory: the experience replay buffer.
        learner: the learner for the corresponding agent.
        device: the calculating device of the model, such as CPU or GPU.
        log_dir: the directory of the log file.
        model_dir: the directory for models saving.
    """
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 memory: BaseBuffer,
                 learner: LearnerMAS,
                 device: Optional[Union[str, int, torch.device]] = None,
                 log_dir: str = "./logs/",
                 model_dir: str = "./models/"):
        self.args = config
        self.n_agents = config.n_agents
        self.dim_obs = self.args.dim_obs
        self.dim_act = self.args.dim_act
        self.dim_id = self.n_agents
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and config.device in ["gpu", "cuda:0"]) else "cpu")
        self.start_training = config.start_training

        self.policy = policy
        self.memory = memory
        self.learner = learner
        self.device = device
        self.log_dir = log_dir
        self.model_dir_save, self.model_dir_load = config.model_dir_save, config.model_dir_load
        create_directory(log_dir)
        create_directory(model_dir)

    def save_model(self, model_name):
        model_path = os.path.join(self.model_dir_save, model_name)
        self.learner.save_model(model_path)

    def load_model(self, path, seed=1):
        self.learner.load_model(path, seed)

    def act(self, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError
