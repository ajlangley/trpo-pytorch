from argparse import ArgumentParser
import gym
from gym.spaces import Box, Discrete
import numpy as np
import os
import roboschool
import torch
from yaml import load

from models import build_diag_gauss_policy, build_multinomial_policy
from torch_utils import get_device

parser = ArgumentParser(prog='render_simulation.py',
                        description='Render a simulation of an agent in the' \
                        ' specified environment.')
parser.add_argument('--model-name', type=str, dest='model_name', required=True,
                    help='The entry in config.yaml for which the model and settings '\
                    'should be loaded.')
parser.add_argument('-T', '--max-timesteps', type=int, dest='max_timesteps', default=10e10,
                    help='The maximum number of timesteps for which to run the simulation.')
# parser.add_argument('--save-video', type=str, dest='video_save_file',
#                     help='Save the video in the specified file.')

args = parser.parse_args()
model_name = args.model_name
max_timesteps = args.max_timesteps

config_filename = 'config.yaml'

all_configs = load(open(config_filename, 'r'))
config = all_configs[model_name]

device = get_device()

env_name = config['env_name']
env = gym.make(env_name)
env._max_episode_steps = max_timesteps
action_space = env.action_space
observation_space = env.observation_space
policy_hidden_dims = config['policy_hidden_dims']
policy_args = (observation_space.shape[0], policy_hidden_dims, action_space.shape[0])

if type(action_space) is Box:
    policy = build_diag_gauss_policy(*policy_args)
elif type(action_space) is Discrete:
    policy = build_multinomial_policy(*policy_args)
else:
    raise NotImplementedError

session_dir = all_configs['session_save_dir']
load_path = os.path.join(session_dir, model_name + '.pt')

if device.type is 'cuda':
    ckpt = torch.load(load_path)
else:
    ckpt = torch.load(load_path, map_location='cpu')

policy.load_state_dict(ckpt['policy state dict'])
policy.to(device)
state_filter = ckpt['state_filter']

# Run the simulation
policy.eval()
state = env.reset()
done = False

while not done:
    state = torch.tensor(state).float()
    state = state_filter(state).to(device)

    action_dist = policy(state)
    action = action_dist.sample().cpu()

    state, reward, done, _ = env.step(action.numpy())
    env.render()
