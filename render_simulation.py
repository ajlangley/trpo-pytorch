from argparse import ArgumentParser
import gym
import os
import roboschool
import torch
import numpy as np

from models import build_diag_gauss_policy, build_multinomial_policy
from torch_utils import get_device

parser = ArgumentParser(prog='render_simulation.py',
                        description='Render a simulation of an agent in the' \
                        ' specified environment.')
parser.add_argument('--env-name', type=str, dest='env_name', required=True,
                    help='The name of the OpenAI Gym environment to simulate.')
parser.add_argument('--policy-file', type=str, dest='agent_file', required=True,
                    help='The name of the file where the policy is stored.')
parser.add_argument('--save-video', type=str, dest='video_save_file',
                    help='Save the video in the specified file.')

args = parser.parse_args()

save_dir = 'saved-sessions'

def load_model_from_file(policy, model_name):
    load_path = os.path.join(save_dir, model_name + '.pt')
    if torch.cuda.is_available():
        ckpt = torch.load(load_path)
    else:
        ckpt = torch.load(load_path, map_location='cpu')

    policy.load_state_dict(ckpt['policy state dict'])

    state_filter = ckpt['state_filter']

    return state_filter

env = gym.make('Walker2d-v3')
# env = gym.make('LunarLanderContinuous-v2')
env._max_episode_steps = 10e10
policy = build_diag_gauss_policy(17, [96, 64, 32], 6)
# policy = build_diag_gauss_policy(8, [32], 2)
policy.to(get_device())
policy.eval()
state_filter = load_model_from_file(policy, 'walker-gae')
# load_model_from_file(policy, 'lunar-lander-cont')
state = env.reset()
done = False

while not done:
    state = torch.tensor(state).to(get_device()).float()
    state = state_filter(state.cpu())
    dist = policy(state.to(get_device()))
    action = dist.sample().cpu()
    state, reward, done, _ = env.step(action.numpy())
    env.render()
