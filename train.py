from argparse import ArgumentParser
import gym
from yaml import load

from models import build_diag_gauss_policy, build_mlp, build_multinomial_policy
from simulators import *
from transforms import *
from torch_utils import get_device
from trpo import TRPO

config_filename = 'config.yaml'

parser = ArgumentParser(prog='train.py',
                        description='Train a policy on the specified environment' \
                        ' using Trust Region Policy Optimization (Schulman 2015)' \
                        ' with Generalized Advantage Estimation (Schulman 2016).')
parser.add_argument('--continue', dest='continue_from_file', action='store_true',
                    help='Set this flag to continue training from a previously ' \
                    'saved session')
parser.add_argument('--model-name', type=str, dest='model_name', required=True,
                    help='The entry in trpo_experiments.yaml from which settings' \
                    'should be loaded.')
parser.add_argument('--simulator', dest='simulator_type', type=str, default='single-path',
                    choices=['single-path', 'vine'], help='The type of simulator' \
                    ' to use when collecting training experiences.')

args = parser.parse_args()
continue_from_file = args.continue_from_file
model_name = args.model_name
simulator_type = args.simulator_type

all_configs = load(open(config_filename, 'r'))
config = all_configs[model_name]

device = get_device()

# Find the input size, hidden dim sizes, and output size
env = gym.make(config['env_name'])
action_space = env.action_space
observation_space = env.observation_space
policy_hidden_dims = config['policy_hidden_dims']
vf_hidden_dims = config['vf_hidden_dims']
policy_args = (observation_space.shape[0], policy_hidden_dims, action_space.shape[0])
vf_args = (observation_space.shape[0] + 1, vf_hidden_dims, 1)

# Initialize the policy and value function
if config['policy_type'] == 'gaussian':
    policy = build_diag_gauss_policy(*policy_args)
elif config['policy_type'] == 'multinomial':
    policy = build_multinomial_policy(*policy_args)

policy.to(device)
value_fun = build_mlp(*vf_args)
value_fun.to(device)

# Initialize the state transformation
z_filter = ZFilter()
state_bound = Bound(-5, 5)
state_filter = Transform(state_bound, z_filter)

# Initialize the simulator
env_name = config['env_name']
n_trajectories = config['n_trajectories']
max_timesteps = config['max_timesteps']
try:
    env_args = config['env_args']
except:
    env_args = {}

if simulator_type == 'single-path':
    simulator = SinglePathSimulator(env_name, policy, n_trajectories,
                                    max_timesteps, state_filter=state_filter,
                                    **env_args)
elif simulator_type == 'vine':
    raise NotImplementedError

try:
    trpo_args = config['trpo_args']
except:
    trpo_args = {}

trpo = TRPO(policy, value_fun, simulator, model_name=model_name,
            continue_from_file=continue_from_file, **trpo_args)

print(f'Training policy {model_name} on {env_name} environment...\n')

trpo.train(config['n_episodes'])

print('\nTraining complete.\n')
