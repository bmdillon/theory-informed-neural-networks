print('-- starting job', flush=True)

print('-- imports', flush=True)
import os
import sys
import time
import yaml
import shutil
import numpy as np
import argparse
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append("/users/bdillon/projects/tinn/")
from utils.matrix_element import get_me
from utils.replay_buffer import replay_buffer
from rl.dqn_trainer import DQNTrainer as dqn_trainer

# parsing args
print( '-- parsing arguments', flush=True )
parser = argparse.ArgumentParser( description="RL for PID assignment using the matrix element" )
parser.add_argument( "--runname", type=str, help="name of the run", required=True )
parser.add_argument( "--config", type=str, help="config filepath for run params", required=True )
parser.add_argument( "--save_path", type=str, help="filepath to save results", required=True )
parser.add_argument( "--data_path", type=str, help="", required=False )
parser.add_argument( "--reuse_actions", type=bool, help="", required=False )
parser.add_argument( "--matrix2py_path", type=str, help="", required=False )
parser.add_argument( "--paramcard_path", type=str, help="", required=False )
parser.add_argument( "--process", type=str, help="", required=False )
parser.add_argument( "--num_steps", type=int, help="", required=False )
parser.add_argument( "--sparse_rewards", type=str, help="", required=False )
parser.add_argument( "--d_input", type=int, help="", required=False )
parser.add_argument( "--d_model", type=int, help="", required=False )
parser.add_argument( "--num_heads", type=int, help="", required=False )
parser.add_argument( "--num_layers", type=int, help="", required=False )
parser.add_argument( "--position_embedding", type=bool, help="", required=False )
parser.add_argument( "--dropout", type=float, help="", required=False )
parser.add_argument( "--num_episodes", type=int, help="", required=False )
parser.add_argument( "--eval_every", type=int, help="", required=False )
parser.add_argument( "--test_frac", type=float, help="", required=False )
parser.add_argument( "--num_eval", type=int, help="", required=False )
parser.add_argument( "--gamma", type=float, help="", required=False )
parser.add_argument( "--learning_rate", type=float, help="", required=False )
parser.add_argument( "--batch_size", type=int, help="", required=False )
parser.add_argument( "--scale_reward", type=float, help="", required=False )
parser.add_argument( "--memory_size", type=int, help="", required=False )
parser.add_argument( "--target_update_freq", type=int, help="", required=False )
parser.add_argument( "--epsilon_start", type=float, help="", required=False )
parser.add_argument( "--epsilon_end", type=float, help="", required=False )
parser.add_argument( "--epsilon_decay", type=float, help="", required=False )
parser.add_argument( "--init_net", type=str, help="", required=False )

args = parser.parse_args()

# save args to save path
shutil.copy( args.config, args.save_path )

# config
config = {}
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
for key, value in config.items():
    if hasattr(args, key) and getattr(args, key) is None:
        setattr(args, key, value)
print( vars(args), flush=True )
missing = [ k for k in vars(args).keys() if getattr(args,k) is None ]
if missing:
    raise ValueError(f"Missing required YAML values: {missing}")

# load environment
if args.process == 'tt':
    from rl.environments import tt_env as rl_env
    num_partons = 8
elif args.process == 'wwuu00':
    from rl.environments import wwuu_env as rl_env
    num_partons = 8
elif args.process == 'wwuuT0':
    from rl.environments import wwuu_env as rl_env
    num_partons = 8
elif args.process == 'wwuu0T':
    from rl.environments import wwuu_env as rl_env
    num_partons = 8
elif args.process == 'wwuuTT':
    from rl.environments import wwuu_env as rl_env
    num_partons = 8
elif args.process == 'ttw':
    from rl.environments import ttw_env as rl_env
    num_partons = 10
elif args.process == 'tttt':
    from rl.environments import tttt_env as rl_env
    num_partons = 14
else:
    print( "no valid environment found", flush=True )
    sys.exit()


# print run name
print( '---', flush=True )
print( 'runname is ', args.runname, flush=True )
print( '---', flush=True )

# loading data
print( '-- loading data', flush=True )
with open( args.data_path, 'rb' ) as handle:
    events_rl = pickle.load( handle )
data_orig_me = events_rl['orig_me']
data = np.array( events_rl['rl'] )
num_events = data.shape[0]
num_tst = int( args.test_frac * num_events )
num_trn = num_events - num_tst
data_tst = data[-num_tst:]
data = data[:num_trn]
data_orig_me_tst = data_orig_me[-num_tst:]
data_orig_me = data_orig_me[0:num_trn]

# setting up environment
print( '-- setting up environment', flush=True )

# save out env params
env_params_save = {
    "dataset": '',
    "true_mes": '',
    "dataset_tst": '',
    "true_mes_tst": '',
    "reuse_actions": args.reuse_actions,
    "num_steps": args.num_steps,
    "sparse_rewards": args.sparse_rewards,
    "matrix2py_path": args.matrix2py_path,
    "paramcard_path": args.paramcard_path }
with open( args.save_path + '/env_params.json', 'w' ) as handle:
    json.dump( env_params_save, handle, ensure_ascii=False, indent=4 )

# init the env
env_params = {
    "dataset": data,
    "true_mes": data_orig_me,
    "dataset_tst": data_tst,
    "true_mes_tst": data_orig_me_tst,
    "reuse_actions": args.reuse_actions,
    "num_steps": args.num_steps,
    "sparse_rewards": args.sparse_rewards,
    "matrix2py_path": args.matrix2py_path,
    "paramcard_path": args.paramcard_path }
env = rl_env(**env_params)


# training
print( '-- training', flush=True )

#save out training params
train_params_save = {
    "d_input": args.d_input,
    "d_model": args.d_model,
    "num_heads": args.num_heads,
    "num_layers": args.num_layers,
    "d_output": env.num_actions,
    "num_partons": num_partons,
    "position_embedding": args.position_embedding,
    "dropout": args.dropout,
    "environment": args.process,
    "num_episodes": args.num_episodes,
    "eval_every": args.eval_every,
    "num_eval": args.num_eval,
    "gamma": args.gamma,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "scale_reward": args.scale_reward,
    "memory_size": args.memory_size,
    "target_update_freq": args.target_update_freq,
    "epsilon_start": args.epsilon_start,
    "epsilon_end": args.epsilon_end,
    "epsilon_decay": args.epsilon_decay,
    "save_path": args.save_path,
    "init_net": args.init_net
}
with open( args.save_path + '/train_params.json', 'w' ) as handle:
    json.dump( train_params_save, handle, ensure_ascii=False, indent=4 )

# run the dqn trainer
train_params = {
    "d_input": args.d_input,
    "d_model": args.d_model,
    "num_heads": args.num_heads,
    "num_layers": args.num_layers,
    "d_output": env.num_actions,
    "num_partons": num_partons,
    "position_embedding": args.position_embedding,
    "dropout": args.dropout,
    "environment": env,
    "num_episodes": args.num_episodes,
    "eval_every": args.eval_every,
    "num_eval": args.num_eval,
    "gamma": args.gamma,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "scale_reward": args.scale_reward,
    "memory_size": args.memory_size,
    "target_update_freq": args.target_update_freq,
    "epsilon_start": args.epsilon_start,
    "epsilon_end": args.epsilon_end,
    "epsilon_decay": args.epsilon_decay,
    "save_path": args.save_path,
    "init_net": args.init_net
}
trainer = dqn_trainer( **train_params )
trainer.train_model()

# training done, output the transformed events using the final net
print( '-- training done, creating and saving transformed events', flush=True )
trainer.test_transform_events( args.save_path+'/test_events.npy' )
print( '-- ALL DONE', flush=True )

