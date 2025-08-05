import os
import sys
import pickle
import argparse
from me_utils import prep_for_me2, prep_for_me2_ordered, get_me_events

# parse data from pkl format to the format required for rl

parser = argparse.ArgumentParser()
parser.add_argument( "--in_pkl_file", help="input pkl file", type=str )
parser.add_argument( "--out_pkl_file", help="output pkl filename", type=str )
parser.add_argument( "--matrix2py_path", help="path to madgraph dir with matrix2py", type=str )
parser.add_argument( "--paramcard_path", help="path to madgraph dir with param card", type=str )
args = parser.parse_args()

data = {
    "orig":[],
    "rl":[],
    "orig_me":[],
    "rl_me":[]
}

print( "loading data" )
with open(args.in_pkl_file, 'rb') as handle:
    events = pickle.load( handle )

print( "getting events prepped" )
for event in events:
    event_prep = prep_for_me2( event )
    event_prep_ordered = prep_for_me2_ordered( event )
    data["orig"].append( event_prep )
    data["rl"].append( event_prep_ordered )

print( "getting MEs" )
data["orig_me"] = get_me_events( data["orig"], args.matrix2py_path, args.paramcard_path )
data["rl_me"] = get_me_events( data["rl"], args.matrix2py_path, args.paramcard_path )

print( "saving data" )
with open(args.out_pkl_file, 'wb') as handle:
    pickle.dump( data, handle )
