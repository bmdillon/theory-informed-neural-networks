import os
import sys
import pickle
import argparse
from data_utils import read_lhe_events, boost_events_to_com_frame

# parse data from lhe gzipped format to pkl

parser = argparse.ArgumentParser()
parser.add_argument( "--lhe_file", help="gzipped lhe file", type=str )
parser.add_argument( "--out_file", help="output events pickle filename", type=str )
parser.add_argument( "--n_events", help="number of events to process", type=int )
args = parser.parse_args()

lhe_file = args.lhe_file
num_events = args.n_events
events = read_lhe_events( lhe_file, max_events=num_events )
events = boost_events_to_com_frame( events )
with open( args.out_file, 'wb' ) as handle:
    pickle.dump( events, handle )