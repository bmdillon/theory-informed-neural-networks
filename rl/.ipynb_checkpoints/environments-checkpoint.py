import os
import sys
import numpy as np
import torch
import torch.nn as nn
import random
from importlib import import_module

# environment class for ttbar
class tt_env:
    def __init__(self, dataset, true_mes, dataset_tst, true_mes_tst, num_steps, sparse_rewards,  matrix2py_path, paramcard_path, reuse_actions=True ):
        self.dataset = dataset
        self.true_mes = true_mes
        self.dataset_tst = dataset_tst
        self.true_mes_tst = true_mes_tst
        self.num_steps = num_steps
        self.sparse_rewards = sparse_rewards
        self.matrix2py_path = matrix2py_path 
        self.paramcard_path = paramcard_path
        self.reuse_actions = reuse_actions

        # init env params
        self.done = False
        self.state = None
        self.new_score = 0.0

        # load matrix element module
        sys.path.append( self.matrix2py_path )
        self.matrix2py = import_module("matrix2py")
        self.matrix2py.initialisemodel( self.paramcard_path )

        # set num partons
        self.num_partons = 8

        # compute num swap actions
        self.swap_actions = [(i, j) for i in range(2, self.num_partons) for j in range(i + 1, self.num_partons)]
        self.swap_actions.append(None)
        self.num_actions = len( self.swap_actions )
        self.graph_masses = {}

    # reset function, return new sampled state
    def reset(self):
        self.idx = np.random.randint(0, len(self.dataset))
        self.state = self.dataset[self.idx].copy()
        self.initial_score = self.score( self.state )
        self.done = False
        self.current_step = 0
        self.update_graph_masses()
        self.action_list = []
        self.action_mask = torch.ones( self.num_actions ).reshape(1,-1)
        return self.state.copy()

    # reset function for testing
    def reset_tst(self):
        self.idx = np.random.randint(0, len(self.dataset_tst))
        self.state = self.dataset_tst[self.idx].copy()
        self.initial_score = self.score( self.state )
        self.done = False
        self.current_step = 0
        self.update_graph_masses()
        self.action_list = []
        self.action_mask = torch.ones( self.num_actions ).reshape(1,-1)
        return self.state.copy()

    # compute matrix element score
    def score(self, state):
        particles = state
        alphas = 0.13
        nhel = -1 # means sum over all helicity     
        me2 = self.matrix2py.get_value(particles, alphas, nhel)
        return me2

    # compute step in the MDP
    def step(self, action):
        # keep track of actions
        self.action_list.append( action )
        
        # check that MDP is not over
        if self.done:
            raise RuntimeError("Step called after done=True")

        # the 'do-nothing' action
        if action == self.num_actions-1:
            reward = 0.0
        # a swap action
        else:
            # apply action-mask if we're not re-using actions 
            if not self.reuse_actions:
                self.action_mask[0][ action ] = 0.0
            # get current / previous score
            self.previous_score = self.score( self.state )
            # get indices for action
            i, j = self.swap_actions[action]
            # implement indices swap to get new state
            particles = self.state.copy()
            particles[:, [i, j]] = particles[:, [j, i]]
            new_state = particles
            self.state = new_state
            # get new score
            self.new_score = self.score( self.state )
            # sparse rewards means we only reward at the end
            if self.sparse_rewards:
                reward = 0.0
            else:
                reward = np.log( self.new_score / self.previous_score )

        # check if we're at the last step, compute final score, set 'done' flag to True
        if self.current_step == self.num_steps-1:
            self.final_score = self.new_score
            if self.sparse_rewards:
                reward = np.log( self.final_score / self.initial_score )
            self.done = True
        else:
            self.current_step += 1

        # update graph masses
        self.update_graph_masses()

        # return
        return self.state.copy(), reward, self.new_score, self.done

    # compute a random action
    def random_action(self):
        return random.randint(0, self.num_actions - 1)

    # get state from dataset
    def get_state(self, idx):
        return self.dataset[idx].copy()

    # get state from test dataset
    def get_state_tst(self, idx):
        return self.dataset_tst[idx].copy()

    # get true matrix element from dataset
    def get_true_me(self, idx):
        if self.true_mes:
            return self.true_mes[idx]
        else:
            return 0.0

    # get true matrix element from test dataset
    def get_true_me_tst(self, idx):
        if self.true_mes_tst:
            return self.true_mes_tst[idx]
        else:
            return 0.0
    # get true matrix element of the current sampled state in dataset
    def get_current_true_me(self):
        if self.true_mes:
            return self.true_mes[self.idx]
        else:
            return 0.0
    
    # get true matrix element of the current sampled state in test dataset
    def get_current_true_me_tst(self):
        if self.true_mes_tst:
            return self.true_mes_tst[self.idx]
        else:
            return 0.0

    # update masses in graph
    def update_graph_masses(self):
        rm = { "b":2, "u":3, "dbar":4, "bbar":5, "ubar":6, "d":7 }
        p = self.state[0:4,:].copy()
        p_b = p[ :, rm['b'] ]
        p_u = p[ :, rm['u'] ]
        p_dbar = p[ :, rm['dbar'] ]
        p_bbar = p[ :, rm['bbar'] ]
        p_ubar = p[ :, rm['ubar'] ]
        p_d = p[ :, rm['d'] ]
        p_wp = [ i+j for i,j in zip(p_u,p_dbar) ]
        p_wm = [ i+j for i,j in zip(p_ubar,p_d) ]
        p_t = [ i+j for i,j in zip(p_b,p_wp) ]
        p_tbar = [ i+j for i,j in zip(p_bbar,p_wm) ]
        mwp = np.sqrt( p_wp[0]**2 - p_wp[1]**2 - p_wp[2]**2 - p_wp[3]**2 )
        mwm = np.sqrt( p_wm[0]**2 - p_wm[1]**2 - p_wm[2]**2 - p_wm[3]**2 )
        mt = np.sqrt( p_t[0]**2 - p_t[1]**2 - p_t[2]**2 - p_t[3]**2 )
        mtb = np.sqrt( p_tbar[0]**2 - p_tbar[1]**2 - p_tbar[2]**2 - p_tbar[3]**2 )
        self.graph_masses = { "mt":mt, "mtb":mtb, "mw+":mwp, "mw-":mwm }

# environment class for wwuu
class wwuu_env(tt_env):
    def __init__(self, dataset, true_mes, dataset_tst, true_mes_tst, num_steps, sparse_rewards,  matrix2py_path, paramcard_path, reuse_actions=True ):
        self.dataset = dataset
        self.true_mes = true_mes
        self.dataset_tst = dataset_tst
        self.true_mes_tst = true_mes_tst
        self.num_steps = num_steps
        self.sparse_rewards = sparse_rewards
        self.matrix2py_path = matrix2py_path 
        self.paramcard_path = paramcard_path
        self.reuse_actions = reuse_actions
        
        self.done = False
        self.state = None

        sys.path.append( self.matrix2py_path )
        self.matrix2py = import_module("matrix2py")
        self.matrix2py.initialisemodel( self.paramcard_path )
        
        self.num_partons = 8

        self.swap_actions = [(i, j) for i in range(2, self.num_partons) for j in range(i + 1, self.num_partons)]
        self.swap_actions.append(None)
        self.num_actions = len( self.swap_actions )
        self.graph_masses = {}

    def update_graph_masses(self):
        rm = { "u":2, "dbar":3, "ubar":4, "d":5, "u1":6, "u2":7 }
        p = self.state[0:4,:].copy()
        p_u = p[ :, rm['u'] ]
        p_dbar = p[ :, rm['dbar'] ]
        p_ubar = p[ :, rm['ubar'] ]
        p_d = p[ :, rm['d'] ]
        p_wp = [ i+j for i,j in zip(p_u,p_dbar) ]
        p_wm = [ i+j for i,j in zip(p_ubar,p_d) ]
        mwp = np.sqrt( p_wp[0]**2 - p_wp[1]**2 - p_wp[2]**2 - p_wp[3]**2 )
        mwm = np.sqrt( p_wm[0]**2 - p_wm[1]**2 - p_wm[2]**2 - p_wm[3]**2 )
        self.graph_masses = { "mw+":mwp, "mw-":mwm }

# environment class for ttw
class ttw_env(tt_env):
    def __init__(self, dataset, true_mes, dataset_tst, true_mes_tst, num_steps, sparse_rewards,  matrix2py_path, paramcard_path, reuse_actions=True ):
        self.dataset = dataset
        self.true_mes = true_mes
        self.dataset_tst = dataset_tst
        self.true_mes_tst = true_mes_tst
        self.num_steps = num_steps
        self.sparse_rewards = sparse_rewards
        self.matrix2py_path = matrix2py_path 
        self.paramcard_path = paramcard_path
        self.reuse_actions = reuse_actions
        
        self.done = False
        self.state = None

        sys.path.append( self.matrix2py_path )
        self.matrix2py = import_module("matrix2py")
        self.matrix2py.initialisemodel( self.paramcard_path )
        
        self.num_partons = 10

        self.swap_actions = [(i, j) for i in range(2, self.num_partons) for j in range(i + 1, self.num_partons)]
        self.swap_actions.append(None)
        self.num_actions = len( self.swap_actions )
        self.graph_masses = {}

    def update_graph_masses(self):
        rm = { "b":2, "u":3, "dbar":4, "bbar":5, "ubar":6, "d":7, "u2":8, "dbar2":9 }
        p = self.state[0:4,:].copy()
        p_b = p[ :, rm['b'] ]
        p_u = p[ :, rm['u'] ]
        p_dbar = p[ :, rm['dbar'] ]
        p_bbar = p[ :, rm['bbar'] ]
        p_ubar = p[ :, rm['ubar'] ]
        p_d = p[ :, rm['d'] ]
        p_u2 = p[ :, rm['u2'] ]
        p_dbar2 = p[ :, rm['dbar2'] ]
        p_wp = [ i+j for i,j in zip(p_u,p_dbar) ]
        p_wm = [ i+j for i,j in zip(p_ubar,p_d) ]
        p_t = [ i+j for i,j in zip(p_b,p_wp) ]
        p_tbar = [ i+j for i,j in zip(p_bbar,p_wm) ]
        p_wp2 = [ i+j for i,j in zip(p_u2,p_dbar2) ]
        mwp = np.sqrt( p_wp[0]**2 - p_wp[1]**2 - p_wp[2]**2 - p_wp[3]**2 )
        mwm = np.sqrt( p_wm[0]**2 - p_wm[1]**2 - p_wm[2]**2 - p_wm[3]**2 )
        mt = np.sqrt( p_t[0]**2 - p_t[1]**2 - p_t[2]**2 - p_t[3]**2 )
        mtb = np.sqrt( p_tbar[0]**2 - p_tbar[1]**2 - p_tbar[2]**2 - p_tbar[3]**2 )
        mwp2 = np.sqrt( p_wp2[0]**2 - p_wp2[1]**2 - p_wp2[2]**2 - p_wp2[3]**2 )
        self.graph_masses = { "mt":mt, "mtb":mtb, "mw+":mwp, "mw-":mwm, "mw+2":mwp2 }

# environment class for tttt
class tttt_env(tt_env):
    def __init__(self, dataset, true_mes, dataset_tst, true_mes_tst, num_steps, sparse_rewards,  matrix2py_path, paramcard_path, reuse_actions=True ):
        self.dataset = dataset
        self.true_mes = true_mes
        self.dataset_tst = dataset_tst
        self.true_mes_tst = true_mes_tst
        self.num_steps = num_steps
        self.sparse_rewards = sparse_rewards
        self.matrix2py_path = matrix2py_path 
        self.paramcard_path = paramcard_path
        self.reuse_actions = reuse_actions
        
        self.done = False
        self.state = None

        sys.path.append( self.matrix2py_path )
        self.matrix2py = import_module("matrix2py")
        self.matrix2py.initialisemodel( self.paramcard_path )
        
        self.num_partons = 14

        self.swap_actions = [(i, j) for i in range(2, self.num_partons) for j in range(i + 1, self.num_partons)]
        self.swap_actions.append(None)
        self.num_actions = len( self.swap_actions )
        self.graph_masses = {}

    def update_graph_masses(self):
        rm = { "b":2, "u":3, "dbar":4, "bbar":5, "ubar":6, "d":7, "b2":8, "u2":9, "dbar2":10, "bbar2":11, "ubar2":12, "d2":13 }
        p = self.state[0:4,:].copy()
        p_b = p[ :, rm['b'] ]
        p_u = p[ :, rm['u'] ]
        p_dbar = p[ :, rm['dbar'] ]
        p_bbar = p[ :, rm['bbar'] ]
        p_ubar = p[ :, rm['ubar'] ]
        p_d = p[ :, rm['d'] ]
        p_b2 = p[ :, rm['b2'] ]
        p_u2 = p[ :, rm['u2'] ]
        p_dbar2 = p[ :, rm['dbar2'] ]
        p_bbar2 = p[ :, rm['bbar2'] ]
        p_ubar2 = p[ :, rm['ubar2'] ]
        p_d2 = p[ :, rm['d2'] ]
        p_wp = [ i+j for i,j in zip(p_u,p_dbar) ]
        p_wm = [ i+j for i,j in zip(p_ubar,p_d) ]
        p_t = [ i+j for i,j in zip(p_b,p_wp) ]
        p_tbar = [ i+j for i,j in zip(p_bbar,p_wm) ]
        p_wp2 = [ i+j for i,j in zip(p_u2,p_dbar2) ]
        p_wm2 = [ i+j for i,j in zip(p_ubar2,p_d2) ]
        p_t2 = [ i+j for i,j in zip(p_b2,p_wp2) ]
        p_tbar2 = [ i+j for i,j in zip(p_bbar2,p_wm2) ]
        mwp = np.sqrt( p_wp[0]**2 - p_wp[1]**2 - p_wp[2]**2 - p_wp[3]**2 )
        mwm = np.sqrt( p_wm[0]**2 - p_wm[1]**2 - p_wm[2]**2 - p_wm[3]**2 )
        mt = np.sqrt( p_t[0]**2 - p_t[1]**2 - p_t[2]**2 - p_t[3]**2 )
        mtb = np.sqrt( p_tbar[0]**2 - p_tbar[1]**2 - p_tbar[2]**2 - p_tbar[3]**2 )
        mwp2 = np.sqrt( p_wp2[0]**2 - p_wp2[1]**2 - p_wp2[2]**2 - p_wp2[3]**2 )
        mwm2 = np.sqrt( p_wm2[0]**2 - p_wm2[1]**2 - p_wm2[2]**2 - p_wm2[3]**2 )
        mt2 = np.sqrt( p_t2[0]**2 - p_t2[1]**2 - p_t2[2]**2 - p_t2[3]**2 )
        mtb2 = np.sqrt( p_tbar2[0]**2 - p_tbar2[1]**2 - p_tbar2[2]**2 - p_tbar2[3]**2 )
        self.graph_masses = { "mt":mt, "mtb":mtb, "mw+":mwp, "mw-":mwm,"mt2":mt2, "mtb2":mtb2, "mw+2":mwp2, "mw-2":mwm2 }


# environment class for polarised wwuu
class wwuu_pol_env(tt_env):
    def __init__(self, dataset, true_mes, dataset_tst, true_mes_tst, num_steps, sparse_rewards, matrix2py_00_path, paramcard_00_path, matrix2py_0T_path, paramcard_0T_path, matrix2py_T0_path, paramcard_T0_path, matrix2py_TT_path, paramcard_TT_path, reuse_actions=True ):
        self.dataset = dataset
        self.true_mes = true_mes
        self.dataset_tst = dataset_tst
        self.true_mes_tst = true_mes_tst
        self.num_steps = num_steps
        self.sparse_rewards = sparse_rewards
        self.matrix2py_00_path = matrix2py_00_path 
        self.paramcard_00_path = paramcard_00_path
        self.matrix2py_T0_path = matrix2py_T0_path 
        self.paramcard_T0_path = paramcard_T0_path
        self.matrix2py_0T_path = matrix2py_0T_path 
        self.paramcard_0T_path = paramcard_0T_path
        self.matrix2py_TT_path = matrix2py_TT_path 
        self.paramcard_TT_path = paramcard_TT_path
        self.reuse_actions = reuse_actions
        
        self.done = False
        self.state = None

        self.matrix2py_paths = {'00':self.matrix2py_00_path, '0T':self.matrix2py_0T_path, 'T0':self.matrix2py_T0_path, 'TT':self.matrix2py_TT_path }
        self.paramcard_paths = {'00':self.paramcard_00_path, '0T':self.paramcard_0T_path, 'T0':self.paramcard_T0_path, 'TT':self.paramcard_TT_path }
        
        self.matrix2py_dict = {}
        for pk in ['00','0T','T0','TT']:
            sys.path.append( self.matrix2py_paths[pk] )
            self.matrix2py_dict[pk] = import_module("matrix2py")
            self.matrix2py_dict[pk].initialisemodel( self.paramcard_paths[pk] )
        
        self.num_partons = 8

        self.swap_actions = [(i, j) for i in range(2, self.num_partons) for j in range(i + 1, self.num_partons)]
        self.swap_actions.append(None)
        self.num_actions = len( self.swap_actions )
        self.graph_masses = {}

    def update_graph_masses(self):
        rm = { "u":2, "dbar":3, "ubar":4, "d":5, "d1":6, "d2":7 }
        p = self.state[0:4,:].copy()
        p_u = p[ :, rm['u'] ]
        p_dbar = p[ :, rm['dbar'] ]
        p_ubar = p[ :, rm['ubar'] ]
        p_d = p[ :, rm['d'] ]
        p_wp = [ i+j for i,j in zip(p_u,p_dbar) ]
        p_wm = [ i+j for i,j in zip(p_ubar,p_d) ]
        mwp = np.sqrt( p_wp[0]**2 - p_wp[1]**2 - p_wp[2]**2 - p_wp[3]**2 )
        mwm = np.sqrt( p_wm[0]**2 - p_wm[1]**2 - p_wm[2]**2 - p_wm[3]**2 )
        self.graph_masses = { "mw+":mwp, "mw-":mwm }


    def score(self, state):
        particles = state
        alphas = 0.13
        nhel = -1 # means sum over all helicity     
        me2 = 0.0
        for pk in ['00','0T','T0','TT']:
            me2 += self.matrix2py_dict[pk].get_value(particles, alphas, nhel) / 4
        return me2
