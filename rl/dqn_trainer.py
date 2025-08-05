import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import vector
import random
import pickle
from collections import deque
from utils.ml import transformer_net as tfrmr
from utils.replay_buffer import replay_buffer

class DQNTrainer:
    def __init__(self, d_input, d_model, num_heads, num_layers, d_output, num_partons, position_embedding, dropout, environment, num_episodes, eval_every, num_eval, gamma, learning_rate, batch_size, scale_reward, memory_size, target_update_freq, epsilon_start, epsilon_end, epsilon_decay, save_path, init_net=None):

        # transformer
        self.d_input = d_input
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_output = d_output
        self.num_partons = num_partons
        self.position_embedding = position_embedding
        self.dropout = dropout

        # environment
        self.environment = environment
        self.num_episodes = num_episodes

        # rl
        self.scale_reward = scale_reward
        self.gamma = gamma
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # training
        self.eval_every = eval_every
        self.num_eval = num_eval
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_path = save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # define nets
        self.policy_net = tfrmr( self.d_input, self.d_model, self.num_heads, self.num_layers, self.d_output, self.num_partons, position_embedding=self.position_embedding, dropout=self.dropout )
        self.target_net = tfrmr( self.d_input, self.d_model, self.num_heads, self.num_layers, self.d_output, self.num_partons, position_embedding=self.position_embedding, dropout=self.dropout )
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # init net from saved model
        if init_net:
            self.policy_net.load_state_dict(torch.load( init_net, map_location=torch.device(self.device) ))
            print( f'weights loaded from {init_net}', flush=True )
        self.target_net.load_state_dict( self.policy_net.state_dict())
        
        # set target net to eval
        self.target_net.eval()

        # print device and network architecture
        print( '-- device ' + str(self.device), flush=True )
        summary( self.policy_net )

        # define optimizer
        self.optimizer = optim.Adam( self.policy_net.parameters(), lr=self.learning_rate)

        # set up replay buffers
        self.negative_memory_1 = replay_buffer(self.memory_size//4)
        self.negative_memory_2 = replay_buffer(self.memory_size//4)
        self.positive_memory_1 = replay_buffer(self.memory_size//4)
        self.positive_memory_2 = replay_buffer(self.memory_size//4)
        self.sample_size = self.batch_size//4

        # rl exploration
        self.epsilon = self.epsilon_start
        self.notify_training_started = False
        self.i_nts = 0

        # init performance tracking lists
        self.gains = []
        self.accs = []
        self.eval_gains = []
        self.eval_accs = []

    def train_model(self):
    
        for episode in range(self.num_episodes):

            # reset env and get new state
            state = self.environment.reset()

            # init the 'done' flag
            done = False

            # loop over actions
            while not done:
                # random action -> exploration (only explore every second episode)
                if random.random() < self.epsilon and episode%2 == 1:
                    action = self.environment.random_action()
                # else get the action from the policy net forward pass
                else:
                    with torch.no_grad():
                        # shape : (1, -, num_partons)
                        state_tensor = torch.tensor( state, dtype=torch.float32 ).unsqueeze(0).to(self.device)
                        # q-values : (1, num_actions)
                        q_values = self.policy_net( state_tensor )
                        # action mask
                        q_values[self.environment.action_mask == 0.0] = -np.inf
                        # action : int
                        action = q_values.argmax().item()

                # update the environment with the action
                next_state, reward, _, done = self.environment.step( action )

                # scale reward if needed
                reward = self.scale_reward * reward

                # update the replay buffers
                if reward <= 0:
                    if reward < -10 * self.scale_reward:
                        self.negative_memory_2.push( state, action, reward, next_state, done )
                    else:
                        self.negative_memory_1.push( state, action, reward, next_state, done )
                if reward > 0:
                    if reward > 10 * self.scale_reward:
                        self.positive_memory_2.push( state, action, reward, next_state, done )
                    else:
                        self.positive_memory_1.push( state, action, reward, next_state, done )

                # replace current state with next state
                state = next_state

                # if there is enough samples in the buffers, notify that training has started
                if self.notify_training_started and self.i_nts==0:
                    self.i_nts+=1
                    print(" ")
                    print(" ")
                    print(" ")
                    print("--------------- training started ---------------")
                    print(" ")
                    print(" ")
                    print(" ")

                # training loop
                # get buffer sizes
                lnm1 = len( self.negative_memory_1 )
                lnm2 = len( self.negative_memory_2 )
                lpm1 = len( self.positive_memory_1 )
                lpm2 = len( self.positive_memory_2 )

                # if there's enough samples in the buffer, train
                if ( lnm1 >= self.sample_size ) and ( lpm1 >= self.sample_size ) and ( lnm2 >= self.sample_size ) and ( lpm2 >= self.sample_size ):
                    # update the training_started flag
                    self.notify_training_started = True
                    # sample states from the buffer
                    sn1, an1, rn1, nsn1, dn1 = self.negative_memory_1.sample(self.sample_size)
                    sp1, ap1, rp1, nsp1, dp1 = self.positive_memory_1.sample(self.sample_size)
                    sn2, an2, rn2, nsn2, dn2 = self.negative_memory_2.sample(self.sample_size)
                    sp2, ap2, rp2, nsp2, dp2 = self.positive_memory_2.sample(self.sample_size)
                    s = torch.concat( [sn1, sp1, sn2, sp2] )
                    a = torch.concat( [an1, ap1, an2, ap2] )
                    r = torch.concat( [rn1, rp1, rn2, rp2] )
                    ns = torch.concat( [nsn1, nsp1, nsn2, nsp2] )
                    d = torch.concat( [dn1, dp1, dn2, dp2] )
                    s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)
                    # get q-values for sampled states using policy net
                    q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
                    # get target net predictions (no gradients)
                    with torch.no_grad():
                        next_q = self.target_net(ns).max(1)[0]
                        target = r + self.gamma * next_q * (1 - d)
                    # compute loss and update policy net weights
                    loss = nn.MSELoss()(q_values, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
            # update target net
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict( self.policy_net.state_dict() )

            # compute gains when not exploring
            if episode%2 == 0:
                gain = np.log(self.environment.final_score / self.environment.initial_score)
                self.gains.append( gain )
                true_me = self.environment.get_current_true_me()
                acc = np.log( self.environment.final_score/true_me )
                self.accs.append( acc )

            # eval time
            if (episode+1)%self.eval_every == 0 and episode>0:
                print("-- eval tst", flush=True)
                scores, best_scores, true_scores = self.eval( num_steps=self.environment.num_partons-1 )
                eval_gain = np.log( scores/true_scores ).mean()
                eval_gain_best = np.log( best_scores/true_scores ).mean()
                eval_acc = ((scores - true_scores) >= 0).mean()
                eval_acc_best = ((best_scores - true_scores) >= 0).mean()
                self.eval_gains.append( eval_gain )
                self.eval_accs.append( eval_acc )
                print( f"-- gain: {eval_gain}, -- best gain: {eval_gain_best}, acc: {eval_acc}, acc_best: {eval_acc_best}", flush=True )
                np.save( self.save_path + "/eval_"+str(episode)+"_best_scores.npy", best_scores, allow_pickle=True )
                np.save( self.save_path + "/eval_"+str(episode)+"_true_scores.npy", true_scores, allow_pickle=True )
                self.policy_net.save( self.save_path + "/policy_net.pth" )

            # save stats
            if episode%1000 == 0:
                np.save( self.save_path+'/gains.npy', np.array(self.gains), allow_pickle=True )
                np.save( self.save_path+'/accs.npy', np.array(self.accs), allow_pickle=True )
                print(f"episode {episode}, initial score: {self.environment.initial_score}, final score: {self.environment.final_score}, gain: {gain}, gains (running-ave-1000): {np.mean(self.gains[-1000:])}, test_acc: {acc}, accs (running-ave-1000): {np.mean(self.accs[-1000:])}, step: {self.environment.current_step}, epsilon: {self.epsilon:.2f}", flush=True)
                graph_masses = ""
                for key,value in self.environment.graph_masses.items():
                    graph_masses += key + ": " + str(value) + " "
                print(graph_masses, flush=True)

    # eval function
    def eval(self, num_steps=10):
        rl_scores = []
        best_rl_scores = []
        true_scores = []
        for i in range(len(self.environment.dataset_tst)):
            state = self.environment.reset_tst()
            scores = []
            with torch.no_grad():
                for step in range(num_steps):
                    state_tensor = torch.tensor( state, dtype=torch.float32 ).unsqueeze(0).to(self.device)
                    q_values = self.policy_net( state_tensor )
                    # action mask
                    q_values[self.environment.action_mask == 0.0] = -np.inf
                    action = q_values.argmax().item()
                    next_state, reward, new_score, done = self.environment.step( action )
                    state = next_state
                    scores.append( new_score )
                    if done:
                        break
            rl_scores.append( scores[-1] )
            best_rl_scores.append( max( scores ) )
            true_scores.append( self.environment.get_current_true_me_tst() )
        return np.array(rl_scores), np.array(best_rl_scores), np.array(true_scores)

    # transform events
    def test_transform_events(self, save_states_file, num_events='all'):
        if num_events == 'all':
            num_events = len( self.environment.dataset )
        transformed_states = []
        for idx in range(num_events):
            print( "event "+str(idx), flush=True )
            state = self.environment.reset()
            #state = self.environment.get_state( idx )
            #self.environment.state = state.copy()
            #self.environment.initial_score = self.environment.score( state )
            #self.environment.current_step = 0
            #self.environment.done = False
            done = False
            while not done:
                with torch.no_grad():
                    # shape : (1, -, num_partons)
                    state_tensor = torch.tensor( state, dtype=torch.float32 ).unsqueeze(0).to(self.device)
                    # q-values : (1, num_actions)
                    q_values = self.policy_net( state_tensor )
                    # action mask
                    q_values[self.environment.action_mask == 0.0] = -np.inf
                    # action : int
                    action = q_values.argmax().item()
                next_state, reward, _, done = self.environment.step( action )
                state = next_state
                #print( "reward: "+str(reward), flush=True )
            if done:
                transformed_states.append( state )
        transformed_states = np.array( transformed_states )
        np.save( save_states_file, transformed_states, allow_pickle=True )
            

    # load saved weights
    def load_net(self, model_path):
        if self.device == 'gpu':
            self.policy_net.load_state_dict( torch.load( model_path ) )
            self.target_net.load_state_dict( torch.load( model_path ) )
        else:
            self.policy_net.load_state_dict( torch.load( model_path, map_location=torch.device('cpu') ) )
            self.target_net.load_state_dict( torch.load( model_path, map_location=torch.device('cpu') ) )
        
