import os
from REINFORCE.policy import Policy
from REINFORCE.critic import Critic
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import socket
import math
import json
import config

from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size, lr=5e-5, critic_lr=1e-3, gamma=.99, entropy_beta=1e-2, annotations=None):

        self.lr = lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        
        self.policy = Policy(state_size=state_size, action_size=action_size).to(device)
        self.critic = Critic(state_size=state_size).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=1e-5)
        
        self.annotations = annotations
        self.creation_timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        self.curr_epoch = 0

        self.agent_id = None
        self.writer = None
        self.rewards = None

    def train(self, envs, dataset_name, n_epochs, batch_size=16, model_path=None):
        self.policy.train()
        self.critic.train()

        exp_keys = np.array(list(envs.keys()))

        hostname = socket.gethostname()
        
        self.agent_id = '{}_{}_{}_{}{}{}lr{}_gm{}_lambda{}_eps{}'.format(self.creation_timestamp, hostname, dataset_name.lower(), 
                                                                           'vdan_' if envs[exp_keys[0]].use_vdan else 'vdan+_', 
                                                                           'no_positional_encoding_' if envs[exp_keys[0]].no_positional_encoding else '', 
                                                                           'no_skip_info_' if envs[exp_keys[0]].no_skip_info else '', 
                                                                           self.lr, self.gamma, envs[exp_keys[0]].lambda_param, n_epochs)
        
        self.writer = SummaryWriter(log_dir=f'{config.logs_base_folder}/{self.agent_id}/', filename_suffix=f'_{self.creation_timestamp}')

        print(f'\n[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}] Training will be performed on: {device}')
        print(f'\n[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}] Agent ID: {self.agent_id}\n')

        # Writing Frames Dots Fig.
        for exp_name, env in envs.items():
            self.write_states_and_dots_to_log(env, dataset_name, exp_name)

        #### Running REINFORCE ####
        Rt = {exp_key: [] for exp_key in envs.keys()}
        splits_idxs = [batch_size*i for i in range(1, math.ceil(len(exp_keys)/batch_size))]
        for i_epoch in tqdm(range(self.curr_epoch, n_epochs), desc='Epochs'):

            epoch_speedups_deviation = []
            epoch_losses = []
            critic_losses = []
            entropies_epoch = []

            # Shuffle the experiments at every new epoch
            np.random.shuffle(exp_keys)

            keys_batches = np.split(exp_keys, splits_idxs)
            
            for keys_batch in tqdm(keys_batches, desc='Batch', leave=False): # Run an episode for each environment
                curr_batch_size = len(keys_batch)

                # Init batch
                envs_batch = [] # Batch of environments (we used batch_size=1)
                states_batch = [] # Batch of states
                state_values_batch = []
                log_probs_batch = []
                entropies_batch = []
                rewards_batch = np.array([None]*curr_batch_size)
                
                states_batches = []
                for exp_idx, exp_key in enumerate(keys_batch): # Reset all environments in the batch
                    env = envs[exp_key]
                    
                    state = env.reset()

                    envs_batch.append(env)
                    states_batch.append(state)
                
                    rewards_batch[exp_idx] = []
      
                ## Main LOOP (Episode run)
                done_batch = [False for _ in range(curr_batch_size)] # Batch is only done when every environment is already done
                while not all(done_batch):
                    states_batches.append(states_batch)
                    states_batches[-1] = torch.stack(states_batches[-1]).to(device)
                    action_batch, log_prob_batch, entropy_batch, _ = self.policy.act(states_batches[-1])

                    state_values_batch.append(self.critic.criticize(states_batches[-1])) # δ(st|θδ)
                    log_probs_batch.append(log_prob_batch)
                    entropies_batch.append(entropy_batch)

                    states_batch = []
                    for exp_idx in range(curr_batch_size):
                        state, reward, done_batch[exp_idx], _ = envs_batch[exp_idx].step(action_batch[exp_idx].item())

                        states_batch.append(state)
                        rewards_batch[exp_idx].append(reward)
                
                num_overall_steps = len(rewards_batch[0])
                state_values_batch = torch.stack(state_values_batch).squeeze(3)
                log_probs_batch = torch.stack(log_probs_batch)
                entropies_batch = torch.stack(entropies_batch)
                rewards_batch = np.asarray([np.asarray(rewards_batch[idx]) for idx in range(curr_batch_size)])
                
                Rt_batch, cumul_r = np.zeros_like(rewards_batch), np.zeros((curr_batch_size,))
                for t in reversed(range(num_overall_steps)): # Rt = E[ sum_{n=0}^{T−t} γ^n r_{t+n} ]
                    cumul_r = np.nansum(np.dstack((rewards_batch[:,t], cumul_r * self.gamma)), axis=2)
                    Rt_batch[:,t] = cumul_r

                Rt_batch = torch.from_numpy(Rt_batch.transpose(1,0).copy()).float().to(device)

                # Ldec(θπ) = L′(θπ) − sum[ β · H(π(at|st, θπ)) ], where L′(θπ) = −sum[ (log π(at|st, θπ)) (Rt − v(st|θv)) ] (Equations 7 and 8)
                policy_losses_batch = -log_probs_batch * (Rt_batch.unsqueeze(2) - state_values_batch.detach().requires_grad_(False)) - self.entropy_beta*entropies_batch

                valid_rewards_in_batch = ~np.isnan(rewards_batch) # Check which episodes in batch had not already finished (nan rewards mean 'finished')
                num_steps_per_env = np.count_nonzero(valid_rewards_in_batch, axis=1)

                ### Policy Loss is the average of the Monte Carlo sampling over the trajectories (full episodes)
                policy_loss_batch = torch.cat([policy_losses_batch[:num_steps_per_env[batch_idx], batch_idx].sum().unsqueeze(0) for batch_idx in range(curr_batch_size)])
                policy_loss = policy_loss_batch.mean()

                ### Critic Loss is Mean Squared Error - Lδ(θδ) = sum_t [ (δ(st|θδ) −Rt)² ]
                critic_loss_batch = torch.cat([F.mse_loss(state_values_batch[:num_steps_per_env[batch_idx], batch_idx], Rt_batch[:num_steps_per_env[batch_idx], batch_idx].unsqueeze(1)).unsqueeze(0) for batch_idx in range(curr_batch_size)])
                critic_loss = critic_loss_batch.mean()
                
                ### Apending losses for future optimization
                epoch_losses.append(policy_loss.item())
                critic_losses.append(critic_loss.item())
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True) ### Computes the derivative of loss with respect to theta (dLoss/dTheta)
                self.policy_optimizer.step() ### Updates the theta parameters (e.g., θ = θ -lr * L′/∇θ in SGD)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True) ### Computes the derivative of loss with respect to theta (dLoss/dTheta)
                self.critic_optimizer.step() ### Updates the theta parameters (e.g., θ = θ -lr * L′/∇θ in SGD)

                for exp_idx, exp_key in enumerate(keys_batch):
                    Rt[exp_key].append(np.sum(Rt_batch[:num_steps_per_env[exp_idx], exp_idx].detach().cpu().numpy()))
                    episode_speedup_deviation = envs_batch[exp_idx].num_frames/envs_batch[exp_idx].get_num_selected_frames() - envs_batch[exp_idx].desired_skip
                    epoch_speedups_deviation.append(episode_speedup_deviation)

                    entropies_epoch.append(torch.cat([entropies_batch[: num_steps_per_env[exp_idx], exp_idx]]).mean().item())
                
                    ### Logging to Tensorboard (check the file location for inspection)
                    self.writer.add_scalar('Discounted_Rewards_{}/avg_sum_discounted_rewards_{}'.format(dataset_name, exp_key), np.mean(Rt[exp_key]), i_epoch)
                    self.writer.add_scalar('Curr_Discounted_Rewards_{}/sum_discounted_reward_{}'.format(dataset_name, exp_key), Rt[exp_key][-1], i_epoch)
                    self.writer.add_scalar('Entropy_{}/curr_entropy_{}'.format(dataset_name, exp_key), torch.cat([entropies_batch[:num_steps_per_env[exp_idx],exp_idx]]).sum(), i_epoch)
                    self.writer.add_scalar('Speedup_{}/episode_speedup_deviation_{}'.format(dataset_name, exp_key), episode_speedup_deviation, i_epoch)
                    self.writer.add_scalar('Policy_Loss_{}/curr_loss_{}'.format(dataset_name, exp_key), policy_loss_batch[exp_idx].item(), i_epoch)
                    self.writer.add_scalar('Critic_Loss_{}/curr_loss_{}'.format(dataset_name, exp_key), critic_loss_batch[exp_idx].item(), i_epoch)
        
            self.writer.add_scalar('Discounted_Rewards_{}/_overall_avg_sum_discounted_rewards'.format(dataset_name), np.mean(np.array([np.mean(Rt[exp_key]) for exp_key in exp_keys])), i_epoch)
            self.writer.add_scalar('Speedup_{}/_overall_episode_speedup_deviation'.format(dataset_name), np.mean(np.array(epoch_speedups_deviation)), i_epoch)
            self.writer.add_scalar('Policy_Loss_{}/_avg_epoch_loss'.format(dataset_name), np.mean(np.array(epoch_losses)), i_epoch)
            self.writer.add_scalar('Critic_Loss_{}/_avg_epoch_loss'.format(dataset_name), np.mean(np.array(critic_losses)), i_epoch)
            self.writer.add_scalar('Entropy_{}/_avg_epoch_entropy'.format(dataset_name), np.mean(np.array(entropies_epoch)), i_epoch)

            # Save model at every 10 epochs (snapshot)
            if i_epoch % 10 == 9:
                self.save_model(model_path, i_epoch+1)
        
        # Save final model
        self.save_model(model_path, i_epoch+1)

    def test(self, env, dataset_name, exp_name, log=False):
        self.policy.eval()

        if self.writer is None:
            hostname = socket.gethostname()
            self.writer = SummaryWriter(log_dir=f'{config.logs_base_folder}/{self.creation_timestamp}_{hostname}_{dataset_name.lower()}_test/', filename_suffix=f'_{self.creation_timestamp}')
        
        self.write_states_and_dots_to_log(env, dataset_name, exp_name)

        self.rewards = []
        state = env.reset()

        done = False
        while not done:
            state = state.unsqueeze(0).to(device)
            action, _ = self.policy.argmax_action(state) # argmax_a π(a|st, θπ)
            state, reward, done, _ = env.step(action)
            self.rewards.append(reward)
            
        Rt, cumul_r = np.zeros_like(self.rewards), 0
        for t in reversed(range(len(self.rewards))):
            cumul_r = self.rewards[t] + cumul_r * self.gamma
            Rt[t] = cumul_r

        if log:
            print('Sum of Rewards (Sum(r_t)): {:.3f}\nUtility (Sum(R_t)): {:.3f}\nNum selected frames: {}\nDesired Speed-up: {} | Achieved Speed-up: {:.3f}'.format(sum(self.rewards), sum(Rt), len(env.selected_frames), env.desired_skip, float(env.num_frames)/len(env.selected_frames)))
        
    def write_selected_frames_image_to_log(self, env, i_episode, prefix, suffix=None):      
        skips = np.array(env.selected_frames[1:]) - np.array(env.selected_frames[:-1])
        
        fig, ax = plt.subplots()
        ax.scatter(env.selected_frames[:-1], skips)
        
        if self.annotations is not None: # If we have annotations, let's use them to print segments for debugging
            video_annotations = self.annotations['database'][env.experiment_name]['annotations'] if env.dataset_name == 'YouCook2' else self.annotations['database'][env.experiment_name]['annotation']
            cmap = plt.cm.get_cmap('hsv', len(video_annotations) + 1)
            for idx, region in enumerate(video_annotations):
                video_region = region['segment']
                ax.axvspan(round(video_region[0] * env.experiment.fps), round(video_region[1] * env.experiment.fps), alpha=0.5, color=cmap(idx))
                
        if suffix:            
            self.writer.add_figure('{}/{}'.format(prefix, suffix), fig, i_episode)
        else:
            self.writer.add_figure('{}'.format(prefix), fig, i_episode)

    def write_rewards_image_to_log(self, env, rewards, i_episode, prefix, suffix=None):      
        
        fig, ax = plt.subplots()
        ax.scatter(env.selected_frames, rewards)
        
        if self.annotations is not None: # If we have annotations, let's use them to print segments for debugging
            video_annotations = self.annotations['database'][env.experiment_name]['annotations'] if env.dataset_name == 'YouCook2' else self.annotations['database'][env.experiment_name]['annotation']
            cmap = plt.cm.get_cmap('hsv', len(video_annotations) + 1)
            for idx, region in enumerate(video_annotations):
                video_region = region['segment']
                ax.axvspan(round(video_region[0] * env.experiment.fps), round(video_region[1] * env.experiment.fps), alpha=0.5, color=cmap(idx))
                
        if suffix:            
            self.writer.add_figure('{}/{}'.format(prefix, suffix), fig, i_episode)
        else:
            self.writer.add_figure('{}'.format(prefix), fig, i_episode)
            
    def write_states_and_dots_to_log(self, env, dataset_name, exp_name):
        
        fig, ax = plt.subplots()   
        plot_vec = env.dots
        idxs = np.arange(len(plot_vec))
        
        ax.plot(idxs, plot_vec)

        if self.annotations is not None: # If we have annotations, let's use them to print segments for debugging
            video_annotations = self.annotations['database'][env.experiment_name]['annotations'] if dataset_name == 'YouCook2' else self.annotations['database'][env.experiment_name]['annotation']
            cmap = plt.cm.get_cmap('hsv', len(video_annotations) + 1)
            for idx, region in enumerate(video_annotations):
                video_region = region['segment']
                ax.axvspan(round(video_region[0] * env.experiment.fps), round(video_region[1] * env.experiment.fps), alpha=0.5, color=cmap(idx))

        self.writer.add_figure('Frames_Dots_{}/{}'.format(dataset_name, exp_name), fig)

        fig2, ax2 = plt.subplots()
        ax2.matshow(torch.cat((env.NRPE, env.NRPE), dim=1), aspect='auto')
        self.writer.add_figure('States_Composition{}/NRPE_{}'.format(dataset_name, exp_name), fig2)

        fig3, ax3 = plt.subplots()
        ax3.matshow(env.semantic_embeddings, aspect='auto')
        self.writer.add_figure('States_Composition{}/Semantic_Embeddings_{}'.format(dataset_name, exp_name), fig3)

    def save_model(self, model_path, i_epoch, log=False):
        if not os.path.isdir(os.path.abspath(model_path)):
            os.mkdir(os.path.abspath(model_path))

        model_path = os.path.join(os.path.abspath(model_path), f'checkpoint_ep{i_epoch}.pth')
        training_state = {'curr_epoch': i_epoch,
                        'policy_state_dict': self.policy.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'optimizer_state_dict': self.policy_optimizer.state_dict(),
                        'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                        'lr': self.lr,
                        'critic_lr': self.critic_lr,
                        'gamma': self.gamma,
                        'entropy_beta': self.entropy_beta,
                        'creation_timestamp': self.creation_timestamp}
                        
        if log:
            print('[{}] Saving model...'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
            torch.save(training_state, model_path)
            print('[{}] Done!'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
        else:
            torch.save(training_state, model_path)


    def load_model(self, model_path):
        print('[{}] Loading model...'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
        training_state = torch.load(model_path)

        self.policy.load_state_dict(training_state['policy_state_dict'])
        self.critic.load_state_dict(training_state['critic_state_dict'])
        self.policy_optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(training_state['critic_optimizer_state_dict'])
        self.lr = training_state['lr']
        self.critic_lr = training_state['critic_lr']
        self.gamma = training_state['gamma']
        self.entropy_beta = training_state['entropy_beta']
        self.creation_timestamp = training_state['creation_timestamp']
        self.curr_epoch = training_state['curr_epoch']

        print('[{}] Done!'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
