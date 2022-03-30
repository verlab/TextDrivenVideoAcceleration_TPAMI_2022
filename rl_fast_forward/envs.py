import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_encoding.utils import extract_vdan_plus_feats, extract_vdan_feats
from semantic_encoding._utils.experiment_to_video_mapping import Experiment2VideoMapping

import torch.backends.cudnn as cudnn
from datetime import datetime
import numpy as np
import torch
import gym
import config

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(123)
torch.manual_seed(123)

class VideoEnvironment(gym.core.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, semantic_encoder_data, document_filename=None, dataset_name=None, experiment_name=None, input_video_filename=None, batch_size=32, speedup=0, use_vdan=False, no_positional_encoding=False, no_skip_info=False, lambda_param='F*', sigma_param=0.5):
        
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        
        if self.experiment_name: # Get all experiment details
            self.experiment = Experiment2VideoMapping(self.experiment_name, self.dataset_name)

        ## 3.2.1 Action Space and States Composition
        self.MIN_SKIP = 1 # ν_min = 1
        self.MAX_SKIP = 25 # ν_max = 25
        self.MIN_ACC = 1 # ω_min = 1
        self.MAX_ACC = 5 # ω_max = 5

        ## Action Space
        self.action_size = 3
        self.speedup = speedup # Desired Speedup (S* ∈ N+)
        self.acceleration = 1

        ## States Composition
        self.vid_feats_size = semantic_encoder_data['model_params']['feat_embed_size']
        self.doc_feats_size = semantic_encoder_data['model_params']['feat_embed_size']
        
        self.avg_skip_embedding_dims = self.MAX_SKIP*2
        self.reversed_pos_enc_size = self.avg_skip_embedding_dims

        self.no_positional_encoding = no_positional_encoding
        self.no_skip_info = no_skip_info

        # Ablation
        if self.no_positional_encoding and self.no_skip_info:
            self.state_size = self.doc_feats_size + self.vid_feats_size

        elif self.no_positional_encoding:
            self.state_size = self.doc_feats_size + self.vid_feats_size + self.avg_skip_embedding_dims

        elif self.no_skip_info:
            self.state_size = self.doc_feats_size + self.vid_feats_size + self.reversed_pos_enc_size

        else:
            self.state_size = self.doc_feats_size + self.vid_feats_size + self.reversed_pos_enc_size + self.avg_skip_embedding_dims

        self.use_vdan = use_vdan

        self.semantic_embeddings = self.get_semantic_embeddings(semantic_encoder_data, document_filename, input_video_filename, batch_size)        
        self.NRPE = self.get_NRPE() # Normalized Reversed Positional Encoding (NRPE) - Equation 2
        self.Im = torch.eye(self.avg_skip_embedding_dims) # I_m, where m = 2*ν_max

        ## 3.3.2 Reward Function
        self.lambda_param = lambda_param # λ in Equation 3
        self.sigma = sigma_param # σ in Equation 3
        
        # Computing dot product for semantic alignment (Equation 3)
        self.dots = np.zeros((self.num_frames,), dtype=np.float32)
        for idx in range(self.num_frames):
            self.dots[idx] = torch.dot(self.semantic_embeddings[idx, :self.vid_feats_size], self.semantic_embeddings[idx, self.vid_feats_size:]).item()
        
        ## Other Training Parameters
        self.curr_frame_id = None
        self.skip = None
        self.skips = []        
        
    def get_semantic_embeddings(self, semantic_encoder_data, document_filename, input_video_filename, batch_size):
        if self.experiment_name:
            self.input_video_filename = self.experiment.video_filename

            if document_filename is None:
                document_filename = self.experiment.document_filename
            
            if self.use_vdan:
                vid_feats_base_dir = '{}/{}/VDAN/img_feats/{}'.format(config.deep_feats_base_folder, self.experiment.dataset, semantic_encoder_data['model_name'])
                
                doc_feats_base_dir = '{}/{}/VDAN/doc_feats/{}'.format(config.deep_feats_base_folder, self.experiment.dataset, semantic_encoder_data['model_name'])
                
            else:
                vid_feats_base_dir = '{}/{}/VDAN+/vid_feats/{}'.format(config.deep_feats_base_folder, self.experiment.dataset, semantic_encoder_data['model_name'])
                
                doc_feats_base_dir = '{}/{}/VDAN+/doc_feats/{}'.format(config.deep_feats_base_folder, self.experiment.dataset, semantic_encoder_data['model_name'])
                
            document_basename = os.path.basename(os.path.splitext(document_filename)[0])

            # Check if model folder exists
            os.makedirs(vid_feats_base_dir, exist_ok=True)
            os.makedirs(doc_feats_base_dir, exist_ok=True)

            vid_feats_filename = '{}/{}_{}_feats.npz'.format(vid_feats_base_dir, self.experiment.video_name, 'img' if self.use_vdan else 'vid')
            
            doc_feats_filename = '{}/{}/{}_doc_feats.npz'.format(doc_feats_base_dir, document_basename, self.experiment.video_name)

        elif input_video_filename:
            self.input_video_filename = input_video_filename
            
            vid_feats_filename = '{}/{}_{}_feats.npz'.format(os.path.dirname(input_video_filename), os.path.basename(os.path.splitext(input_video_filename)[0]), 'img' if self.use_vdan else 'vid')
            
            doc_feats_filename = '{}/{}_{}_doc_feats.npz'.format(os.path.dirname(input_video_filename), os.path.basename(os.path.splitext(input_video_filename)[0]), os.path.basename(os.path.splitext(document_filename)[0]))
        else:
            print('ERROR: experiment name or input video filename needed!')
            exit(1)
        
        document = np.loadtxt(document_filename, delimiter='\n', dtype=str, encoding='utf-8')

        # Load features if they already exist
        if os.path.exists(vid_feats_filename) and os.path.exists(doc_feats_filename):
            doc_feats_dic = np.load(doc_feats_filename, allow_pickle=True)

            doc_embeddings = doc_feats_dic['features']
            saved_document = doc_feats_dic['document']
            semantic_encoder_name = str(doc_feats_dic['semantic_encoder_name'])

            ##### Checking if the features to be loaded are valid! #####
            # Check if the saved document is the same as the loaded one. If not, somebody changed it and it is not valid since features are different.
            # Also checking if the semantic model is the same as the one used to extract these features
            if (len(document) == len(saved_document) and (document == saved_document).all() and semantic_encoder_name == os.path.basename(os.path.splitext(semantic_encoder_data['model_name'])[0])):
                
                vid_embeddings = np.load(vid_feats_filename, allow_pickle=True)['features']
                self.num_frames = vid_embeddings.shape[0]

                semantic_embeddings = np.concatenate([doc_embeddings, vid_embeddings], axis=1)

                return torch.from_numpy(semantic_embeddings)
            else:
                print(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Saved features cannot be used: the provided document or model has been modified...')

        # If we cannot load or use the features for any reason, keep on going and extract it
        # Check if folders exist before extracting feats
        os.makedirs(os.path.dirname(doc_feats_filename), exist_ok=True) # Creating unexistent directory if needed (doc feats has one extra depth in folder tree structure)

        if self.use_vdan:
            vid_embeddings, doc_embeddings, _, words_atts, sentences_atts = extract_vdan_feats(semantic_encoder_data['semantic_encoder_model'], semantic_encoder_data['train_params'], semantic_encoder_data['model_params'], semantic_encoder_data['word_map'], self.input_video_filename, document_filename, batch_size, semantic_encoder_data['input_frames_length'])
        else:
            vid_embeddings, doc_embeddings, _, words_atts, sentences_atts = extract_vdan_plus_feats(semantic_encoder_data['semantic_encoder_model'], semantic_encoder_data['train_params'], semantic_encoder_data['model_params'], semantic_encoder_data['word_map'], self.input_video_filename, document_filename, batch_size, semantic_encoder_data['input_frames_length'])

        self.num_frames = vid_embeddings.shape[0]

        semantic_embeddings = np.concatenate([doc_embeddings, vid_embeddings], axis=1)

        ## Saving new extracted feats
        semantic_encoder_name = os.path.basename(os.path.splitext(semantic_encoder_data['model_name'])[0])
        np.savez_compressed(vid_feats_filename, features=vid_embeddings, semantic_encoder_name=semantic_encoder_name)
        np.savez_compressed(doc_feats_filename, features=doc_embeddings, document=document, semantic_encoder_name=semantic_encoder_name, words_atts=words_atts, sentences_atts=sentences_atts)

        return torch.from_numpy(semantic_embeddings)

    def step(self, action):
        info = {'items': []}
        if self.curr_frame_id == self.num_frames - 1:  # No more actions to take, episode is done
            observation = torch.zeros(1, self.state_size)
            return observation, float('nan'), True, info

        if action == 0:  # Accelerate
            if self.acceleration < self.MAX_ACC:
                self.acceleration += 1
            if self.skip + self.acceleration < self.MAX_SKIP:
                self.skip += self.acceleration
            else:
                self.skip = self.MAX_SKIP
        elif action == 2:  # Deaccelerate
            if self.acceleration > self.MIN_ACC:
                self.acceleration -= 1
            if self.skip - self.acceleration > self.MIN_SKIP:
                self.skip -= self.acceleration
            else:
                self.skip = self.MIN_SKIP

        self.skips.append(self.skip)
        self.curr_frame_id += int(self.skip)

        self.curr_frame_id = min(self.curr_frame_id, self.num_frames-1)
        self.selected_frames.append(self.curr_frame_id)
        done = self.curr_frame_id == self.num_frames-1

        ### REWARD
        if done: # t = T
            reward = self.gaussian(np.mean(self.skips), self.desired_skip)*self.lambda_value if self.lambda_value else self.get_semantic_reward() # r_t (Equation 3)
        else: # t < T
            reward = self.get_semantic_reward() # r_t (Equation 3)

        ### NEW OBSERVATION
        observation = self.get_observation()

        self.total_reward += reward
        return observation, reward, done, info

    def reset(self):
        self.desired_skip = self.speedup if self.speedup else np.random.randint(self.MIN_SKIP, self.MAX_SKIP)
        self.desired_num_frames = self.num_frames // self.desired_skip
        self.skip = self.desired_skip
        self.acceleration = self.MIN_ACC
        self.skips = [self.skip]
        self.curr_frame_id = 0
        self.total_reward = 0
        self.selected_frames = []

        if self.lambda_param == 'F*':  # desired number of output frames given the speed-up rate
            self.lambda_value = self.desired_num_frames
        elif self.lambda_param == 'F':  # number of input frames
            self.lambda_value = self.num_frames
        elif self.lambda_param == '0':  # number of input frames
            self.lambda_value = 0

        observation = self.get_observation()

        return observation

    def get_observation(self):
        
        if self.no_positional_encoding and self.no_skip_info:
            observation = self.semantic_embeddings[self.curr_frame_id].unsqueeze(0)

        elif self.no_positional_encoding:
            one_hot_vec_skip = self.Im[int(np.round((np.mean(self.skips) - self.desired_skip) + self.avg_skip_embedding_dims/2))]
            observation = torch.cat((self.semantic_embeddings[self.curr_frame_id], one_hot_vec_skip)).unsqueeze(0)

        elif self.no_skip_info:
            observation = torch.cat((self.semantic_embeddings[self.curr_frame_id], self.NRPE[self.curr_frame_id])).unsqueeze(0)

        else:
            one_hot_vec_skip = self.Im[int(np.round((np.mean(self.skips) - self.desired_skip) + self.avg_skip_embedding_dims/2))]
            observation = torch.cat((self.semantic_embeddings[self.curr_frame_id], self.NRPE[self.curr_frame_id], one_hot_vec_skip)).unsqueeze(0)
            
        return observation

    def get_semantic_reward(self):
        return self.dots[self.curr_frame_id]

    def get_num_selected_frames(self):
        return len(self.selected_frames)

    def get_wk(self, k, q):
        return 1/np.power(self.num_frames, 2*k/q)

    def get_NRPE(self):
        wks = np.array([self.get_wk(k, self.reversed_pos_enc_size) for k in range(self.reversed_pos_enc_size//2)])
        NRPE = np.zeros((self.num_frames, self.reversed_pos_enc_size), dtype=np.float32)
        even_idxs = np.array([2*k for k in range(self.reversed_pos_enc_size//2)])
        odd_idxs = even_idxs+1
        for t in range(self.num_frames):
            NRPE[-(t+1), even_idxs] = np.sin(wks*t)
            NRPE[-(t+1), odd_idxs] = np.cos(wks*t)

        return torch.from_numpy(NRPE)

    def gaussian(self, x, mu):
        return np.exp(-0.5 * np.power((x - mu)/self.sigma, 2.))

    def render(self, mode='human'):
        pass
