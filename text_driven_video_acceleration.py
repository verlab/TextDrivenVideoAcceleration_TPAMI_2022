import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from semantic_encoding.models import VDAN_PLUS
from semantic_encoding.utils import load_checkpoint, extract_vdan_plus_feats
from rl_fast_forward.REINFORCE.policy import Policy
from rl_fast_forward.REINFORCE.critic import Critic

MIN_SKIP = 1
MAX_SKIP = 25
MAX_ACC = 5
MIN_ACC = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointModel(nn.Module):
    def __init__(self, vocab_size, doc_emb_size, sent_emb_size, word_emb_size, sent_rnn_layers, word_rnn_layers, hidden_feat_emb_size, final_feat_emb_size,
            sent_att_size, word_att_size, use_visual_shortcut=False, use_sentence_level_attention=False, use_word_level_attention=False,
            learn_first_hidden_vector=True, action_size=3, pretrained=False, progress=False):
        super(JointModel, self).__init__()

        self.vdan_plus = VDAN_PLUS(vocab_size=vocab_size,
                                   doc_emb_size=doc_emb_size,  # R(2+1)D embedding size
                                   sent_emb_size=sent_emb_size,
                                   word_emb_size=word_emb_size,  # GloVe embeddings size
                                   sent_rnn_layers=sent_rnn_layers,
                                   word_rnn_layers=word_rnn_layers,
                                   hidden_feat_emb_size=hidden_feat_emb_size,
                                   final_feat_emb_size=final_feat_emb_size,
                                   sent_att_size=sent_att_size,
                                   word_att_size=word_att_size,
                                   use_visual_shortcut=use_visual_shortcut,
                                   use_sentence_level_attention=use_sentence_level_attention,
                                   use_word_level_attention=use_word_level_attention,
                                   learn_first_hidden_vector=learn_first_hidden_vector)

        self.m = 2*MAX_SKIP  # Size of the one hot vectors for the Skip-Aware vector
        self.q = self.m  # Size of the NRPE vectors
        self.state_size = final_feat_emb_size + final_feat_emb_size + self.m + self.q

        self.policy = Policy(state_size=self.state_size, action_size=action_size)
        self.critic = Critic(state_size=self.state_size)
                
        if pretrained:                        
            _, vdan_plus_model, _, vdan_plus_word_map, vdan_plus_model_params, vdan_plus_train_params = load_checkpoint('https://github.com/verlab/TextDrivenVideoAcceleration_TPAMI_2022/releases/download/pre_release/vdan+_pretrained_model.pth', load_from_url=True, progress=progress)
                            
            self.vdan_plus_data = {
                'model_name': 'vdan+_pretrained_model',
                'semantic_encoder_model': vdan_plus_model,
                'word_map': vdan_plus_word_map,
                'train_params': vdan_plus_train_params,
                'model_params': vdan_plus_model_params,
                'input_frames_length': 32        
            }
            
            self.vdan_plus = vdan_plus_model
            
            agent_state_dict = torch.hub.load_state_dict_from_url('https://github.com/verlab/TextDrivenVideoAcceleration_TPAMI_2022/releases/download/pre_release/youcookii_saffa_model.pth', progress=progress)
            self.policy.load_state_dict(agent_state_dict['policy_state_dict'])

    def fast_forward_video(self, video_filename, document, desired_speedup, output_video_filename=None, vdan_plus_batch_size=16, semantic_embeddings=None):
        
        if semantic_embeddings is None:
            semantic_embeddings = self.get_semantic_embeddings(video_filename, document, vdan_plus_batch_size).unsqueeze(0)
            
        semantic_embeddings = semantic_embeddings.to(device)
        
        if output_video_filename:
            video = cv2.VideoCapture(video_filename)
            fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
            fps = video.get(5)
            frame_width = int(video.get(3))
            frame_height = int(video.get(4))
            output_video = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))

        acceleration = 1
        skip = 1
        frame_idx = 0
        selected_frames = []        
        num_frames = semantic_embeddings.shape[0]
        
        self.Im = torch.eye(self.q).to(device)
        self.NRPE = self.get_NRPE(num_frames).to(device)

        skips = [skip]
        pbar = tqdm(total=num_frames)
        while frame_idx < num_frames:
            if output_video_filename:
                i = 0
                while i < skip and frame_idx < num_frames:
                    ret, frame = video.read()
                    i += 1

                if not ret:
                    print('Error reading frame: {}'.format(frame_idx))
                    break
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, '{}x'.format(skip), (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                output_video.write(frame)

            observation = torch.cat((semantic_embeddings[frame_idx],
                                     self.NRPE[frame_idx],
                                     self.Im[int(np.round((np.mean(skips) - desired_speedup) + MAX_SKIP))])).unsqueeze(0)
            
            action_probs = self.policy(observation.unsqueeze(0))

            action = torch.argmax(action_probs.squeeze(0)).item()

            if action == 0:  # Accelerate
                if acceleration < MAX_ACC:
                    acceleration += 1
                if skip + acceleration <= MAX_SKIP:
                    skip += acceleration
                else:
                    skip = MAX_SKIP
            elif action == 2:  # Decelerate
                if acceleration > MIN_ACC:
                    acceleration -= 1
                if skip - acceleration >= MIN_SKIP:
                    skip -= acceleration
                else:
                    skip = MIN_SKIP

            skips.append(skip)
            frame_idx += skip
            selected_frames.append(frame_idx+1)
            pbar.update(skip)

        pbar.close()

        return selected_frames

    def get_wk(self, F, k):
        return 1/np.power(F, 2*k/self.q)

    def get_NRPE(self, F):
        wks = np.array([self.get_wk(F, k) for k in range(self.q//2)])
        NRPE = np.zeros((F, self.q), dtype=np.float32)
        even_idxs = np.array([2*k for k in range(self.q//2)])
        odd_idxs = even_idxs+1
        for t in range(F):
            NRPE[-(t+1), even_idxs] = np.sin(wks*t)
            NRPE[-(t+1), odd_idxs] = np.cos(wks*t)

        return torch.from_numpy(NRPE)

    def get_semantic_embeddings(self, video_filename, document, vdan_plus_batch_size=16):
        
        temp_doc_filename = f'{os.path.basename(os.path.abspath(video_filename)).split(".")[0]}.txt'
        np.savetxt(temp_doc_filename, document, fmt='%s')
        
        vid_embeddings, doc_embeddings, _, _, _ = extract_vdan_plus_feats(self.vdan_plus_data['semantic_encoder_model'], self.vdan_plus_data['train_params'], self.vdan_plus_data['model_params'], self.vdan_plus_data['word_map'], video_filename, temp_doc_filename, vdan_plus_batch_size, self.vdan_plus_data['input_frames_length'])
        
        os.remove(temp_doc_filename)
        
        return torch.from_numpy(np.concatenate([doc_embeddings, vid_embeddings], axis=1))