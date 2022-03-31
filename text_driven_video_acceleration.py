import torch
import torch.nn as nn
import json
import cv2
import os
import numpy as np
import torchvideo.transforms as VT
from torchvision.transforms import Compose
from tqdm import tqdm
from semantic_encoding.models import VDAN_PLUS
from semantic_encoding.utils import convert_sentences_to_word_idxs
from rl_fast_forward.REINFORCE.policy import Policy
from rl_fast_forward.REINFORCE.critic import Critic

KINECTS400_MEAN = [0.43216, 0.394666, 0.37645]
KINECTS400_STD = [0.22803, 0.22145, 0.216989]

MIN_SKIP = 1
MAX_SKIP = 25
MAX_ACC = 5
MIN_ACC = 1
MAX_FRAMES = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointModel(nn.Module):
    def __init__(
            self, vocab_size, doc_emb_size, sent_emb_size, word_emb_size, sent_rnn_layers, word_rnn_layers, hidden_feat_emb_size, final_feat_emb_size,
            sent_att_size, word_att_size, use_visual_shortcut=False, use_sentence_level_attention=False, use_word_level_attention=False,
            learn_first_hidden_vector=True, action_size=3):
        super(JointModel, self).__init__()

        self.vdan_plus = VDAN_PLUS(vocab_size=vocab_size,
                                   doc_emb_size=doc_emb_size,  # ResNet-50 embedding size
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

    def fast_forward_video(self, video_filename, document, desired_speedup, output_video_filename=None, max_words=20):
        word_map = json.load(open(f'{os.path.dirname(os.path.abspath(__file__))}/semantic_encoding/resources/glove6B_word_map.json'))
        
        vid_transformer = Compose([
            VT.NDArrayToPILVideo(),
            VT.ResizeVideo(112),
            VT.PILVideoToTensor(),
            VT.NormalizeVideo(mean=KINECTS400_MEAN, std=KINECTS400_STD)
        ])

        converted_sentences, words_per_sentence = convert_sentences_to_word_idxs(document, max_words, word_map)
        sentences_per_document = np.array([converted_sentences.shape[0]])

        transformed_document = torch.from_numpy(converted_sentences).unsqueeze(0).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = torch.from_numpy(sentences_per_document).to(device)  # (batch_size)
        words_per_sentence = torch.from_numpy(words_per_sentence).unsqueeze(0).to(device)  # (batch_size, sentence_limit)

        video = cv2.VideoCapture(video_filename)
        if output_video_filename:
            fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
            fps = video.get(5)
            frame_width = int(video.get(3))
            frame_height = int(video.get(4))
            output_video = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))

        num_frames = int(video.get(7))
        acceleration = 1
        skip = 1
        frame_idx = 0
        selected_frames = []

        self.Im = torch.eye(self.q).to(device)
        self.NRPE = self.get_NRPE(num_frames).to(device)

        curr_frames = [None for _ in range(MAX_FRAMES)]
        skips = [skip]
        pbar = tqdm(total=num_frames)
        while frame_idx < num_frames:
            video.set(1, frame_idx)
            for i in range(MAX_FRAMES):
                ret, frame = video.read()

                if not ret:
                    curr_frames[i] = np.zeros((int(video.get(4)), int(video.get(3)), 3), dtype=np.uint8)
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                curr_frames[i] = frame
                
            transformed_frames = vid_transformer(curr_frames).unsqueeze(0).to(device)
            
            if output_video_filename:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(curr_frames[0], '{}x'.format(skip), (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                output_video.write(curr_frames[0])

            with torch.no_grad():
                vid_embedding, text_embedding, _, _, _ = self.vdan_plus(transformed_frames, transformed_document, sentences_per_document, words_per_sentence)

                SA_vector = self.Im[int(np.round((np.mean(skips) - desired_speedup) + MAX_SKIP))]
                action_probs = self.policy(torch.cat([vid_embedding, text_embedding, SA_vector, self.NRPE[frame_idx]], axis=1))

            action = torch.argmax(action_probs).item()

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
