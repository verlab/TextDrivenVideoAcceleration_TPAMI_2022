import torch.utils.data as data

import random
import os
import cv2
import json
from tqdm import tqdm
from datetime import datetime as dt

import numpy as np
import torch
import glob
from utils import convert_sentences_to_word_idxs

random.seed(123)
np.random.seed(123)

class VaTeXDataLoader(data.Dataset):
    def __init__(self, root, annFile, word_map, vid_transform=None, annotations_transform=None, num_sentences=20, max_words=20, neg_sentences_proportion=0.5, dataset_proportion=1., generate_negative_samples=True, num_negative_samples_per_video=1, training_data=True, num_input_frames=32):
        self.root = os.path.abspath(root)
        self.training_data = training_data
        self.num_input_frames = num_input_frames

        print('[{}] Loading annotations file...'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.dataset = json.load(open(annFile))
        self.anns = {video['videoID']: video['enCap'] for video in tqdm(self.dataset, desc='Sentences')}
        self.ids = list(self.anns.keys())
        self.ids = self.ids[:int(dataset_proportion*len(self.ids))]

        # Checking which videos/feats exist
        videosSet = set(glob.glob('{}/*.mp4'.format(self.root)))
        self.ids = [_id for _id in self.ids if '{}/{}.mp4'.format(self.root, '_'.join(_id.split('_')[:-2])) in videosSet]
        print('[{}] {} videos found!'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), len(self.ids)))
        
        self.ids = np.array(self.ids)
        self.num_vids = len(self.ids)
        print('[{}] Done!\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))

        self.vid_transform = vid_transform
        self.annotations_transform = annotations_transform
        self.generate_negative_samples = generate_negative_samples
        self.neg_sentences_proportion = neg_sentences_proportion # TODO: Use it later (Not using it yet)
        self.word_map = word_map
        self.num_sentences = num_sentences
        self.max_words = max_words
        self.num_negative_sentences = int(self.neg_sentences_proportion*self.num_sentences)
        self.num_positive_sentences = self.num_sentences - self.num_negative_sentences
        self.num_negative_samples_per_video = num_negative_samples_per_video
    
    def get_captions(self, vid_id):  
        captions = list(set(self.anns[vid_id])) # Get all captions (There's about 40 for each video)
        
        return captions

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple ((image, annotations), target). target is a binary list telling if images and captions match.
        """
        is_negative_sample, index = divmod(index, self.num_vids)# If index greater than max, generate a negative sample instead

        vid_id = self.ids[index]
        
        if is_negative_sample: # If it is negative, we should get a random annotation to truly make it negative
            
            rand_ids = random.sample([_id for _id in self.ids if _id != vid_id], 2)
            
            captions = self.get_captions(rand_ids[0])[:self.num_positive_sentences]  # Get only the necessary amount of sentences
            captions.extend(self.get_captions(rand_ids[1])[:self.num_negative_sentences])  # Get only the necessary amount of sentences

        else: # Senteces to compose the positive pair
                        
            rand_id = random.sample([_id for _id in self.ids if _id != vid_id],1)[0]

            captions = self.get_captions(vid_id)[:self.num_positive_sentences] # Get only the necessary amount of sentences
            negative_captions = self.get_captions(rand_id)[:self.num_negative_sentences]  # Get only the necessary amount of sentences
            captions.extend(negative_captions)

        ## Uncomment if needded (num of captions do not match across the dataset)
        # while len(captions) < self.num_sentences:
        #     captions.append('PAD')

        random.shuffle(captions)
        
        # Converting annotations to array of indexes
        converted_annotations, words_per_sentence = convert_sentences_to_word_idxs(captions, self.max_words, self.word_map)

        vid_path = os.path.join(self.root, '{}.mp4'.format('_'.join(vid_id.split('_')[:-2])))
        frames = []
        vid = cv2.VideoCapture(vid_path)
        start_frame = np.random.randint(max(int(vid.get(7))-self.num_input_frames,1))
        vid.set(1, start_frame)

        for _ in range(self.num_input_frames):
            ret, frame = vid.read()

            if ret:
                frames.append(frame)
            else:
                frames.append(np.zeros((int(vid.get(4)), int(vid.get(3)), 3), dtype=np.uint8))

        vid.release()  # Release the video as soon as possible to prevent from thread concurrency
        # try:
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        # except:
        #     print(vid_path)
        #     exit(1)
        clip = self.vid_transform(frames)
                
        if self.annotations_transform is not None:
            converted_annotations = self.annotations_transform(converted_annotations)
            
        return vid_id, captions, clip, converted_annotations, converted_annotations.size()[1], words_per_sentence, torch.tensor([1. if not is_negative_sample else -1.], dtype=torch.float32)

    def __len__(self):
        if self.generate_negative_samples:
            return len(self.ids)*(self.num_negative_samples_per_video+1) # Returns x-times the length because for every positive document match, we create its negative counterpart
        else:
            return len(self.ids)
