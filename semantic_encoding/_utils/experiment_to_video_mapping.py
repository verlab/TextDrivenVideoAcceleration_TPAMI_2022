import cv2
import os
import numpy as np
import pandas as pd
import json

from glob import glob

class Experiment2VideoMapping(object):
    def __init__(self, experiment_name, dataset=None):
        self.experiment_name = experiment_name
        self.video_name = None
        self.video_filename = None
        self.semantics = None
        self.dataset = dataset
        self.video_extension = None
        self.start_frame = None
        self.end_frame = None
        self.num_frames = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.recipe_id = None
        self.split_type = None
        
        experiments = {
            #YouCook2
            'YouCook2': self.youcook2,

            #VaTeX
            'VaTeX': self.vatex,

            #COIN
            'COIN': self.coin
        }
        
        if self.experiment_name in experiments.keys():
            experiments[self.experiment_name]()
        elif dataset is not None:
            experiments[dataset](self.experiment_name)
        elif self.experiment_name in Experiment2VideoMapping.get_dataset_experiments('YouCook2', cvpr2020_subset=False):
            experiments['YouCook2'](self.experiment_name)
        elif self.experiment_name in Experiment2VideoMapping.get_dataset_experiments('VaTeX'):
            experiments['VaTeX'](self.experiment_name)
        elif self.experiment_name in Experiment2VideoMapping.get_dataset_experiments('COIN'):
            experiments['COIN'](self.experiment_name)

    @staticmethod
    def get_dataset_experiments(dataset_name, cvpr2020_subset=False):
        if dataset_name == 'YouCook2':
            if cvpr2020_subset:
                experiments = set([
                    'GLd3aX16zBg',
                    # 'TkK7BbBNPaY', ## UNAVAILABLE ON YOUTUBE
                    '7-WEdqJBXoQ',
                    '5I-_uJQ0t1o',
                    '6gObQR5Vm4M',
                    'QpDxIXV6VTE',
                    'mKIvvx1SwmQ',
                    'IdEZ7LvLZPE',
                    'ny7G1uw36J0',
                    'zuQfLg46-Yc',
                    'GridojtCXDE',
                    'cppB7IXFySk',
                    'T2lxCGJ9ekg',
                    'JlXYqpEWUuA',
                    'Kkhvy9rQHaQ',
                    '14w_h9VFtzU',
                    'm20wLqgdmLY',
                    '92cezdsHEwM',
                    '7Fd7DjXMeaQ',
                    'AoPDhr5qkxY',
                    # 'e-GfjjZMabA', ## UNAVAILABLE ON YOUTUBE
                    '-yfTO7V2d_E',
                    'v-GtKLHLmsU',
                    'Db6SYxAfmPM',
                    'Eg89rR5s8e4',
                    'Iq1Sn9vERcU',
                    '01lB162koHA',
                    '6Rq7O6sX6ds',
                    '6Mi3xrBF1sY',
                    # '2bGOopmLtk4', ## UNAVAILABLE ON YOUTUBE
                    'FplIx3-XZvs',
                    'BgndHaHcHE0',
                    # '1BYKKEvxcVo', ## UNAVAILABLE ON YOUTUBE
                    # '_H0Mmr15_zA', ## UNAVAILABLE ON YOUTUBE
                    # 'fIRadWyEofw', ## UNAVAILABLE ON YOUTUBE
                    # '_Nmu6ezDy-M', ## UNAVAILABLE ON YOUTUBE
                    # '_j-0ElzS88Q', ## UNAVAILABLE ON YOUTUBE
                    '4VQ-etDf0Z8',
                    'W4nvDoCtdHM',
                    '1m17Yoh73uU',
                    'P7XOVPrxEaQ',
                    'JdDL-ekwq2A',
                    '2lHFKR2r65U',
                    'HWdaqQP5460',
                    'ZazBiZ6ktfk',
                    '5riUSC1fRMI',
                    'fR1qLJ1P4DI',
                    'htIpLVWrs0U',
                    'olrxEUXmlVA',
                    'SkPvNb9P7XQ',
                    '1OtWo8cXKWM',
                    '0JVmVXLrNZo',
                    '40UqljqGXXA',
                    # 'OOX012L_cXA', ## UNAVAILABLE ON YOUTUBE
                    '1xjsM-I-KLI',
                    'U0jn_DI5ESg',
                    'HK92pViSZA0',
                    'qxSUbk742Ag',
                    '0-OWf7eul6w',
                    '1HK-p8abRq8',
                    'Dggrreb1T30',
                    '8ZX3Lazhkp4',
                    # 'MJ2mD3blxqA', ## UNAVAILABLE ON YOUTUBE
                    'v-NzE_1_xNM',
                    'r4VSQuNE6D4',
                    # 'ItEqiHzbLj4', ## UNAVAILABLE ON YOUTUBE
                    'FiE1KczH4pc',
                    'AWVaYpPFJqk',
                    'AFc6KPGfVs8',
                    # '5xogySLxeWQ', ## UNAVAILABLE ON YOUTUBE
                    # 'HwLNy9MV6AQ', ## UNAVAILABLE ON YOUTUBE
                    'LA6DXaQ5vGQ',
                    'GaxyzK2mHqw',
                    'Gq9rHij2z20',
                    # 'UesiELtgiVk', ## UNAVAILABLE ON YOUTUBE
                    'tqrCEx9np9Y',
                    'KfAq4KRIVs4',
                    'cFcLgFsiZUE',
                    'vUhoMXc7FJM',
                    'lU28_L508vo',
                    'lBnuFn9q3Xs',
                    'I2AbLUtNSMI',
                    'g0KDpc1kSjo',
                    # 'k38Al8giI-U', ## UNAVAILABLE ON YOUTUBE
                    '-geDRZmY-E8',
                    'VmxFWJkYAqk',
                    'BFpJtwAG-8A',
                    'KF6W7hSjLYI',
                    'ISdbVLH2jO8',
                    'FtHLUsOntqI',
                    'WAevYUItUAY',
                    'D4AnZ0ymfzw',
                    'iX5UqDbD9YE',
                    'vTE9fobspEw',
                    'Acqpfz6lQc4',
                    'p1RgI4R8VX4',
                    'k3nRPKCyyVg',
                    'XsALTvYUTI8',
                    'Avx4fwzRYX4',
                    
                    #Validation == Test
                    'fn9anlEL4FI',
                    '-dh_uGahzYo',
                    'sSO2wO-yaHw',
                    'xkKuIlYSMMU',
                    'CWxjNRIKjA0',
                    # 'BAoQWVV-bh4', ## UNAVAILABLE ON YOUTUBE
                    'jT75QMjRkD0',
                    '-Ju39A-G0Dk',
                    'W6DgS0s0qcI',
                    'GmkRlWA2kGI',
                    # '1vJp-jaIaeE', ## UNAVAILABLE ON YOUTUBE
                    '5Pa79r5Q-ZI',
                    'QUt050AXQMw',
                    '9ekEjxd-A_Y',
                    'yWEq4_EG1us',
                    '7r6JQycloEs',
                    'lwdypoLpMW4',
                    'R3Jc1fXwSnU',
                    'wQc0xmPurDc',
                    'nVERaEFJWLQ',
                    'We2CzpjPD3k',
                    'RHddz6qeJKk',
                    'oYLrSflCI2g',
                    '-ju7_ZORsZw',
                    'Ky0zf0v2F5A',
                    'ZQGfcC62Pys',
                    'YNpVeU1pVZA',
                    # '524UzHtbAcY', ## UNAVAILABLE ON YOUTUBE
                    'SkawoKeyNoQ'
                ])
            else:
                experiments = [line.split('/')[-1] for line in np.loadtxt(f'{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/rl_fast_forward/resources/YouCook2/splits/train_list.txt', delimiter='\n', dtype=str, encoding='utf-8')]

                experiments.extend([line.split('/')[-1] for line in np.loadtxt(f'{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/rl_fast_forward/resources/YouCook2/splits/val_list.txt', delimiter='\n', dtype=str, encoding='utf-8')])
                
                available_experiments = set([fname.split('/')[-1] for fname in glob(f'{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/rl_fast_forward/resources/YouCook2/raw_videos/*/*/*') if len(fname.split('/')[-1].split('.')) == 1])
                
                experiments = list(set(experiments).intersection(available_experiments))
        
        elif dataset_name == 'VaTeX':
            vals = json.load(open(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/resources/vatex_validation_v1.0.json'))
            experiments = list(set([val['videoID'] for val in vals]))  # Validation videos

        elif dataset_name == 'COIN':
            experiments = set(pd.read_csv(f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/rl_fast_forward/resources/COIN/metadata.csv')['vid_id'].tolist())
            
            available_experiments = set([fname.split('/')[-1].split('.mp4')[0] for fname in glob(f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/rl_fast_forward/resources/COIN/videos/*/*mp4')])
                
            experiments = list(set(experiments).intersection(available_experiments))
            
        return experiments

    def youcook2(self, video_name):
        self.dataset = 'YouCook2'
        self.video_name = video_name
        annotations = json.load(open(f'{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/rl_fast_forward/resources/YouCook2/youcookii_annotations_trainval.json'))

        self.split_type = annotations['database'][self.video_name]['subset']
        self.recipe_id = annotations['database'][self.video_name]['recipe_type']
        self.video_filename = f'{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/rl_fast_forward/resources/YouCook2/raw_videos/{self.split_type}/{self.recipe_id}/{self.video_name}'

        duration_totalframe = pd.read_csv(f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/rl_fast_forward/resources/YouCook2/splits/val_duration_totalframe.csv') if self.split_type == 'validation' else pd.read_csv(f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/rl_fast_forward/resources/YouCook2/splits/train_duration_totalframe.csv')

        self.semantics = 'food'
        self.fps = int(duration_totalframe[duration_totalframe['vid_id'] == self.video_name]['total_frame']) / float(duration_totalframe[duration_totalframe['vid_id'] == self.video_name]['duration']) 
        self.start_frame = 1
        self.end_frame = int(duration_totalframe[duration_totalframe['vid_id'] == self.video_name]['total_frame'])
        self.num_frames = self.end_frame - self.start_frame + 1
        self.document_filename = f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/rl_fast_forward/resources/YouCook2/recipes/recipe_{self.video_name}.txt'

    def coin(self, video_name):
        self.dataset = 'COIN'
        self.video_name = video_name
        annotations = json.load(open(f'{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/rl_fast_forward/resources/COIN/COIN.json'))

        self.split_type = annotations['database'][self.video_name]['subset']
        self.recipe_id = annotations['database'][self.video_name]['recipe_type']
        self.video_filename = f'{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/rl_fast_forward/resources/COIN/videos/{self.recipe_id}/{self.video_name}.mp4'

        duration_totalframe = pd.read_csv(f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/rl_fast_forward/resources/COIN/metadata.csv')

        self.semantics = 'instructions'
        self.fps = duration_totalframe[duration_totalframe['vid_id'] == self.video_name]['fps'].item()
        self.start_frame = 1
        self.end_frame = int(duration_totalframe[duration_totalframe['vid_id'] == self.video_name]['total_frame'])
        self.num_frames = self.end_frame - self.start_frame + 1
        self.document_filename = f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/rl_fast_forward/resources/COIN/instructions/instructions_{self.video_name}.txt'

    def vatex(self, video_name):
        videos_root = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/datasets/VaTeX/raw_videos/'
        feats_root = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/datasets/VaTeX/raw_videos/'
        self.dataset = 'VaTeX'
        self.video_name = video_name
        self.video_filename = os.path.join(videos_root, '{}.mp4'.format(video_name))
        self.feats_filename = os.path.join(feats_root, '{}.npy'.format(video_name))
        self.split_type = 'validation'
        self.semantics = 'general'
        video = cv2.VideoCapture(self.video_filename)
        self.frame_width = int(video.get(3))
        self.frame_height = int(video.get(4))
        self.fps = round(video.get(5))
        self.start_frame = 1
        self.end_frame = int(video.get(7))
        self.num_frames = self.end_frame - self.start_frame + 1
        self.document_filename = None
