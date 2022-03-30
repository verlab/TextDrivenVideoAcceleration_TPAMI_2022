import json
import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    annotations = json.load(open('{}/COIN/COIN.json'.format(os.path.abspath(os.path.dirname(__file__)))))

    if not os.path.isdir('{}/COIN/instructions'.format(os.path.abspath(os.path.dirname(__file__)))):
        os.mkdir('{}/COIN/instructions'.format(os.path.abspath(os.path.dirname(__file__))))

    for video_id in tqdm(list(annotations['database'].keys())):
        np.savetxt('{}/COIN/instructions/instructions_{}.txt'.format(os.path.abspath(os.path.dirname(__file__)), video_id), [instruction['label'] for instruction in annotations['database'][video_id]['annotation']], fmt='%s', encoding='utf-8')
