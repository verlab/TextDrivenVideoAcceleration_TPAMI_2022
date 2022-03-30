""" Deep RL Algorithms for OpenAI Gym environments
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_encoding.utils import load_checkpoint
from semantic_encoding._utils.experiment_to_video_mapping import Experiment2VideoMapping
from REINFORCE.agent import Agent
from envs import VideoEnvironment
import argparse
import numpy as np
import socket
import torch
import json
import random
import pandas as pd
import config
from tqdm import tqdm
from datetime import datetime as dt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(123)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # This is for debugging errors only (https://lernapparat.de/debug-device-assert/), uncomment it when necessary
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    
    # General training input
    parser.add_argument('-s', '--semantic_encoder_model_filename', type=str, required=True, help='Semantic encoder model path filename')
    parser.add_argument('-m', '--reinforce_model_filename', type=str, help='Policy Model path filename to resume training... (If not provided, the agent learns from scratch)')
    parser.add_argument('-febs', '--feature_extraction_batch_size', type=int, default=16, help="Batch size for the VDAN/VDAN+ feature extraction")
    parser.add_argument('-n', '--nb_epochs', type=int, default=100, help="Number of training epochs")
    
    # Dataset params
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Name of the dataset that will be used for training [e.g., YouCook2, COIN]")
    parser.add_argument('--cvpr2020_subset', dest='cvpr2020_subset', action='store_true', help="Use the CVPR'20 YouCook2's subset (videos with at most 25 percent composed of relevant frames)")
    parser.add_argument('-dm', '--domain', type=str, dest='domain', help="One of the following COIN domains: ['NC', 'V', 'LP', 'G', 'EA', 'FD', 'SC', 'PF', 'DS', 'D', 'S', 'H']")
    
    # Policy Learning
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='Gamma value (γ ∈ (0, 1]) - The discounting factor')
    parser.add_argument('-eb', '--entropy_beta', type=float, default=1e-2, help='Beta value (β) for the entropy H')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='Learning rate for any algorithm')
    parser.add_argument('-cr', '--critic_lr', type=float, default=1e-3, help='Learning rate for the Critic branch')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='Size of the batch of videos for training the agent (default is 1)')
    parser.add_argument('-lb', '--lambda', dest='lambda_param', type=str, default='F*', help="Define the value for λ (which controls the relative importance of the overall speed-up rate). [F* (desired_num_frames) | F (num_frames) | 0 (Ignore Speed-up Reward)]")
    parser.add_argument('-sg', '--sigma', dest='sigma_param', type=float, default=0.5, help="Define the value for σ (which controls how thin the bell curve w.r.t. overall speed-up rate will be - standard deviation of a gauss function). Default: 0.5")
    
    # Ablation params
    parser.add_argument('--use_vdan', dest='use_vdan', action='store_true', help="Define the usage of VDAN feature extractor instead of VDAN+")
    parser.add_argument('--no_skip_info', dest='no_skip_info', action='store_true', help="Disable the usage of SA")
    parser.add_argument('--no_positional_encoding', dest='no_positional_encoding', action='store_true', help="Disable the usage of NRPE")

    parser.set_defaults(cvpr2020_subset=False, use_vdan=False, no_positional_encoding=False, no_skip_info=False)
    return parser.parse_args(args)


def train(args, training_envs, annotations):
    hostname = socket.gethostname()
    aux_env = list(training_envs.values())[0]

    agent = Agent(state_size=aux_env.state_size, action_size=aux_env.action_size, lr=args.learning_rate, critic_lr=args.critic_lr, gamma=args.gamma, entropy_beta=args.entropy_beta, annotations=annotations)

    if args.reinforce_model_filename is None:
        model_path = f'models/{agent.creation_timestamp}_{hostname}_{args.dataset.lower()}_{"vdan" if aux_env.use_vdan else "vdan+"}_{"no_positional_encoding_" if aux_env.no_positional_encoding else ""}{"no_skip_info_" if aux_env.no_skip_info else ""}lr{agent.lr}_gm{agent.gamma}_lambda{args.lambda_param}_eps{args.nb_epochs}/'
    else:
        agent.load_model(args.reinforce_model_filename)
        model_path = f'{args.reinforce_model_filename.split("checkpoint")[0]}'

        print('\n[{}] Resuming training from epoch {}...'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), agent.curr_epoch))

    agent.train(envs=training_envs, dataset_name=args.dataset, n_epochs=args.nb_epochs, batch_size=args.batch_size, model_path=model_path)
    
    print(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}] Done!\n')


if __name__ == "__main__":
    # Parse arguments
    args = parse_args(sys.argv[1:])

    print(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}] Loading saved model weights: {args.semantic_encoder_model_filename}...')

    _, semantic_encoder_model, _, word_map, model_params, train_params = load_checkpoint(args.semantic_encoder_model_filename, is_vdan=args.use_vdan)
    semantic_encoder_model.to(device)
    semantic_encoder_model.eval()

    semantic_encoder_data = {
        'model_name': os.path.basename(os.path.splitext(args.semantic_encoder_model_filename)[0]),
        'semantic_encoder_model': semantic_encoder_model,
        'word_map': word_map,
        'train_params': train_params,
        'model_params': model_params,
        'input_frames_length': 32        
    }
    print(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}] Done!\n')

    experiments = np.array(Experiment2VideoMapping.get_dataset_experiments(args.dataset, cvpr2020_subset=args.cvpr2020_subset))
    random.shuffle(experiments)

    annotations = json.load(open(config.annotations_filenames[args.dataset]))

    training_set = []
    if args.dataset == 'YouCook2':
        for exp_name in tqdm(experiments, desc='Loading training set'):
            if annotations['database'][exp_name]['subset'] == 'training':
                training_set.append(exp_name)
                
    elif args.dataset == 'COIN':
        if not args.domain:
            print('\nPlease provide a domain using the option -dm\n')
            exit(1)
            
        taxonomy = pd.read_csv(config.coin_taxonomy_filename, dtype=str)
        coin_domains_map = {'Nursing and Care': 'NC', 'Vehicle': 'V', 'Leisure and Performance': 'LP', 'Gadgets': 'G', 'Electrical Appliance': 'EA', 'Furniture and Decoration': 'FD', 'Science and Craft': 'SC', 'Pets and Fruit': 'PF', 'Drink and Snack': 'DS', 'Dish': 'D', 'Sport': 'S', 'Housework': 'H'}
        taxonomy_dict = {j: coin_domains_map[i] for i, j in zip(taxonomy['Domains'], taxonomy['Targets'])}

        for exp_name in tqdm(experiments, desc='Loading training set'):
            if annotations['database'][exp_name]['subset'] == 'training' and taxonomy_dict[annotations['database'][exp_name]['class']] == args.domain:
                training_set.append(exp_name)

    training_envs = {
        exp_key: VideoEnvironment(semantic_encoder_data, dataset_name=args.dataset, experiment_name=exp_key, batch_size=args.feature_extraction_batch_size, use_vdan=args.use_vdan, no_positional_encoding=args.no_positional_encoding, no_skip_info=args.no_skip_info, lambda_param=args.lambda_param, sigma_param=args.sigma_param) 
        for exp_key in tqdm(training_set, desc='Loading training set envs')
    }
    
    print(f'\nTraining set ({len(training_set)} videos):\n{", ".join(training_envs.keys())}\n')

    train(args, training_envs, annotations)
