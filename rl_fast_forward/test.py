""" Deep RL Algorithms for OpenAI Gym environments
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_output_video
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
    parser.add_argument('-febs', '--feature_extraction_batch_size', type=int, default=16, help="Batch size for feature extraction")

    # General testing inputs
    parser.add_argument('-x', '--speedup', type=int, default=0, help='Desired speed-up to be achieved by the agent (WARNING: For training, leave it as default=0 for random selection)')
    parser.add_argument('-u', '--user_document_filename', type=str, default=None, help="Name (complete path) of the sentences document file")
    parser.add_argument('-i', '--input_video_filename', type=str, help="Name (complete path) of the input video file")
    parser.add_argument('-e', '--experiment', type=str, help="Name of the experiment video file [e.g. T2lxCGJ9ekg]")
    
    # Dataset params
    parser.add_argument('-d', '--dataset', type=str, help="Name of the dataset that will be used for training [e.g., YouCook2, COIN]")
    parser.add_argument('--cvpr2020_subset', dest='cvpr2020_subset', action='store_true', help="Use the CVPR'20 YouCook2's subset (videos with at most 25 percent composed of relevant frames)")
    parser.add_argument('-dm', '--domain', type=str, dest='domain', help="One of the following domains: ['NC', 'V', 'LP', 'G', 'EA', 'FD', 'SC', 'PF', 'DS', 'D', 'S', 'H']")
    
    # Agent params
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='Gamma value (γ ∈ (0, 1]) - The discounting factor')
    parser.add_argument('-eb', '--entropy_beta', type=float, default=1e-2, help='Beta value (β) for the entropy H')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='Learning rate for any algorithm')
    parser.add_argument('-cr', '--critic_lr', type=float, default=1e-3, help='Learning rate for the Critic branch')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='Size of the batch of videos for training the agent (default is 1)')
    parser.add_argument('-lb', '--lambda', dest='lambda_param', type=str, default='F*', help="Define the value for λ (which controls the relative importance of the overall speed-up rate). [F* (desired_num_frames) | F (num_frames) | 0 (Ignore Speed-up Reward)]")
    parser.add_argument('-sg', '--sigma', dest='sigma_param', type=float, default=0.5, help="Define the value for σ (which controls how thin the bell curve w.r.t. overall speed-up rate will be - standard deviation of a gauss function). Default: 0.5")
    
    # Ablation params
    parser.add_argument('--use_vdan', dest='use_vdan', action='store_true', help="Define the usage of VDAN feature extractor instead of VDAN+")
    parser.add_argument('--no_positional_encoding', dest='no_positional_encoding', action='store_true', help="Disable the usage of NRPE")
    parser.add_argument('--no_skip_info', dest='no_skip_info', action='store_true', help="Disable the usage of SA")

    # Video rendering and output params
    parser.add_argument('--create_video', dest='create_video', action='store_true', help="Create the output video based on the test environment")
    parser.add_argument('--print_details', dest='print_details', action='store_true', help="Print the speed-up and the groundtruth locations in the created video")

    parser.set_defaults(cvpr2020_subset=False, is_test=False, include_test=False, use_vdan=False, no_positional_encoding=False, no_skip_info=False, create_video=False, print_details=False)
    return parser.parse_args(args)


def test(args, test_envs, annotations):

    env = test_envs[list(test_envs.keys())[0]]
    
    agent = Agent(state_size=env.state_size, action_size=env.action_size, lr=args.learning_rate, critic_lr=args.critic_lr, gamma=args.gamma, entropy_beta=args.entropy_beta, annotations=annotations)
    agent.load_model(args.reinforce_model_filename)

    hostname = socket.gethostname()

    if args.experiment:
        json_sf = {'info': {'version': 'v1.1_{}'.format(dt.now().strftime('%Y%m%d_%H%M%S')), 'dataset': env.experiment.dataset}, 'data': {}}
        
        with torch.no_grad():
            agent.test(env, args.dataset, args.experiment, log=True)
        sf = env.selected_frames
                
        agent.write_selected_frames_image_to_log(env, 0, 'Test_{}'.format(args.experiment))
        agent.write_rewards_image_to_log(env, agent.rewards, 0, 'Rewards_Test_{}'.format(args.experiment))
        agent.writer.close()
        
        output_filename = f'results/{dt.now().strftime("%Y%m%d_%H%M%S")}_agent{agent.creation_timestamp}_lr{agent.lr}_gamma{agent.gamma}_{hostname}_sf_{args.experiment}.npy'
        
        print('Saved to: {}'.format(output_filename))

        if env.experiment.dataset in ['YouCook2', 'COIN']:
            json_sf['data'][args.experiment] = {'file_name': os.path.abspath(output_filename), 'recipe_id': env.experiment.recipe_id, 'split': 'test', 'frames': list(np.array(sf, dtype=str))}

        if args.create_video:
            create_output_video(args, env, output_filename)

    elif args.input_video_filename:
        video_basename = os.path.basename(args.input_video_filename).split('.')[0]
        
        print(f'\nTesting: {video_basename}')
        with torch.no_grad():
            agent.test(env, video_basename, video_basename, log=True)
            
        sf = env.selected_frames
        
        agent.write_selected_frames_image_to_log(env, 0, 'Test_{}'.format(video_basename))
        agent.write_rewards_image_to_log(env, agent.rewards, 0, 'Rewards_Test_{}'.format(video_basename))
        agent.writer.close()
        
        output_filename = f'results/{dt.now().strftime("%Y%m%d_%H%M%S")}_agent{agent.creation_timestamp}_lr{agent.lr}_gamma{agent.gamma}_{hostname}_sf_{video_basename}.npy'
        
        print(f'Saved to: {output_filename}')

        if args.create_video:
            create_output_video(args, env, output_filename)
    else:
        json_sf = {'info': {'version': 'v1.1_{}'.format(dt.now().strftime('%Y%m%d_%H%M%S')), 'dataset': args.dataset}, 'data': {}}
        curr_avg_speedup = np.array([np.float('nan')]*len(test_envs), dtype=np.float32)
        pbar = tqdm(test_envs.items(), desc=f'Testing Envs | Curr. Avg. Speedup -> {np.nanmean(curr_avg_speedup)}')
        for exp_idx, (exp_key, env) in enumerate(pbar):
            
            with torch.no_grad():
                agent.test(env, args.dataset, exp_key)
                
            sf = env.selected_frames
            curr_avg_speedup[exp_idx] = float(env.num_frames)/len(env.selected_frames)
            pbar.set_description(f'Testing Envs | Curr. Avg. Speedup -> {np.nanmean(curr_avg_speedup)}')
            
            agent.write_selected_frames_image_to_log(env, 0, 'Test_{}'.format(args.dataset), suffix=exp_key)
            agent.write_rewards_image_to_log(env, agent.rewards, 0, 'Rewards_Test_{}'.format(args.dataset), suffix=exp_key)
            agent.writer.close()

            output_filename = f'results/{args.dataset}/{dt.now().strftime("%Y%m%d_%H%M%S")}_agent{agent.creation_timestamp}_lr{agent.lr}_gamma{agent.gamma}_{hostname}_sf_{exp_key}.npy'

            if args.dataset in ['YouCook2', 'COIN']:
                json_sf['data'][exp_key] = {'file_name': os.path.abspath(output_filename), 'recipe_id': env.experiment.recipe_id, 'split': 'test', 'frames': list(np.array(sf, dtype=str))}

            if args.create_video:
                create_output_video(args, env, output_filename)

    if args.dataset in ['YouCook2', 'COIN']:
        output_json_filename = f'results/{agent.creation_timestamp}_{hostname}_{args.dataset.lower()}_ours_selected_frames{f"_{args.domain}" if args.domain else ""}_{args.speedup}x.json'
        
        with open(output_json_filename, 'w') as f:
            json.dump(json_sf, f, sort_keys=True)
            print(f'\nJSON results file saved at: {output_json_filename}')

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

    annotations = None
    if args.experiment:
        if args.experiment.startswith('\\'):
            args.experiment = args.experiment[1:]

        exp_map = Experiment2VideoMapping(args.experiment)
        
        annotations = json.load(open(config.annotations_filenames[exp_map.dataset]))

        env = VideoEnvironment(semantic_encoder_data, document_filename=args.user_document_filename, dataset_name=exp_map.dataset, experiment_name=args.experiment, batch_size=args.feature_extraction_batch_size, speedup=args.speedup, use_vdan=args.use_vdan, no_positional_encoding=args.no_positional_encoding, no_skip_info=args.no_skip_info, lambda_param=args.lambda_param, sigma_param=args.sigma_param)

        test_envs = {args.experiment: env}
        
        args.dataset = exp_map.dataset

    elif args.input_video_filename:
        if not args.user_document_filename:
            print('\nPlease provide a text document using the option -u\n')
            exit(1)
                
        video_basename = os.path.basename(args.input_video_filename).split('.')[0]

        env = VideoEnvironment(semantic_encoder_data, document_filename=args.user_document_filename, input_video_filename=args.input_video_filename, batch_size=args.feature_extraction_batch_size, speedup=args.speedup, use_vdan=args.use_vdan, no_positional_encoding=args.no_positional_encoding, no_skip_info=args.no_skip_info, lambda_param=args.lambda_param, sigma_param=args.sigma_param)

        test_envs = {video_basename: env}

        args.dataset = video_basename

    elif args.dataset:
        
        experiments = np.array(Experiment2VideoMapping.get_dataset_experiments(args.dataset, cvpr2020_subset=args.cvpr2020_subset))
        random.shuffle(experiments)

        annotations = json.load(open(config.annotations_filenames[args.dataset]))

        test_set = []
        if args.dataset == 'YouCook2':
            for exp_name in tqdm(experiments, desc='Loading training and test set'):
                if annotations['database'][exp_name]['subset'] != 'training':
                    test_set.append(exp_name)
                    
        elif args.dataset == 'COIN':
            if not args.domain:
                print('\nPlease provide a domain using the option -dm\n')
                exit(1)
            
            taxonomy = pd.read_csv(config.coin_taxonomy_filename, dtype=str)
            coin_domains_map = {'Nursing and Care': 'NC', 'Vehicle': 'V', 'Leisure and Performance': 'LP', 'Gadgets': 'G', 'Electrical Appliance': 'EA', 'Furniture and Decoration': 'FD', 'Science and Craft': 'SC', 'Pets and Fruit': 'PF', 'Drink and Snack': 'DS', 'Dish': 'D', 'Sport': 'S', 'Housework': 'H'}
            taxonomy_dict = {j: coin_domains_map[i] for i, j in zip(taxonomy['Domains'], taxonomy['Targets'])}

            for exp_name in tqdm(experiments, desc='Loading test set'):
                if annotations['database'][exp_name]['subset'] == 'testing' and taxonomy_dict[annotations['database'][exp_name]['class']] == args.domain:
                    test_set.append(exp_name)

        test_envs = {
            exp_key: VideoEnvironment(semantic_encoder_data, document_filename=args.user_document_filename, dataset_name=args.dataset, experiment_name=exp_key, batch_size=args.feature_extraction_batch_size, speedup=args.speedup, use_vdan=args.use_vdan, no_positional_encoding=args.no_positional_encoding, no_skip_info=args.no_skip_info, lambda_param=args.lambda_param, sigma_param=args.sigma_param)
            for exp_key in tqdm(test_set, desc='Loading test set envs')
        }
    
        print(f'\nTest set ({len(test_set)} videos):\n{", ".join(test_envs.keys())}\n')

    test(args, test_envs, annotations)
