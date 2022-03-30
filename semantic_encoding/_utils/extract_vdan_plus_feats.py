
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import config
from experiment_to_video_mapping import Experiment2VideoMapping
from utils import load_checkpoint, extract_vdan_plus_feats
from datetime import datetime
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
    """
    Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-m', '--model_checkpoint_filename', type=str, required=True, dest='model_checkpoint_filename', help="Name (complete path) of the trained model (or checkpoint) file.")
    parser.add_argument('-v', '--video_filename', type=str, default=None, dest='video_filename', help="Filename of the video to extract VDAN+ features")
    parser.add_argument("-e", "--experiment", dest="experiment", default=None, type=str, help="Experiment Name.")
    parser.add_argument("-d", "--dataset", dest="dataset", default=None, type=str, help="The dataset to extract features from. [YouCook2, COIN, etc.]")
    parser.add_argument('-u', '--user_document_filename', type=str, default=None, dest='user_document_filename', help="Filename of the user text to extract RNN (GRU) features")
    parser.add_argument('-o', '--output_folder_path', type=str, default=None, dest='output_folder_path', help="Path for the output file")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, dest='batch_size', help="Batch size for the extraction")

    args = parser.parse_args()

    batch_size = args.batch_size
    document_filename = args.user_document_filename
    semantic_encoder_name = os.path.basename(os.path.splitext(args.model_checkpoint_filename)[0])

    if args.dataset:
        
        vid_feats_base_dir = '{}/{}/VDAN+/vid_feats/{}'.format(config.deep_feats_base_folder, args.dataset, semantic_encoder_name)
        doc_feats_base_dir = '{}/{}/VDAN+/doc_feats/{}'.format(config.deep_feats_base_folder, args.dataset, semantic_encoder_name)

        os.makedirs(vid_feats_base_dir, exist_ok=True)
        os.makedirs(doc_feats_base_dir, exist_ok=True)

        print('[{}] Loading saved model weights: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.model_checkpoint_filename))
        _, model, optimizer_state_dict, word_map, model_params, train_params = load_checkpoint(args.model_checkpoint_filename)
        model.to(device)
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        experiments = np.array(Experiment2VideoMapping.get_dataset_experiments(args.dataset))

        for experiment in tqdm(experiments, desc='Extracting feats...'):
            exp_map = Experiment2VideoMapping(experiment)
            input_video_filename = exp_map.video_filename
            
            if not os.path.exists(input_video_filename):
                continue
            
            document_filename = exp_map.document_filename
            user_doc_basename = os.path.basename(os.path.splitext(document_filename)[0])

            vid_feats_filename = '{}/{}_vid_feats.npz'.format(vid_feats_base_dir, exp_map.video_name)
            doc_feats_filename = '{}/{}/{}_doc_feats.npz'.format(doc_feats_base_dir, user_doc_basename, exp_map.video_name)

            os.makedirs(os.path.dirname(doc_feats_filename), exist_ok=True)

            document = np.loadtxt(document_filename, delimiter='\n', dtype=str, encoding='utf-8')

            vid_embeddings, doc_embeddings, cos, words_atts, sentences_atts = extract_vdan_plus_feats(model, train_params, model_params, word_map, input_video_filename, document_filename, batch_size, model_params['num_input_frames'], tqdm_leave=False)

            np.savez_compressed(vid_feats_filename, features=vid_embeddings, semantic_encoder_name=semantic_encoder_name)
            np.savez_compressed(doc_feats_filename, features=doc_embeddings, document=document, semantic_encoder_name=semantic_encoder_name, words_atts=words_atts, sentences_atts=sentences_atts)

    else:
        if args.experiment:
            exp_map = Experiment2VideoMapping(args.experiment)
            input_video_filename = exp_map.video_filename

            if not os.path.exists(input_video_filename):
                print('\nVideo for the specified experiment does not exist or is not available.\n')
                exit(1)
            
            if not args.user_document_filename:
                document_filename = exp_map.document_filename

            vid_feats_base_dir = '{}/{}/VDAN+/vid_feats/{}'.format(config.deep_feats_base_folder, exp_map.dataset, semantic_encoder_name)
            doc_feats_base_dir = '{}/{}/VDAN+/doc_feats/{}'.format(config.deep_feats_base_folder, exp_map.dataset, semantic_encoder_name)
            user_doc_basename = os.path.basename(os.path.splitext(document_filename)[0])

            os.makedirs(vid_feats_base_dir, exist_ok=True)
            os.makedirs(doc_feats_base_dir, exist_ok=True)

            vid_feats_filename = '{}/{}_vid_feats.npz'.format(vid_feats_base_dir, exp_map.video_name)
            doc_feats_filename = '{}/{}/{}_doc_feats.npz'.format(doc_feats_base_dir, user_doc_basename, exp_map.video_name)

            os.makedirs(os.path.dirname(doc_feats_filename), exist_ok=True)
        elif args.video_filename:
            if not document_filename:
                print('\nPlease provide a text document using the option -u\n')
                exit(1)
                
            input_video_filename = args.video_filename
            output_filename = args.output_folder_path if args.output_folder_path else os.path.dirname(input_video_filename)

            vid_feats_filename = '{}/{}_vid_feats.npz'.format(output_filename, os.path.basename(os.path.splitext(args.video_filename)[0]))
            doc_feats_filename = '{}/{}_{}_doc_feats.npz'.format(output_filename, os.path.basename(os.path.splitext(args.video_filename)[0]), os.path.basename(os.path.splitext(document_filename)[0]))

        print('[{}] Loading saved model weights: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.model_checkpoint_filename))
        _, model, optimizer_state_dict, word_map, model_params, train_params = load_checkpoint(args.model_checkpoint_filename)

        model.to(device)
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        document = np.loadtxt(document_filename, delimiter='\n', dtype=str, encoding='utf-8')
        
        print('[{}] Extracting feats...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        vid_embeddings, doc_embeddings, cos, words_atts, sentences_atts = extract_vdan_plus_feats(model, train_params, model_params, word_map, input_video_filename, document_filename, batch_size, model_params['num_input_frames'])

        print('[{}] Saving feats...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        np.savez_compressed(vid_feats_filename, features=vid_embeddings, semantic_encoder_name=semantic_encoder_name)
        np.savez_compressed(doc_feats_filename, features=doc_embeddings, document=document, semantic_encoder_name=semantic_encoder_name, words_atts=words_atts, sentences_atts=sentences_atts)

        print('[{}] Video feats saved to {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), vid_feats_filename))
        print('[{}] Document feats saved to {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), doc_feats_filename))
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
