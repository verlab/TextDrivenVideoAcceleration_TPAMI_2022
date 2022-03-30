import argparse
import json
import sys
import os
import numpy as np

from tqdm import tqdm
import pandas as pd


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Evaluation parameters')
    #
    parser.add_argument('-gt', '--ground_truth_filename', type=str, required=True, help="JSON File containing the ground truth frames.")
    parser.add_argument('-sf', '--selected_frames_filename', type=str, required=True, help='JSON with the selected frames.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='One of the following datasets [YouCook2, COIN]')
    parser.add_argument('-x', '--speedup', type=int, required=True, help='Inform the desired speedup rate if you want to preemptively stop the video when it reaches the desired number of frames.')

    return parser.parse_args(args)


def compute_f1_score(selected_frames, gt_frames, num_frames):

    # Prepare ground truth vec
    ground_truth = np.array([False]*num_frames, dtype=bool)
    ground_truth[gt_frames-1] = True

    # Prepare selected frames vec
    sf_binary = np.array([False]*num_frames, dtype=bool)

    sf_binary[selected_frames-1] = True

    # Compute Precision and Recall
    true_positives = sf_binary * ground_truth
    precision = np.sum(true_positives)/len(selected_frames)
    recall = np.sum(true_positives)/np.sum(ground_truth)

    f1_score = 2*(precision*recall)/(precision+recall) if np.sum(true_positives) else 0

    return precision, recall, f1_score


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    gt_json = json.load(open(args.ground_truth_filename))
    sf_json = json.load(open(args.selected_frames_filename))

    output_save_basename = os.path.splitext(args.selected_frames_filename)[0]
    json_save_filename = f'{output_save_basename}_results_@{args.speedup}x.json'
    csv_save_filename = f'{output_save_basename}_results_@{args.speedup}x.csv'

    if not args.dataset and 'dataset' in sf_json['info'].keys():
        args.dataset = sf_json['info']['dataset']

    scores_dict = {}
    recipe_id_tuple = []

    if args.dataset == 'YouCook2':
        video_ids = np.loadtxt(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/resources/YouCook2/splits/val_list.txt', dtype=str, delimiter='/', usecols=1, encoding='utf-8')
        metadata = pd.read_csv(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/resources/YouCook2/metadata.csv')
    elif args.dataset == 'COIN':
        annotations_filename = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/resources/COIN/COIN.json'
        annotations = json.load(open(annotations_filename))
        metadata = pd.read_csv(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/resources/COIN/metadata.csv')
        video_ids = np.array([video_id for video_id in metadata['vid_id'].tolist() if annotations['database'][video_id]['subset'] == 'testing'])

    video_ids = set(video_ids).intersection(gt_json['data'].keys())
    for video_id in tqdm(video_ids):
        if video_id not in sf_json['data'].keys():
            scores_dict[video_id] = {
                'precision': float('nan'),
                'recall': float('nan'),
                'f1_score': float('nan'),
                'speedup': float('nan'),
                'f1_speedup': float('nan'),
                'speedup_acc': float('nan')}
            recipe_id_tuple.append((video_id, gt_json['data'][video_id]['recipe_id']))
            continue

        num_frames = metadata[metadata['vid_id'] == video_id]['total_frame'].item()

        gt_frames = np.array(gt_json['data'][video_id]['frames'], dtype=int)
        sf = np.array(sf_json['data'][video_id]['frames'], dtype=int)
        num_selected_frames = len(sf)
        desired_num_frames = int(np.round(num_frames/args.speedup))

        # COMPUTING F1 SCORE
        precision, recall, f1_score = compute_f1_score(sf, gt_frames, num_frames)

        # COMPUTING SPEEDUP
        speedup = float(num_frames)/num_selected_frames
                
        scores_dict[video_id] = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'speedup': speedup}
        recipe_id_tuple.append((video_id, gt_json['data'][video_id]['recipe_id']))

    # pdb.set_trace()
    precisions = []
    recalls = []
    f1_scores = []
    speedups = []
    f = open(csv_save_filename, 'w')
    print('RECIPE_ID, VIDEO_ID, PRECISION, RECALL, F1 SCORE, SPEEDUP')
    f.write('RECIPE_ID, VIDEO_ID, PRECISION, RECALL, F1 SCORE, SPEEDUP\n')
    for tup in sorted(recipe_id_tuple, key=lambda x: (x[1], x[0])):
        video_id = tup[0]
        print_str = '{}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            tup[1],
            video_id, scores_dict[video_id]['precision'],
            scores_dict[video_id]['recall'],
            scores_dict[video_id]['f1_score'],
            scores_dict[video_id]['speedup'])
        precisions.append(scores_dict[video_id]['precision'])
        recalls.append(scores_dict[video_id]['recall'])
        f1_scores.append(scores_dict[video_id]['f1_score'])
        speedups.append(scores_dict[video_id]['speedup'])
        f.write('{}\n'.format(print_str))

    f.close()

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)
    speedups = np.array(speedups)
    print_str = '{}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
        '-', 'Average', np.nanmean(precisions),
        np.nanmean(recalls),
        np.nanmean(f1_scores),
        np.nanmean(speedups))
    print(print_str)
    print_str = '{}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
        '-', 'Std', np.nanstd(precisions),
        np.nanstd(recalls),
        np.nanstd(f1_scores),
        np.nanstd(speedups))
    print(print_str)

    with open(json_save_filename, 'w') as f:
        json.dump(scores_dict, f, sort_keys=True)

    print('JSON results file saved at {}'.format(json_save_filename))
    print('CSV results file saved at {}'.format(csv_save_filename))
