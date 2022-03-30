from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import argparse
import json
import os
import numpy as np

dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/semantic_encoding/_utils/'
if dir_name not in sys.path:
    sys.path.append(dir_name)
from experiment_to_video_mapping import Experiment2VideoMapping as exp2v

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Evaluation parameters')
    parser.add_argument('-gt', '--ground_truth_filename', type=str, required=True, help="File containing a list of the ground truth frames for every video. [JSON format]")
    parser.add_argument('-sfs', '--selected_frames_filenames', nargs='+', type=str, required=True, help='List of files that contains the selected frames from the every algorithm. [JSON format]')
    parser.add_argument('-ns', '--names', nargs='+', type=str, required=True, help='Algorithms names e.g: cvpr, ffnet, ours, ...')
    parser.add_argument('-e', '--experiment_id', type=str, required=True, help="ID of the video to be used in the image generation")
    parser.add_argument('-f', '--format', type=str, default='png', help="The format of the output image. [png, pdf]")
    parser.add_argument('-p', '--plot_format', type=str, default='rug', help="The format of the output image. [rug, dist, kde]")
    parser.add_argument('-o', '--output_folder', type=str, default='./', help="Folder to store the image")

    return parser.parse_args(args)


def save_coverage_fig(video_id, sfs, names, frames_gt, num_frames, fmt, plot_format, output_folder):

    plt.rc('xtick', labelsize=48)
    colors = ['darkgoldenrod', 'forestgreen', 'steelblue', 'firebrick', 'mediumpurple', 'red', 'blue']

    plt.clf()
    fig, axes = plt.subplots(len(sfs)+1, 1, sharex=True, figsize=(100, 4*(len(sfs) + 1)))

    for i, sf in enumerate(tqdm(sfs)):
        if plot_format == 'rug':
            ax = sns.rugplot(sf, height=1, color=colors[i], ax=axes[i], label=names[i])
        elif plot_format == 'dist':
            ax = sns.distplot(sf, kde=True, color=colors[i], ax=axes[i], label=names[i], lw=2)
        elif plot_format == 'kde':
            ax = sns.kdeplot(sf, shade=True, color=colors[i], ax=axes[i], label=names[i], lw=2)
        else:
            print('Plot format error!')
            exit(1)
        ax.set_xlim(0, num_frames)
        ax.get_yaxis().set_visible(False)

    ax = sns.rugplot(frames_gt, height=1, color='black', ax=axes[-1], label='GT')
    ax.set_xlim(0, num_frames)
    ax.get_yaxis().set_visible(False)

    print('Saving...')
    fig.legend(loc='center left', fontsize=72)
    plt.savefig('{}/{}.{}'.format(output_folder, video_id, fmt), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print('Saved!')

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    if (len(args.names) != len(args.selected_frames_filenames)):
        print('# Method Names != # Selected frames files')
        exit(1)

    if args.experiment_id[0] == '\\':
        args.experiment_id = args.experiment_id[1:]

    ground_truth = json.load(open(args.ground_truth_filename))

    selected_frames_methods = {args.names[i]: json.load(open(sf_fname)) for i, sf_fname in enumerate(args.selected_frames_filenames)}

    gt_sf = np.array(ground_truth['data'][args.experiment_id]['frames'], dtype=int)
    sfs = []
    video_exp = exp2v(args.experiment_id)
    num_frames = video_exp.num_frames
    for method in selected_frames_methods.keys():
        sf = np.array(selected_frames_methods[method]['data'][args.experiment_id]['frames'], dtype=int)
        sfs.append(sf)

    save_coverage_fig(args.experiment_id, sfs, args.names, gt_sf, video_exp.num_frames, args.format, args.plot_format, args.output_folder)
