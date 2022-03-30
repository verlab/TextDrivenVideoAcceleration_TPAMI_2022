import sys
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind

import argparse
import numpy as np


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Evaluation parameters')
    parser.add_argument('-rfs', '--results_filenames', nargs='+', type=str, required=True, help='List of files that contains the results every algorithm. [CSV format]')
    parser.add_argument('-m', '--metric', type=str, required=True, help='One of the metrics in columns of the results files (e.g., Precision, Recall, F1 Score, Speed-up)')
    parser.add_argument('-ns', '--names', nargs='+', type=str, required=True, help='Methods Names e.g: cvpr, ffnet, ours, ...')
    parser.add_argument('-sd', '--speedup_deviation', type=int, help='Use the deviation to a certain speed-up as a value to be compared')
    parser.add_argument('-o', '--output_folder', type=str, default='./', help='Folder to save the outputs')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    results_dfs = []
    for results_filename in tqdm(args.results_filenames,'Loading CSVs'):
        results_dfs.append(pd.read_csv(results_filename))

    ttest_table = np.zeros((len(results_dfs), len(results_dfs)), dtype=np.float32)
    for i in tqdm(range(len(results_dfs))):
        i_values = np.array(results_dfs[i][f' {args.metric.upper()}'].tolist(), dtype=np.float32)
        if args.metric.upper() == 'SPEEDUP' and args.speedup_deviation is not None:
            i_values = np.abs(args.speedup_deviation - i_values)

        for j in range(i, len(results_dfs)):
            j_values = np.array(results_dfs[j][f' {args.metric.upper()}'].tolist(), dtype=np.float32)
            if args.metric.upper() == 'SPEEDUP' and args.speedup_deviation is not None:
                j_values = np.abs(args.speedup_deviation - j_values)
            ttest_table[i, j] = ttest_ind(i_values, j_values, nan_policy='omit')[1]
            ttest_table[j, i] = ttest_table[i, j]

    ttest_table = pd.DataFrame(ttest_table, index=args.names, columns=args.names)
    ttest_table_fname = f'{args.output_folder}/{"_".join(args.names)}_{args.metric}_ttest_p-values.csv'
    ttest_table.to_csv(ttest_table_fname)
    print(ttest_table)
    print(f'\nSaved to: {ttest_table_fname}')
