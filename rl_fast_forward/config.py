import os

annotations_filenames = {
    'YouCook2': f'{os.path.dirname(os.path.abspath(__file__))}/resources/YouCook2/youcookii_annotations_trainval.json',
    'COIN': f'{os.path.dirname(os.path.abspath(__file__))}/resources/COIN/COIN.json'
}

deep_feats_base_folder = '/srv/storage/datasets/semantic-hyperlapse/data/deep_features'
coin_taxonomy_filename = 'resources/COIN/taxonomy.csv'
logs_base_folder = 'logs/'