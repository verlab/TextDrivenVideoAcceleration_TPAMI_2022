import torch.nn as nn
import socket
import getpass

model_params = {
    'num_input_frames': 32,
    'word_embed_size': 300,
    'sent_embed_size': 512,  # h_ij
    'doc_embed_size': 512,  # h_i
    'hidden_feat_size': 512,
    'feat_embed_size': 128,  # d = 128. We also tested with 512 and 1024, but no substantial changes
    'sent_rnn_layers': 1,  # Not used in our paper, but feel free to change
    'word_rnn_layers': 1,  # Not used in our paper, but feel free to change
    'word_att_size': 1024,  # c_p
    'sent_att_size': 1024,  # c_d

    'use_sentence_level_attention': True,  # Not used in our paper, but feel free to change
    'use_word_level_attention': True,  # Not used in our paper, but feel free to change
    'use_visual_shortcut': True,  # Uses the R(2+1)D output as the first hidden state (h_0) of the document embedder Bi-GRU.
    'learn_first_hidden_vector': False  # Learns the first hidden state (h_0) of the document embedder Bi-GRU.
}

ETA_MARGIN = 0.  # η from Equation 1 - (Section 3.1.3 Training)

train_params = {

    # VaTeX
    'captions_train_fname': 'resources/vatex_training_v1.0.json', # Run semantic_encoding/resources/download_resources.sh first to obtain this file
    'captions_val_fname': 'resources/vatex_validation_v1.0.json', # Run semantic_encoding/resources/download_resources.sh first to obtain this file
    'train_data_path': 'datasets/VaTeX/raw_videos/', # Download all Kinetics-600 (10-seconds) validation videos using the semantic_encoding/resources/download_vatex_videos.sh script
    'val_data_path': 'datasets/VaTeX/raw_videos/', # Download all Kinetics-600 (10-seconds) validation videos using the semantic_encoding/resources/download_vatex_videos.sh script

    'embeddings_filename': 'resources/glove.6B.300d.txt', # Run semantic_encoding/resources/download_resources.sh first to obtain this file

    'max_sents': 20,  # maximum number of sentences per document
    'max_words': 20,  # maximum number of words per sentence

    # Training parameters
    'train_batch_size': 64, # We used a batch size of 64 (requires a 24Gb GPU card)
    'val_batch_size': 64, # We used a batch size of 64 (requires a 24Gb GPU card)
    'num_epochs': 100, # We ran in 100 epochs
    'learning_rate': 1e-5,
    'model_checkpoint_filename': None,  # Add an already trained model to continue training (Leave it as None to train from scratch)...

    # Video transformation parameters
    'resize_size': (128, 171),  # h, w
    'random_crop_size': (112, 112),  # h, w
    'do_random_horizontal_flip': True,  # Horizontally flip the whole video randomly in block

    # Training process
    'optimizer': 'Adam',
    'eta_margin': ETA_MARGIN,
    'criterion': nn.CosineEmbeddingLoss(ETA_MARGIN),

    # Machine and user data
    'username': getpass.getuser(),
    'hostname': socket.gethostname(),

    # Logging parameters
    'checkpoint_folder': 'models/',
    'log_folder': 'logs/',

    # Debugging helpers (speeding things up for debugging)
    'use_random_word_embeddings': False,  # Choose if you want to use random embeddings
    'train_data_proportion': 1.,  # Choose how much data you want to use for training
    'val_data_proportion': 1.,  # Choose how much data you want to use for validation
}

models_paths = {
    'VDAN': '<PATH/TO/THE/VDAN/MODEL>', # OPTIONAL: Provide the path to the VDAN model (https://github.com/verlab/StraightToThePoint_CVPR_2020/releases/download/v1.0.0/vdan_pretrained_model.pth) from the CVPR paper: https://github.com/verlab/StraightToThePoint_CVPR_2020/
    'VDAN+': '<PATH/TO/THE/VDAN+/MODEL>' # You must fill this path after training the VDAN+ to train the SAFFA agent
}

deep_feats_base_folder = '<PATH/TO/THE/VDAN+EXTRACTED_FEATS/FOLDER>' # Provide the location you stored/want to store your VDAN+ extracted feature vectorsfeature vectors