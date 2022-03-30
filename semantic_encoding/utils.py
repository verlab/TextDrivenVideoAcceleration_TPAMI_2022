import os
import sys
import cv2
import numpy as np
import nltk
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvideo.transforms as VT

from torchvision.transforms import Compose
from datetime import datetime
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from models import VDAN, VDAN_PLUS

np.random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

KINECTS400_MEAN = [0.43216, 0.394666, 0.37645]
KINECTS400_STD = [0.22803, 0.22145, 0.216989]

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings_matrix(embeddings_file, embeddings_dim, use_random_word_embeddings=False):

    # Creating representation for PAD and UNK
    if use_random_word_embeddings:
        vocabulary = np.concatenate([['PAD'], ['UNK']])
        vocab_size = len(vocabulary)

        word_map = {k: v for v, k in enumerate(vocabulary)}

        print('[{}] Loading fake word embeddings to speed-up debugging...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        embeddings = np.random.random((vocab_size, embeddings_dim)).astype(np.float32)
    else:
        print('[{}] Loading word embeddings and vocab...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        embeddings = [] 
        vocab = []
        embeddings.append(np.zeros(embeddings_dim))  # PAD vector
        embeddings.append(np.random.random(embeddings_dim))  # UNK vector (unknown word)
        f = open(embeddings_file, 'r', encoding='utf-8')
        for _, line in enumerate(tqdm(f)):
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            vocab.append(word)
            embeddings.append(embedding)
        embeddings = np.array(embeddings, dtype=np.float32)

        vocabulary = np.concatenate([['PAD'], ['UNK'], vocab])
        vocab_size = len(vocabulary)

        word_map = {k: v for v, k in enumerate(vocabulary)}

    embeddings = torch.from_numpy(embeddings)

    print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    return embeddings, word_map


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def save_checkpoint(epoch, model, optimizer, word_map, datetimestamp, model_params, train_params):
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'word_map': word_map,
             'model_params': model_params,
             'train_params': train_params}

    if not os.path.isdir(train_params['checkpoint_folder']):
        print('Folder "{}" does not exist. We are attempting creating it... '.format(train_params['checkpoint_folder']))
        os.mkdir(train_params['checkpoint_folder'])
        print('Folder created!')

    if train_params['model_checkpoint_filename']:
        filename = '{}/{}_checkpoint_lr{}_{}eps{}{}_{}_{}_ft{}.pth'.format(train_params['checkpoint_folder'], datetimestamp, train_params['learning_rate'], train_params['num_epochs'], '_w-att' if model_params['use_word_level_attention'] else '', '_s-att' if model_params['use_sentence_level_attention'] else '', train_params['hostname'], train_params['username'], train_params['model_checkpoint_filename'].split('/')[-1].split('.')[0])
    else:
        filename = '{}/{}_checkpoint_lr{}_{}eps{}{}_{}_{}.pth'.format(train_params['checkpoint_folder'], datetimestamp, train_params['learning_rate'], train_params['num_epochs'], '_w-att' if model_params['use_word_level_attention'] else '', '_s-att' if model_params['use_sentence_level_attention'] else '', train_params['hostname'], train_params['username'])

    print('\t[{}] Saving checkpoint file for epoch {}: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, filename))
    torch.save(state, filename)
    print('\t[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def load_checkpoint(filename, is_vdan=False):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filename, map_location=device)

    epoch = checkpoint['epoch']
    word_map = checkpoint['word_map']
    model_params = checkpoint['model_params']
    train_params = checkpoint['train_params']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    vocab_size = len(word_map)

    if is_vdan:
        model = VDAN(vocab_size=vocab_size,
                     doc_emb_size=model_params['doc_embed_size'],
                     sent_emb_size=model_params['sent_embed_size'],
                     word_emb_size=model_params['word_embed_size'],
                     hidden_feat_emb_size=model_params['hidden_feat_size'],
                     final_feat_emb_size=model_params['feat_embed_size'],
                     sent_rnn_layers=model_params['sent_rnn_layers'],
                     word_rnn_layers=model_params['word_rnn_layers'],
                     sent_att_size=model_params['sent_att_size'],
                     word_att_size=model_params['word_att_size'],
                     use_visual_shortcut=model_params['use_visual_shortcut'],
                     use_sentence_level_attention=model_params['use_sentence_level_attention'],
                     use_word_level_attention=model_params['use_word_level_attention'])
    else:
        model = VDAN_PLUS(vocab_size=vocab_size,
                          doc_emb_size=model_params['doc_embed_size'],
                          sent_emb_size=model_params['sent_embed_size'],
                          word_emb_size=model_params['word_embed_size'],
                          hidden_feat_emb_size=model_params['hidden_feat_size'],
                          final_feat_emb_size=model_params['feat_embed_size'],
                          sent_rnn_layers=model_params['sent_rnn_layers'],
                          word_rnn_layers=model_params['word_rnn_layers'],
                          sent_att_size=model_params['sent_att_size'],
                          word_att_size=model_params['word_att_size'],
                          use_visual_shortcut=model_params['use_visual_shortcut'],
                          use_sentence_level_attention=model_params['use_sentence_level_attention'],
                          use_word_level_attention=model_params['use_word_level_attention'],
                          learn_first_hidden_vector=model_params['learn_first_hidden_vector'] if 'learn_first_hidden_vector' in model_params.keys() else False)

    # Init word embeddings layer with random embeddings
    model.text_embedder.doc_embedder.sent_embedder.init_pretrained_embeddings(torch.rand(vocab_size, model_params['word_embed_size']))
    model.load_state_dict(model_state_dict)

    return epoch, model, optimizer_state_dict, word_map, model_params, train_params


def convert_sentences_to_word_idxs(sentences, max_words, word_map):
    converted_sentences = np.zeros((len(sentences), max_words), dtype=int)
    words_per_sentence = np.zeros((len(sentences),), dtype=int)
    for aid, annotation in enumerate(sentences):
        tokenized_annotation = nltk.tokenize.word_tokenize(annotation.lower())
        for wid, word in enumerate(tokenized_annotation[:min(len(tokenized_annotation), max_words)]):
            if word in word_map:
                converted_sentences[aid, wid] = word_map[word]
            else:
                converted_sentences[aid, wid] = word_map['UNK']

            words_per_sentence[aid] += 1  # Increment number of words

    return converted_sentences, words_per_sentence

def computeMRR(X, Y):
    
    norms_X = torch.norm(X, p=2, dim=1, keepdim=True)
    norms_Y = torch.norm(Y, p=2, dim=1, keepdim=True)

    X_norm = X.div(norms_X.expand_as(X))  # Normalize X
    Y_norm = Y.div(norms_Y.expand_as(Y))  # Normalize Y

    X_dot_Y = torch.mm(X_norm, Y_norm.T)

    max_cos_idxs = torch.argsort(X_dot_Y, dim=1, descending=True)
    MRR = np.mean(np.array([1/(idx_max.index(curr_idx)+1) for curr_idx, idx_max in enumerate(max_cos_idxs.tolist())]))

    return MRR

def extract_vdan_plus_feats(model, train_params, model_params, word_map, video_filename, document_filename, batch_size, max_frames, use_vid_transformer=False, tqdm_leave=True):
    # Load inputs
    document = np.loadtxt(document_filename, delimiter='\n', dtype=str, encoding='utf-8')

    video = cv2.VideoCapture(video_filename)
    num_frames = int(video.get(7))

    cos = np.zeros((num_frames,), dtype=np.float32)
    vid_embeddings = np.zeros((num_frames, model_params['feat_embed_size']), dtype=np.float32)
    doc_embeddings = np.zeros((num_frames, model_params['feat_embed_size']), dtype=np.float32)
    words_atts = np.zeros((num_frames, len(document), train_params['max_words']), dtype=np.float32)
    sentences_atts = np.zeros((num_frames, len(document)), dtype=np.float32)

    # Parameters preparation
    vid_transformer = Compose([
        VT.NDArrayToPILVideo(),
        VT.ResizeVideo(train_params['random_crop_size']),
        VT.PILVideoToTensor(),
        VT.NormalizeVideo(mean=KINECTS400_MEAN, std=KINECTS400_STD)
    ])

    # Preparing documents
    converted_sentences, words_per_sentence = convert_sentences_to_word_idxs(document, train_params['max_words'], word_map)

    documents = np.tile(converted_sentences, (batch_size, 1)).reshape(batch_size, converted_sentences.shape[0], converted_sentences.shape[1])

    documents = torch.from_numpy(documents).to(device)  # (batch_size, sentence_limit, word_limit)
    sentences_per_document = torch.from_numpy(np.tile([converted_sentences.shape[0]], batch_size)).to(device)  # (batch_size)
    words_per_sentence = torch.from_numpy(np.tile([words_per_sentence], batch_size).reshape(batch_size, -1)).to(device)  # (batch_size, sentence_limit)

    # First clip
    X = torch.zeros(((batch_size, 3, max_frames) + train_params['random_crop_size'])).to(device)

    ### Filling the first batch ###
    curr_frames = []
    for idx in range(max_frames+batch_size):
        ret, frame = video.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_frames.append(frame)

    clips = vid_transformer(curr_frames).unsqueeze(0).to(device)

    # Creating batch
    for i in range(batch_size):
        X[i] = clips[0, :, i:max_frames+i]

    with torch.no_grad():
        clip_feats, documents_feats, words_alphas, sentences_alphas, _ = model(X, documents, sentences_per_document, words_per_sentence)

    cos[:batch_size] = torch.mm(clip_feats, documents_feats.T).diag().detach().cpu().numpy()
    # Storing feats
    vid_embeddings[:batch_size] = clip_feats.detach().cpu().numpy()
    doc_embeddings[:batch_size] = documents_feats.detach().cpu().numpy()

    if words_alphas is not None:
        words_atts[:batch_size, : words_alphas.shape[1], : words_alphas.shape[2]] = words_alphas.detach().cpu().numpy()
    if sentences_alphas is not None:
        sentences_atts[:batch_size, :sentences_alphas.shape[1]] = sentences_alphas.detach().cpu().numpy()

    ### Filling the remaining batches ###
    num_remaining_batches = int(np.ceil((num_frames-batch_size)/float(batch_size)))
    # per_frame_run_times = []
    # start = timer()
    for batch_idx in tqdm(range(1, num_remaining_batches+1), desc=f'Extracting from {video_filename}', leave=tqdm_leave):
        current_batch_size = min(num_frames - batch_idx*batch_size, batch_size)

        curr_frames = []
        for idx in range(current_batch_size):
            ret, frame = video.read()

            if not ret:
                curr_frames.append(np.zeros((int(video.get(4)), int(video.get(3)), 3), dtype=np.uint8))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            curr_frames.append(frame)

        ## Left-shifting frames (IMPORTANT!) ##
        clips[:, :, :-current_batch_size] = clips[:, :, current_batch_size:].clone()

        curr_clips = vid_transformer(curr_frames).unsqueeze(0)
        clips[:, :, -current_batch_size:] = curr_clips.to(device)  # Adding new frames to the final gap

        # Creating new batch
        for i in range(current_batch_size):
            X[i] = clips[0, :, i:max_frames+i]

        with torch.no_grad():
            clip_feats, documents_feats, words_alphas, sentences_alphas, _ = model(X, documents, sentences_per_document, words_per_sentence)
            
        cos[batch_size*batch_idx:batch_size*batch_idx+current_batch_size] = torch.mm(clip_feats[:current_batch_size], documents_feats[:current_batch_size].T).diag().detach().cpu().numpy()

        # Storing feats
        vid_embeddings[batch_size*batch_idx:batch_size*batch_idx+current_batch_size] = clip_feats[:current_batch_size].detach().cpu().numpy()
        doc_embeddings[batch_size*batch_idx:batch_size*batch_idx+current_batch_size] = documents_feats[:current_batch_size].detach().cpu().numpy()

        if words_alphas is not None:
            words_atts[batch_size*batch_idx:batch_size*batch_idx+current_batch_size, : words_alphas.shape[1], : words_alphas.shape[2]] = words_alphas[:current_batch_size].detach().cpu().numpy()
        if sentences_alphas is not None:
            sentences_atts[batch_size*batch_idx:batch_size*batch_idx+current_batch_size, :sentences_alphas.shape[1]] = sentences_alphas[:current_batch_size].detach().cpu().numpy()

    video.release()
    return vid_embeddings, doc_embeddings, cos, words_atts, sentences_atts


def extract_vdan_feats(model, train_params, model_params, word_map, video_filename, document_filename, batch_size, max_frames=32, tqdm_leave=False):
    ## Load inputs
    document = np.loadtxt(document_filename, delimiter='\n', dtype=str, encoding='utf-8')

    video = cv2.VideoCapture(video_filename)
    num_frames = int(video.get(7))
    num_batches = int(np.ceil(num_frames/float(batch_size)))

    cos = np.zeros((num_frames,), dtype=np.float32)
    img_embeddings = np.zeros((num_frames, model_params['feat_embed_size']), dtype=np.float32)
    doc_embeddings = np.zeros((num_frames, model_params['feat_embed_size']), dtype=np.float32)
    words_atts = np.zeros((num_frames, len(document), train_params['max_words']), dtype=np.float32)
    sentences_atts = np.zeros((num_frames, len(document)), dtype=np.float32)

    ## Parameters preparation
    img_transform = T.Compose([T.Resize((224, 224)),
                               T.ToTensor(),
                               T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    ## Preparing documents
    converted_sentences, words_per_sentence = convert_sentences_to_word_idxs(document, train_params['max_words'], word_map)

    documents = np.tile(converted_sentences, (batch_size, 1)).reshape(batch_size, converted_sentences.shape[0], converted_sentences.shape[1])

    documents = torch.from_numpy(documents).to(device)  # (batch_size, sentence_limit, word_limit)
    sentences_per_document = torch.from_numpy(np.tile([converted_sentences.shape[0]], batch_size)).to(device)  # (batch_size)
    words_per_sentence = torch.from_numpy(np.tile([words_per_sentence], batch_size).reshape(batch_size, -1)).to(device)  # (batch_size, sentence_limit)

    for batch_idx in tqdm(range(num_batches), desc=f'Extracting from {video_filename}', leave=tqdm_leave):
        current_batch_size = min(num_frames - batch_idx*batch_size, batch_size)

        X = np.zeros((current_batch_size, 3, 224, 224), dtype=np.float32)
        for idx_j in range(current_batch_size):
            ret, frame = video.read()

            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            x = img_transform(frame)

            X[idx_j, :, :, :] = x

        X = torch.from_numpy(X).to(device)

        with torch.no_grad():
            _, imgs_out, docs_out, words_alphas_out, sentences_alphas_out = model(X[:current_batch_size], documents[:current_batch_size], sentences_per_document[:current_batch_size], words_per_sentence[:current_batch_size])

        cos[batch_size*batch_idx:batch_size*batch_idx+current_batch_size] = torch.mm(imgs_out, docs_out.T).diag().detach().cpu().numpy()

        # Storing feats
        img_embeddings[batch_size*batch_idx:batch_size*batch_idx+current_batch_size] = imgs_out[:current_batch_size].detach().cpu().numpy()
        doc_embeddings[batch_size*batch_idx:batch_size*batch_idx+current_batch_size] = docs_out[:current_batch_size].detach().cpu().numpy()

        if words_alphas_out is not None:
            words_atts[batch_size*batch_idx:batch_size*batch_idx+current_batch_size, : words_alphas_out.shape[1], : words_alphas_out.shape[2]] = words_alphas_out[:current_batch_size].detach().cpu().numpy()
        if sentences_alphas_out is not None:
            sentences_atts[batch_size*batch_idx:batch_size*batch_idx+current_batch_size, :sentences_alphas_out.shape[1]] = sentences_alphas_out[:current_batch_size].detach().cpu().numpy()

    video.release()
    return img_embeddings, doc_embeddings, cos, words_atts, sentences_atts


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
