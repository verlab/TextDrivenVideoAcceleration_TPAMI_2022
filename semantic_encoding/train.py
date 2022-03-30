#!/usr/bin/python3

import config

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as IT
import torchvideo.transforms as VT
from torchvision.transforms import Compose
import torch.optim as optim

from models import VDAN_PLUS
from utils import AverageMeter, computeMRR, load_embeddings_matrix, load_checkpoint, save_checkpoint, init_weights
from data_loaders.vatex_dataloader import VaTeXDataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import time
import multiprocessing
import random
import os

import warnings
warnings.filterwarnings("ignore")

WORKERS = int(multiprocessing.cpu_count())  # number of workers for loading data in the DataLoader
PRINT_FREQ = 100  # print training or validation status every __ batches

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

KINECTS400_MEAN = [0.43216, 0.394666, 0.37645]
KINECTS400_STD = [0.22803, 0.22145, 0.216989]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cudnn.benchmark = True # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_sets(word_map, train_params, model_params):
    if train_params['do_random_horizontal_flip']:
        train_transform = Compose([
            VT.NDArrayToPILVideo(),
            VT.ResizeVideo(train_params['resize_size']),
            VT.RandomCropVideo(train_params['random_crop_size']),
            VT.RandomHorizontalFlipVideo(p=0.5),
            VT.PILVideoToTensor(),
            VT.NormalizeVideo(mean=KINECTS400_MEAN, std=KINECTS400_STD)
        ])
    else:
        train_transform = Compose([
            VT.NDArrayToPILVideo(),
            VT.ResizeVideo(train_params['resize_size']),
            VT.CenterCropVideo(train_params['random_crop_size']),
            VT.RandomHorizontalFlipVideo(p=0.5),
            VT.PILVideoToTensor(),
            VT.NormalizeVideo(mean=KINECTS400_MEAN, std=KINECTS400_STD)
        ])

    val_transform = Compose([
        VT.NDArrayToPILVideo(),
        VT.ResizeVideo(train_params['resize_size']),
        VT.CenterCropVideo(train_params['random_crop_size']),
        VT.PILVideoToTensor(),
        VT.NormalizeVideo(mean=KINECTS400_MEAN, std=KINECTS400_STD)
    ])

    # Data location and settings
    training_data = VaTeXDataLoader(root=train_params['train_data_path'],
                                    annFile=train_params['captions_train_fname'],
                                    word_map=word_map,
                                    vid_transform=train_transform,
                                    annotations_transform=IT.ToTensor(),
                                    num_sentences=train_params['max_sents'],
                                    max_words=train_params['max_words'],
                                    dataset_proportion=train_params['train_data_proportion'],
                                    training_data=True,
                                    num_input_frames=model_params['num_input_frames'])

    validation_data = VaTeXDataLoader(root=train_params['val_data_path'],
                                      annFile=train_params['captions_val_fname'],
                                      word_map=word_map,
                                      vid_transform=val_transform,
                                      annotations_transform=IT.ToTensor(),
                                      num_sentences=train_params['max_sents'],
                                      max_words=train_params['max_words'],
                                      dataset_proportion=train_params['val_data_proportion'],
                                      training_data=False,
                                      num_input_frames=model_params['num_input_frames'])

    # Data loaders
    training_dataloader = torch.utils.data.DataLoader(training_data,
                                                      batch_size=train_params['train_batch_size'],
                                                      num_workers=WORKERS,
                                                      worker_init_fn=np.random.seed(123),
                                                      shuffle=True)

    validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                        batch_size=train_params['val_batch_size'],
                                                        num_workers=WORKERS,
                                                        worker_init_fn=np.random.seed(123),
                                                        shuffle=False)

    return training_dataloader, validation_dataloader, training_data, validation_data


def train(training_dataloader, training_data, model, criterion, optimizer, epoch, writer):
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss

    start = time.time()

    num_batches = len(training_dataloader)
    for i, (vids_paths, captions_docs, vids, documents, sentences_per_document, words_per_sentence, labels) in enumerate(training_dataloader):

        data_time.update(time.time() - start)

        vids = vids.to(device)
        documents = documents.squeeze(1).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        # Forward prop.
        vids_embeddings, texts_embeddings, word_alphas, sentence_alphas, _ = model(vids, documents, sentences_per_document, words_per_sentence)
        
        # Loss
        L_enc = criterion(vids_embeddings, texts_embeddings, labels)  ## Apply Eq. 1 from the paper: Cosine Embedding Loss

        # Back prop.
        optimizer.zero_grad()
        L_enc.backward()

        # Update
        optimizer.step()
        
        # Keep track of metrics
        losses.update(L_enc.item(), labels.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print training status
        if i % PRINT_FREQ == 0 or i == num_batches-1:
            print('[{0}] Epoch: [{1}][{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                epoch+1, i, num_batches,
                                                                batch_time=batch_time,
                                                                data_time=data_time,
                                                                loss=losses))

            writer.add_scalar('Batch_Loss/train', losses.val, epoch*num_batches + i)
            
    writer.add_scalar('Epoch_Loss/train', losses.avg, epoch)

    return losses.avg


def validate(validation_dataloader, validation_data, model, criterion, epoch, writer):
    model.eval()  # training mode enables dropout

    # UNCOMMENT TO PERFORM VALIDATION
    val_batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    val_data_time = AverageMeter()  # data loading time per batch
    val_losses = AverageMeter()  # cross entropy loss

    val_start = time.time()

    num_batches = len(validation_dataloader)

    val_dots = np.ndarray((len(validation_data),), dtype=np.float32)

    positive_embeddings = []
    
    for i, (vids_paths, captions_docs, vids, documents, sentences_per_document, words_per_sentence, labels) in enumerate(validation_dataloader):

        val_data_time.update(time.time() - val_start)

        vids = vids.to(device)
        documents = documents.squeeze(1).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        # Forward prop.
        vids_embeddings, texts_embeddings, word_alphas, sentence_alphas, _ = model(vids, documents, sentences_per_document, words_per_sentence)

        # Loss
        loss = criterion(vids_embeddings, texts_embeddings, labels)  # scalar

        vids_embeddings = vids_embeddings.detach().cpu()
        texts_embeddings = texts_embeddings.detach().cpu()

        val_dots[i * train_params['val_batch_size']: (i + 1) * train_params['val_batch_size']] = np.dot(vids_embeddings, texts_embeddings.T).diagonal() / (np.linalg.norm(vids_embeddings, axis=1) * np.linalg.norm(texts_embeddings, axis=1))

        # Keep track of metrics
        val_losses.update(loss.item(), labels.size(0))
        val_batch_time.update(time.time() - val_start)

        val_start = time.time()

        # Print training status
        if i % PRINT_FREQ == 0:
            print('\tEpoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch+1, i, len(validation_dataloader),
                                                                batch_time=val_batch_time,
                                                                data_time=val_data_time,
                                                                loss=val_losses))
            
        positive_embeddings.extend(torch.stack([vids_embeddings[labels > 0], texts_embeddings[labels > 0]], dim=1))

    positive_embeddings = torch.stack(positive_embeddings, dim=0)
    val_MRR = np.mean(computeMRR(positive_embeddings[:, 0], positive_embeddings[:, 1]))
    
    writer.add_scalar('Epoch_Loss/val', val_losses.avg, epoch)
    writer.add_scalar('Epoch_MRR/val', val_MRR, epoch)
    writer.add_histogram('Val_Dots_Distribution', val_dots, epoch)

    print('\tEpoch: [{0}][{1}/{2}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\n\t'
          'Validation MRR: {3}'.format(epoch + 1, num_batches - 1,
                                       len(validation_dataloader), val_MRR,
                                       batch_time=val_batch_time,
                                       data_time=val_data_time,
                                       loss=val_losses))

    return val_losses.avg, val_MRR


def main(model_params, train_params):

    if not os.path.isdir(train_params['log_folder']):
        print('Log folder "{}" does not exist. We are attempting creating it... '.format(train_params['log_folder']))
        os.mkdir(train_params['log_folder'])
        print('Folder created!')

    if train_params['model_checkpoint_filename']:
        print('[{}] Loading saved model weights to finetune (or continue training): {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_params['model_checkpoint_filename']))
        _, model, optimizer_state_dict, word_map, model_params, train_params = load_checkpoint(train_params['model_checkpoint_filename'])

        datetimestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        writer = SummaryWriter(log_dir='{}/{}_{}_lr{}_{}eps_ft/'.format(train_params['log_folder'], datetimestamp, train_params['hostname'], train_params['learning_rate'], train_params['num_epochs']), filename_suffix='_{}'.format(datetimestamp))
    else:
        embeddings, word_map = load_embeddings_matrix(train_params['embeddings_filename'], model_params['word_embed_size'], train_params['use_random_word_embeddings'])

        vocab_size = len(word_map)

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
                           learn_first_hidden_vector=model_params['learn_first_hidden_vector'],
                           use_sentence_level_attention=model_params['use_sentence_level_attention'],
                           use_word_level_attention=model_params['use_word_level_attention'])

        # Init word embeddings layer with pretrained embeddings
        model.text_embedder.doc_embedder.sent_embedder.init_pretrained_embeddings(embeddings)
        model.vid_embedder.fine_tune(False)
        model.apply(init_weights)

        datetimestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        writer = SummaryWriter(log_dir='{}/{}_{}_lr{}_{}eps/'.format(train_params['log_folder'], datetimestamp, train_params['hostname'], train_params['learning_rate'], train_params['num_epochs']), filename_suffix='_{}'.format(datetimestamp))

    training_dataloader, validation_dataloader, training_data, validation_data = create_sets(word_map, train_params, model_params)

    if train_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])
    elif train_params['optimizer'] == 'SGD':
        optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])
    else:
        print(f'Optmizer not implemented: {train_params["optimizer"]}')

    # Loss functions
    criterion = train_params['criterion']

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    print(model)

    # Epochs
    curr_val_loss = float('inf')
    # curr_max_MRR = -float('inf')
    for epoch in range(0, train_params['num_epochs']):
        # One epoch's training
        train_loss = train(training_dataloader=training_dataloader,
                           training_data=training_data,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           writer=writer)

        val_loss, MRR = validate(validation_dataloader=validation_dataloader,
                                 validation_data=validation_data,
                                 model=model,
                                 criterion=criterion,
                                 epoch=epoch,
                                 writer=writer)

        if val_loss < curr_val_loss or epoch == 0:
            # Save checkpoint
            save_checkpoint(epoch+1, model, optimizer, word_map, datetimestamp, model_params, train_params)
            curr_val_loss = val_loss

if __name__ == '__main__':
    
    model_params = config.model_params
    train_params = config.train_params
    
    print('\nModel Params:\n', model_params)
    print('\nTrain Params:\n', train_params)
    print('\n')
    
    main(model_params, train_params)
    
    print('\nModel Params:\n', model_params)
    print('\nTrain Params:\n', train_params)
    print('\n')
