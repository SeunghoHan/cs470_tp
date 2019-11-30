import os
import argparse

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn

from data.load_datasets import *
from models.decoder_with_attention import DecoderWithAttention
import models.model_parameters as params
from experiment._train_one_epoch import *
from experiment._validation_one_epoch import *
from experiment.utils import *
from experiment.earlystopping import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # Read caption word map
    cap_word_map_file = os.path.join(params.data_folder, 'WORDMAP_' + params.data_name + '.json')
    with open(cap_word_map_file, 'r') as j:
        cap_word_map = json.load(j)

    # Read attribute word map
    attr_word_map_file = os.path.join(params.data_folder, 'ATTRS_WORDMAP_' + params.data_name + '.json')
    with open(attr_word_map_file, 'r') as j:
        attr_word_map = json.load(j)


    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(params.data_folder, params.data_name, 'TRAIN'),
        batch_size=params.batch_size, 
        shuffle=True, 
        num_workers=params.workers, 
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(params.data_folder, params.data_name, 'VAL'),
        batch_size=params.batch_size, 
        shuffle=True, 
        num_workers=params.workers, 
        pin_memory=True)

    if args.ckpt == 'None':
        checkpoint = None
    else:
        checkpoint = os.path.join(params.ckpt_folder, args.ckpt)
    # Initialize / load checkpoint
    if checkpoint is None: 
        # Creaste new visual-semantic attention decoder
        start_epoch = 0
        vs_att_decoder = DecoderWithAttention(visual_attention_dim=params.visual_attention_dim,
                                              semantic_attention_dim=params.semantic_attention_dim,
                                              cap_embed_dim=params.cap_emb_dim,
                                              attr_embed_dim=params.attr_emb_dim,
                                              decoder_dim=params.decoder_dim,
                                              cap_vocab_size=len(cap_word_map),
                                              attr_vocab_size=len(attr_word_map),
                                              features_dim=params.features_dim,
                                              dropout=params.dropout_rate)
        decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, vs_att_decoder.parameters()))

    else:
        print("Loading checkpoint: {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        vs_att_decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']


    # Move to GPU, if available
    vs_att_decoder = vs_att_decoder.to(params.device)

    # Loss functions
    loss_fn_ce = nn.CrossEntropyLoss().to(params.device)
    loss_fn_dis = nn.MultiLabelMarginLoss().to(params.device)

    # For EarlyStopping & Decaying learning rate
    best_bleu4 = 0.  # BLEU-4 score right now
    earlystop = EarlyStopping(best_bleu4, params.max_patience)

    for epoch in range(start_epoch, params.epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if earlystop.patience > 0 and earlystop.patience % params.decay_epochs_interval == 0:
            earlystop.lr_decay(optimizer, 0.8)

        if earlystop.patience == earlystop.max_patience: break

        # One epoch's training
        train(train_loader=train_loader,
              vs_att_decoder=vs_att_decoder,
              loss_fn_ce=loss_fn_ce,
              loss_fn_dis=loss_fn_dis,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)


        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                vs_att_decoder=vs_att_decoder,
                                loss_fn_ce=loss_fn_ce,
                                loss_fn_dis=loss_fn_dis,
                                word_map=cap_word_map)

        # Check if there was an improvement
        is_improved = earlystop.check_improvement(recent_bleu4)

        state_dict = {'epoch': epoch,
                      'epochs_since_improvement': earlystop.patience,
                      'bleu-4': recent_bleu4,
                      'decoder': vs_att_decoder,
                      'decoder_optimizer': decoder_optimizer}

        ckpt_path = os.path.join(params.ckpt_folder, 
                                 'checkpoint_' + str(epoch+1) + '_' + params.data_name + '.pth.tar')
        torch.save(state_dict, ckpt_path)
        if is_improved:
            ckpt_path = os.path.join(params.ckpt_folder, 
                                     'BEST_checkpoint_' + str(epoch+1) + '_' + params.data_name + '.pth.tar')
            torch.save(state_dict, ckpt_path)
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', 
                        type=str,
                        help="Saved checkpoint name(e.g., BEST_checkpoint_6_preprocessed_coco.pth.tar, if you have no any checkpoint, write None")

    args = parser.parse_args()
    
    main(args)