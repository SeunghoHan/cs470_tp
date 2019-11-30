import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import models.model_parameters as params
from experiment.utils import *

def train(train_loader, vs_att_decoder, loss_fn_ce, loss_fn_dis, decoder_optimizer, epoch):
    vs_att_decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (_, imgs, caps, caplens, attributes) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(params.device)
        caps = caps.to(params.device)
        caplens = caplens.to(params.device)
        attributes = attributes.to(params.device)
        
        # Forward prop.
        scores, scores_d_v, scores_d_s, caps_sorted, decode_lengths, sort_ind = vs_att_decoder(imgs, caps, caplens, attributes)
        
        #Max-pooling across predicted words across time steps for discriminative supervision
        scores_d_v = scores_d_v.max(1)[0]
        scores_d_s = scores_d_s.max(1)[0]

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        targets_d_v = torch.zeros(scores_d_v.size(0),scores_d_v.size(1)).to(params.device)
        targets_d_v.fill_(-1)
        
        targets_d_s = torch.zeros(scores_d_s.size(0),scores_d_s.size(1)).to(params.device)
        targets_d_s.fill_(-1)

        for length in decode_lengths:
            targets_d_v[:,:length-1] = targets[:,:length-1]
            targets_d_s[:,:length-1] = targets[:,:length-1]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss_d_v = loss_fn_dis(scores_d_v, targets_d_v.long())
        loss_d_s = loss_fn_dis(scores_d_s, targets_d_s.long())
        loss_g = loss_fn_ce(scores, targets)
        loss = loss_g + (5 * loss_d_v) + (5 * loss_d_s)
        
        
        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()
	
        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, vs_att_decoder.parameters()), 0.25)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % params.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            
            
