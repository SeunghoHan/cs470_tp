import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from experiment.utils import *
import models.model_parameters as params

def validate(val_loader, vs_att_decoder, loss_fn_ce, loss_fn_dis, word_map):
    vs_att_decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad(): 
        for i, (_, imgs, caps, caplens, attributes, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(params.device)
            caps = caps.to(params.device)
            caplens = caplens.to(params.device)
            attributes = attributes.to(params.device)

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
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss_d_v = loss_fn_dis(scores_d_v, targets_d_v.long())
            loss_d_s = loss_fn_dis(scores_d_s, targets_d_s.long())
            loss_g = loss_fn_ce(scores, targets)
            loss = loss_g + (5 * loss_d_v) + (5 * loss_d_s)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % params.log_interval == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    bleu4 = round(bleu4,4)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4