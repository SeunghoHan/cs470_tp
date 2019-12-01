import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import json
from nlgeval import NLGEval

import models.model_parameters as params

# Load word map (word2ix)
word_map_file = os.path.join(params.data_folder, 'WORDMAP_' + params.data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Load model
torch.nn.Module.dump_patches = True
nlgeval = NLGEval()  # loads the evaluator


def make_prediction(image_features, caps, caplens, attributes, allcaps, checkpoint):
    checkpoint = torch.load(os.path.join(params.ckpt_folder, checkpoint), map_location = params.device)
    vs_att_decoder = checkpoint['decoder']
    
    vs_att_decoder = vs_att_decoder.to(params.device)
    vs_att_decoder.eval()

    references = list()
    hypotheses = list()

    image_features = image_features.to(params.device)
    caps = caps.to(params.device)
    caplens = caplens.to(params.device)
    attributes = attributes.to(params.device)

    k = params.beam_size

    # Move to GPU device, if available
    image_features = image_features.to(params.device)  # (1, 3, 256, 256)
    image_features_mean = image_features.mean(1)
    image_features_mean = image_features_mean.expand(k, params.features_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(params.device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(params.device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1

    h1_v, c1_v = vs_att_decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
    h1_s, c1_s = vs_att_decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
    h2, c2 = vs_att_decoder.init_hidden_state(k)  # (batch_size, decoder_dim)

    attr_embedding = vs_att_decoder.attr_embedding(attributes)
    attr_embedding_mean = attr_embedding.mean(1)
    attr_embedding_mean = attr_embedding_mean.expand(k, params.attr_emb_dim)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        cap_embeddings = vs_att_decoder.cap_embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        h1_v, c1_v = vs_att_decoder.top_down_visual_attention(
            torch.cat([h2, image_features_mean, cap_embeddings], dim=1),
            (h1_v, c1_v))  # (batch_size_t, decoder_dim)

        visual_attention_weighted_encoding = vs_att_decoder.visual_attention(image_features, h1_v)

        h1_s, c1_s = vs_att_decoder.top_down_sematic_attention(
            torch.cat([h2, attr_embedding_mean, cap_embeddings], dim=1), 
            (h1_s, c1_s))

        semantic_attention_weighted_encoding = vs_att_decoder.semantic_attention(attr_embedding_mean, h1_s)


        h2, c2 = vs_att_decoder.language_model(
            torch.cat([visual_attention_weighted_encoding, h1_v, semantic_attention_weighted_encoding, h1_s], dim=1), (h2, c2))

        scores = vs_att_decoder.fc(h2)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        h1_v = h1_v[prev_word_inds[incomplete_inds]]
        c1_v = c1_v[prev_word_inds[incomplete_inds]]
        h1_s = h1_s[prev_word_inds[incomplete_inds]]
        c1_s = c1_s[prev_word_inds[incomplete_inds]]
        h2 = h2[prev_word_inds[incomplete_inds]]
        c2 = c2[prev_word_inds[incomplete_inds]]
        image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
        attr_embedding_mean = attr_embedding_mean[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    # References
    img_caps = allcaps[0].tolist()
    img_captions = list(
        map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps))  # remove <start> and pads
    img_caps = [' '.join(c) for c in img_captions]
    references.append(img_caps)

    # Hypotheses
    hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
    hypothesis = ' '.join(hypothesis)
    hypotheses.append(hypothesis)
    assert len(references) == len(hypotheses)

    # Calculate scores
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    
    return references, hypotheses, metrics_dict

