import torch
from torch import nn
from models.attentions import VisualAttention, SemanticAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderWithAttention(nn.Module):
    def __init__(self, visual_attention_dim, semantic_attention_dim, cap_embed_dim, attr_embed_dim, decoder_dim, cap_vocab_size, attr_vocab_size, features_dim, dropout):
        """
        :param attention_dim: size of attention network
        :param cap_embed_dim: caption embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.features_dim = features_dim
        self.visual_attention_dim = visual_attention_dim
        self.semantic_attention_dim = semantic_attention_dim
        self.cap_embed_dim = cap_embed_dim
        self.decoder_dim = decoder_dim
        self.cap_vocab_size = cap_vocab_size
        self.attr_vocab_size = attr_vocab_size
        self.dropout = dropout
        # self.device = device
        
        # visual attention network
        self.visual_attention = VisualAttention(features_dim, 
                                                decoder_dim, 
                                                visual_attention_dim)  
        # semantic attention network
        self.semantic_attention = SemanticAttention(attr_embed_dim, 
                                                    decoder_dim, 
                                                    semantic_attention_dim)  
        
        self.attr_embedding = nn.Embedding(attr_vocab_size, attr_embed_dim)
        
        # caption embedding layer
        self.cap_embedding = nn.Embedding(cap_vocab_size, cap_embed_dim)  
        
        self.dropout = nn.Dropout(p=self.dropout)
        # top down visual attention LSTMCell
        self.top_down_visual_attention = nn.LSTMCell(cap_embed_dim + features_dim + decoder_dim, 
                                                     decoder_dim, 
                                                     bias=True) 
        # top down semantic attention LSTMCell
        self.top_down_sematic_attention = nn.LSTMCell(cap_embed_dim + attr_embed_dim + decoder_dim,
                                                      decoder_dim, 
                                                      bias=True) 
        # language model LSTMCell
        self.language_model = nn.LSTMCell(features_dim + decoder_dim + attr_embed_dim + decoder_dim,
                                          decoder_dim, 
                                          bias=True)  
        
        self.fc_v = nn.utils.weight_norm(nn.Linear(decoder_dim, cap_vocab_size))
        self.fc_s = nn.utils.weight_norm(nn.Linear(decoder_dim, cap_vocab_size))
        
        # linear layer to find scores over vocabulary
        self.fc = nn.utils.weight_norm(nn.Linear(decoder_dim, cap_vocab_size))  
        
        # initialize some layers with the uniform distribution
        self.init_weights()  

    def init_weights(self):
        self.cap_embedding.weight.data.uniform_(-0.1, 0.1)
        self.attr_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).to(device)
        return h, c
    
    def init_semantic_hidden_state(self, batch_size):
        h_s = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c_s = torch.zeros(batch_size, self.decoder_dim).to(device)
        return h_s, c_s
    
    def forward(self, image_features, encoded_captions, caption_lengths, encoded_attributes):
        batch_size = image_features.size(0)
        cap_vocab_size = self.cap_vocab_size
        attr_vocab_size = self.attr_vocab_size

        # Flatten image
        image_features_mean = image_features.mean(1).to(device)  # (batch_size, num_pixels, encoder_dim)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding for attributes
        attr_embedding = self.attr_embedding(encoded_attributes) # (batch_size, 5, attr_embedding_dim)
        attr_embedding_mean = attr_embedding.mean(1).to(device) # (batch_size, attr_embedding_dim)
        
        # Embedding for captions
        cap_embeddings = self.cap_embedding(encoded_captions)  # (batch_size, max_caption_length, cap_embed_dim)

        # Initialize LSTM state
        h1_v, c1_v = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h1_s, c1_s = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), cap_vocab_size).to(device)
        predictions_v = torch.zeros(batch_size, max(decode_lengths), cap_vocab_size).to(device)
        predictions_s = torch.zeros(batch_size, max(decode_lengths), cap_vocab_size).to(device)
        
        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up 
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model 
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # Visual-attention LSTM
            h1_v, c1_v = self.top_down_visual_attention(torch.cat([h2[:batch_size_t],
                                                                   image_features_mean[:batch_size_t],
                                                                   cap_embeddings[:batch_size_t, t, :]], dim=1), 
                                                        (h1_v[:batch_size_t], c1_v[:batch_size_t]))
            
            visual_attention_weighted_encoding = self.visual_attention(image_features[:batch_size_t], 
                                                                       h1_v[:batch_size_t])
            
            preds_v = self.fc_v(self.dropout(h1_v))
            
            # Semantic-attention LSTM
            h1_s, c1_s = self.top_down_sematic_attention(torch.cat([h2[:batch_size_t],
                                                                    attr_embedding_mean[:batch_size_t],
                                                                    cap_embeddings[:batch_size_t, t, :]], dim=1), 
                                                         (h1_s[:batch_size_t], c1_s[:batch_size_t]))
            
            semantic_attention_weighted_encoding = self.semantic_attention(attr_embedding_mean[:batch_size_t],  
                                                                           h1_s[:batch_size_t])
            
            preds_s = self.fc_s(self.dropout(h1_s))
            
            # features_dim + decoder_dim + attr_embed_dim + decoder_dim, decoder_dim,
            h2, c2 = self.language_model(
                torch.cat([visual_attention_weighted_encoding[:batch_size_t],
                           h1_v[:batch_size_t],
                           semantic_attention_weighted_encoding[:batch_size_t],
                           h1_s[:batch_size_t]], dim=1), (h2[:batch_size_t], c2[:batch_size_t]))

            
            
            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            
            predictions[:batch_size_t, t, :] = preds
            predictions_v[:batch_size_t, t, :] = preds_v
            predictions_s[:batch_size_t, t, :] = preds_s

        return predictions, predictions_v, predictions_s, encoded_captions, decode_lengths, sort_ind

