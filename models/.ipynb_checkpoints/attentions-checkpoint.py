import torch
from torch import nn

class VisualAttention(nn.Module):
    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        super(VisualAttention, self).__init__()
        self.features_att = nn.utils.weight_norm(nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = nn.utils.weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = nn.utils.weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden):
        att1 = self.features_att(image_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)

        return attention_weighted_encoding
    
class SemanticAttention(nn.Module):
    def __init__(self, attr_emb_dim, decoder_dim, attention_dim, dropout=0.5):
        super(SemanticAttention, self).__init__() #wTa
        self.attribute_att = nn.utils.weight_norm(nn.Linear(attr_emb_dim, attention_dim))  
        self.decoder_att = nn.utils.weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = nn.utils.weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, attr_embeddings, decoder_hidden):
        att1 = self.attribute_att(attr_embeddings)  # (batch_size, 5, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (attr_embeddings * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, attr_embeddings)

        return attention_weighted_encoding
    