import torch 
import torch.backends.cudnn as cudnn

# Data parameters
data_folder = 'preprocessed_data'  # folder with data files saved by create_input_files.py
data_name = 'preprocessed_coco'  # base name shared by data files
ckpt_folder = 'ckpt'

# Model parameters
features_dim = 2048
cap_emb_dim = 1024  # dimension of word embeddings
attr_emb_dim = 256  # dimension of attribute word embeddings
visual_attention_dim = 1024  # dimension of visual attention linear layers
semantic_attention_dim = 128  # dimension of visual attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
dropout_rate = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
epochs = 50  # number of epochs to train for (if early stopping is not triggered)
max_patience = 20
batch_size = 50
workers = 0 # for data-loading; right now, only 1 works with h5py
# checkpoint = None  # path to checkpoint, None if none
decay_epochs_interval = 8
log_interval = 100

# Evaluation parameters 
beam_size = 5
checkpoint = 'checkpoint_10_preprocessed_coco.pth.tar'