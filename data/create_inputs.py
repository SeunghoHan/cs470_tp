import os
import numpy as np
import h5py
import json
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pickle

def create_input_files(dataset, karpathy_json_path, attribute_path, captions_per_image, min_word_freq, pkl_folder, output_folder, max_len=100):
    """
    Creates input files for training, validation, and test data.
    :param dataset: name of dataset. Since bottom up features only available for coco, we use only coco
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    
    with open(os.path.join(pkl_folder,'train36_imgid2idx.pkl'), 'rb') as j:
        train_data = pickle.load(j)
        
    with open(os.path.join(pkl_folder,'val36_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)
    
    # Read image paths and captions for each image
    train_image_captions = []
    val_image_captions = []
    test_image_captions = []
    train_image_det = []
    val_image_det = []
    test_image_det = []
    
    coco_attributes = {}
    train_attributes = []
    val_attributes = []
    test_attributes = []
    attribute_dic = []
    
    img_urls = {}
    train_img_urls = []  
    val_img_urls = []  
    test_img_urls = []   
    
    word_freq = Counter()
    
    # Load attributes of train_image_dataset 
    train_attribute_path = os.path.join(attribute_path, 'train_attributes.txt')
    with open(train_attribute_path, 'r') as f_train:
        while True: 
            first_line = f_train.readline()
            if not first_line: break
                
            coco_url = first_line.split(' ')[1]
            file_name = coco_url.split('/')[4]
            file_name = file_name.split('.')[0]
            
            coco_url = coco_url.replace(' ', '')
            img_urls[file_name] = coco_url.replace('\n', '')
            
            
            second_line = f_train.readline()
            coco_attributes[file_name] = second_line.replace(' ', '')
            
            
    val_attribute_path = os.path.join(attribute_path, 'val_attributes.txt')
    with open(val_attribute_path, 'r') as f_val:
        while True: 
            first_line = f_val.readline()
            if not first_line: break
                
            coco_url = first_line.split(' ')[1]
            file_name = coco_url.split('/')[4]
            file_name = file_name.split('.')[0]
            coco_url = coco_url.replace(' ', '')
            img_urls[file_name] = coco_url.replace('\n', '')
            
            second_line = f_val.readline()
            coco_attributes[file_name] = second_line.replace(' ', '')
            
    for img in data['images']:
        att_key = img['filename'].split('.')[0]
        if att_key in coco_attributes:
            atts = coco_attributes[att_key]
            atts_list = []
            for att in atts.split(','):
                if att == '\n': continue
                att_word = att.replace(',', '')
                if att_word == '': continue
                atts_list.append(att_word)
                attribute_dic.append(att_word)
        else:
            continue
            
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue
        
        image_id = img['filename'].split('_')[2]
        image_id = int(image_id.lstrip("0").split('.')[0])
        
        if img['split'] in {'train', 'restval'}:
            if img['filepath'] == 'train2014':
                if image_id in train_data:
                    train_image_det.append(("t",train_data[image_id]))
            else:
                if image_id in val_data:
                    train_image_det.append(("v",val_data[image_id]))
            train_image_captions.append(captions)
            train_attributes.append(atts_list)
            train_img_urls.append(img_urls[att_key])
        elif img['split'] in {'val'}:
            if image_id in val_data:
                val_image_det.append(("v",val_data[image_id]))
            val_image_captions.append(captions)
            val_attributes.append(atts_list)
            val_img_urls.append(img_urls[att_key])
        elif img['split'] in {'test'}:
            if image_id in val_data:
                test_image_det.append(("v",val_data[image_id]))
            test_image_captions.append(captions)
            test_attributes.append(atts_list)
            test_img_urls.append(img_urls[att_key])
    
    
    # Sanity check
    assert len(train_image_det) == len(train_image_captions)
    assert len(val_image_det) == len(val_image_captions)
    assert len(test_image_det) == len(test_image_captions)
    
    
    base_filename = 'preprocessed_' + dataset
    
    
    # Save image urls to a JSON
    with open(os.path.join(output_folder, 'TRAIN' + '_IMG_URLS_' + base_filename + '.json'), 'w') as j:
        json.dump(train_img_urls, j)
        
    with open(os.path.join(output_folder, 'VAL' + '_IMG_URLS_' + base_filename + '.json'), 'w') as j:
        json.dump(val_img_urls, j)
        
    with open(os.path.join(output_folder, 'TEST' + '_IMG_URLS_' + base_filename + '.json'), 'w') as j:
        json.dump(test_img_urls, j)
    
    # Remove duplicate attribute
    attribute_dic = list(set(attribute_dic))
    # Create attribute word map
    attribute_words = {k: v for v, k in enumerate(attribute_dic)}
    
    # Save word map to a JSON
    with open(os.path.join(output_folder, 'ATTRS_WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(attribute_words, j)
        
    for atts_list, split in [(train_attributes, 'TRAIN'),
                             (val_attributes, 'VAL'),
                             (test_attributes, 'TEST')]:
        
        enc_attributes = []
        for i, atts in enumerate(tqdm(atts_list)):
            enc_att = [attribute_words[att_word] for att_word in atts]
            enc_attributes.append(enc_att)
            
        with open(os.path.join(output_folder, split + '_ATTRIBUTES_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_attributes, j)     
            

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
   
    # Create a base/root name for all output files
    #base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    
    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
        
    
    for impaths, imcaps, split in [(train_image_det, train_image_captions, 'TRAIN'),
                                   (val_image_det, val_image_captions, 'VAL'),
                                   (test_image_det, test_image_captions, 'TEST')]:
        enc_captions = []
        caplens = []
        
        for i, path in enumerate(tqdm(impaths)):
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)
                
            # Sanity check
            assert len(captions) == captions_per_image
            
            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)
        
        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)
    
    # Save bottom up features indexing to JSON files
    with open(os.path.join(output_folder, 'TRAIN' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(train_image_det, j)
        
    with open(os.path.join(output_folder, 'VAL' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(val_image_det, j)
        
    with open(os.path.join(output_folder, 'TEST' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(test_image_det, j)
