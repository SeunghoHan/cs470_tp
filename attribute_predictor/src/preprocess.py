#from gensim.models import word2vec
import gensim.models.keyedvectors as word2vec
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from pycocotools.coco import COCO
import argparse
import os
import pickle
import progressbar
import nltk
import numpy as np

def count_words(annotations):
    img_count = {}
    img_captions = {}
    
    stop = set(nltk.corpus.stopwords.words('english'))
    print('Count word frequencies, number of annotations = {}'.format(len(annotations)))
    bar = progressbar.ProgressBar()
    for ann in bar(annotations):
        caption = ann['caption']
        image_id = ann['image_id']
        tokens = nltk.word_tokenize(caption)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        if image_id not in img_count:
            img_count[image_id] = {}
            img_captions[image_id] = [caption]
        else:
            img_captions[image_id].append(caption)

        for w in tokens:
            if w in img_count[image_id]:
                img_count[image_id][w] += 1
            else:
                img_count[image_id][w] = 1

    print('Training: number of images = {}'.format(len(img_count)))
    return img_count, img_captions

def calc_features(coco_train, img_count, img_captions, tagsPerImage):
    #model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
    text8_w2v_feature = os.path.join('features/text8_w2v_features/', 'text.model.bin')
    model = word2vec.KeyedVectors.load_word2vec_format(text8_w2v_feature, binary=True)
    net = VGG16(weights='imagenet', include_top=True)
    net.layers.pop()
    net.outputs = [net.layers[-1].output]
    net.layers[-1].outbound_nodes = []

    # TAGS_PER_IMAGE = args.tagsPerImage
    TAGS_PER_IMAGE = tagsPerImage
    print('Tags per image {}'.format(TAGS_PER_IMAGE))
    img_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 4096), dtype=np.float32)
    tag_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 200), dtype=np.float32)

    possible_tags = set()

    f = open('features/train_tags.txt', 'w')
    pos = 0
    print('Training: calculate image features, choose tag for each image')
    bar = progressbar.ProgressBar()
    for image_id, words in bar(img_count.items()):
        file_name = coco_train.imgs[image_id]['file_name']
        coco2014_dir = '/archive/MyHome/Programs/git/my_research/dataset/coco_2014/'
        coco2014_train_dir = os.path.join(coco2014_dir, 'train2014/')
        img = image.load_img(coco2014_train_dir + file_name, target_size=(224, 224))

        words_list = []
        words_count = []
        for w in words:
            if w in model.wv.vocab:
                words_list.append(w)
                words_count.append(img_count[image_id][w])

        words_count = np.array(words_count)
        index = np.argsort(words_count)[::-1]

        f.write(coco_train.imgs[image_id]['flickr_url'] + '\n')
        for i in range(TAGS_PER_IMAGE):
            f.write(words_list[ index[i] ] + '\n')
        for i in range(0,min(5,len(index))):
            ind = index[i]
            f.write(words_list[ind] + ', ' + str(words_count[ind]) + '\n')
        for caption in img_captions[image_id]:
            f.write(caption + '\n')

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = net.predict(img)
        features = features.reshape(-1)

        for i in range(TAGS_PER_IMAGE):
            ind = index[i]
            img_features[TAGS_PER_IMAGE * pos + i,:] = features
            tag_features[TAGS_PER_IMAGE * pos + i,:] = model[ words_list[ind] ]
        possible_tags.add(words_list[ index[0] ])

        pos += 1
        if pos % 20000 == 0:
            print('Training: saving features calculated for the first {} images'.format(pos))
            np.savez_compressed('features/features', img_features=img_features[:TAGS_PER_IMAGE * pos,:], tag_features=tag_features[:TAGS_PER_IMAGE * pos,:])

    print('Training: saving features calculated for all the images')
    np.savez_compressed('features/features', img_features=img_features, tag_features=tag_features)

    print('Training: number of possible tags = {}'.format(len(possible_tags)))
    pickle.dump(possible_tags, open('features/possible_tags.pkl', 'wb'))