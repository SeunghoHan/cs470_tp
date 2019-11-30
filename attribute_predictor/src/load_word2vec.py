from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
# import gensim.models as word2vec

import logging

def load_wv():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = Text8Corpus('text8')
    model = Word2Vec(sentences, size=200)
    model.save('features/text8_w2v_features/text8.model')
    model.wv.save_word2vec_format('features/text8_w2v_features/text.model.bin', binary=True)
