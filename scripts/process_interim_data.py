"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import pickle
from collections import namedtuple

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import gensim
import gensim.corpora as corpora


def _read_text_file(path):
    with open(path, 'rt', encoding='utf-8') as file:
        # Read a list of strings.
        lines = file.readlines()
        # Concatenate to a single string.
        text = " ".join(lines)
    return text


def save_data(x, y, topic_names, data_dir, filename):
    with open(os.path.join(data_dir, filename), 'wb') as f:
        pickle.dump({'x': x, 'y': y, 'topic_names': topic_names}, f)


def load_data(data_dir, filename):
    with tf.gfile.Open(os.path.join(data_dir, filename), 'rb') as f:
        data_dict = pickle.load(f)
        x = data_dict['x']
        y = data_dict['y']
        topic_names = data_dict['topic_names']
    return x, y, topic_names

def create_vocab_list(x):
    data_words= [doc.split(" ") for doc in x]
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer 
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_words_bigrams = [bigram_mod[doc] for doc in data_words]
    # Create Dictionary
    dct = corpora.Dictionary(data_words_bigrams)
    words = [dct[i] for i in dct.cfs.keys()]
    print(len(words))
    counts = list(dct.cfs.values())
    zipped = zip(words, counts)
    zipped = sorted(zipped, key=lambda t: t[1], reverse=True)
    vocab_size = len(zipped)
    vocab_words = words
    print('Vocabulary size:', len(vocab_words))

    outVocab = {}
    for w in vocab_words:
        outVocab[w] = dct.cfs[dct.token2id[w]]
    

    with tf.gfile.Open("data/vocab_interim.pkl", "w+") as file1:
        pickle.dump(outVocab, file1)


def read_interim_data(data_dir):
    """
    Writes Newsgroups 20 from ana.cachopo.org/datasets-for-single-label-text-categorization
    into a suitable format.
    This version was also used in arxiv.org/abs/1511.01432 and arxiv.org/abs/1602.02373
    """

    data = pd.read_pickle(os.path.join(data_dir, "data_20200823.pkl"))
    x = list(data.Content)

    target_names = [" "]
    y = [0 for i in x]

    create_vocab_list(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    return namedtuple('Data', ['data', 'target', 'target_names'])(x_train, y_train, target_names), \
           namedtuple('Data', ['data', 'target', 'target_names'])(x_test, y_test, target_names)

def textPrecessing(text):
    text = text.lower()
    text = " ".join([word for word in text.split(" ") if not word.startswith("http")])
    text = re.sub('[^A-Za-z ]+', '', text)

    wordLst = nltk.word_tokenize(text)
    stop_words = stopwords.words('english')
    #word that found inrelevant when evaluation
    stop_words.extend(['said','would','also','say','says','saying'])
    filtered = [w for w in wordLst if w not in stop_words]

    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]

    return " ".join(filtered)

def run_nltk_word_tokenizer(data):

    x, y = [], []
    n_empty_docs = 0
    for i, doc in enumerate(data.data):
        tokenized_text = textPrecessing(doc)
        if i % 1000 == 0:
            print(i)
        if len(tokenized_text) == 0:
            n_empty_docs += 1
        else:
            x.append(tokenized_text)
            y.append(data.target[i])

    print('n total docs', len(data.data))
    print('n empty docs', n_empty_docs)

    return x, y


def preprocess_interim(data_dir):
    train_data, test_data = read_interim_data(data_dir)
    topic_names = train_data.target_names

    x, y = run_nltk_word_tokenizer(train_data)
    save_data(x, y, topic_names, data_dir, 'tokenized_interim_train.pkl')

    x, y = run_nltk_word_tokenizer(test_data)
    save_data(x, y, topic_names, data_dir, 'tokenized_interim_test.pkl')


def make_reference_corpus_for_topic_coherence(data_dir):
    """
    Writes Newsgroups 20 into a format suitable for running topics coherence evaluation code
    from github.com/jhlau/topic_interpretability
    """
    x_train, _, _ = load_data(data_dir, 'tokenized_ng20_train.pkl')
    x_test, _, _ = load_data(data_dir, 'tokenized_ng20_test.pkl')

    docs_train = [' '.join(doc) for doc in x_train]
    docs_test = [' '.join(doc) for doc in x_test]
    all_docs = docs_train + docs_test
    with open('reference_ng20.txt', 'a') as f:
        for doc in all_docs:
            f.write(doc + '\n')


if __name__ == '__main__':
    print('Processing interim data')
    preprocess_interim('data')

    # print('Making reference corpus for topics coherence')
    # make_reference_corpus_for_topic_coherence('data_debug')
