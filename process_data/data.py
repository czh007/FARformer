# -*- coding:utf-8 -*-

import csv
from nltk.corpus import stopwords
from string import punctuation
import os
from collections import Counter
import pickle

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
ROOT = 'ROOT'


def processEHR(str):
    # Preprocessing operations on strings
    str = str.lower()  # Convert all to lowercase
    str = ' '.join([word for word in str.split() if word not in stopwords.words('english')])  # Delete deactivated words
    str = ' '.join([word for word in str.split() if word not in punctuation])  # Remove punctuation
    str = ' '.join([word for word in str.split() if word.isalpha()])  # Remove number
    str = ' '.join([word for word in str.split() if len(word) >= 2])
    return str


class Data(object):
    def __init__(self, datafolder, vocab_size, labels, samples):
        self._word2id = {}
        self._id2word = {}

        # Read data
        patientWords = []  # All words in unstructured data
        self.patientDescribs = []
        self.labels = []

        if os.path.exists(os.path.join(datafolder, 'patientDescribes_labels_%d_sample%d.pkl' % (labels, samples))):
            with open(os.path.join(datafolder, 'patientDescribes_labels_%d_sample%d.pkl' % (labels, samples)),
                      'rb') as f:
                self.patientDescribs, self.labels = pickle.load(f)
            with open(os.path.join(datafolder, 'patientWords_%d_sample%d.pkl' % (labels, samples)), 'rb') as f:
                patientWords = pickle.load(f)
        else:

            with open(os.path.join(datafolder, 'filter_top_%d_sample%d.csv' % (labels, samples))) as f:
                reader = csv.reader(f)
                next(reader)
                data = [row for row in reader]
                for row in data:
                    str = processEHR(row[2])
                    self.patientDescribs.append(str)
                    patientWords.extend(str.split())
                    self.labels.append(row[3].strip())

            # Save the processed data for a second use
            with open(os.path.join(datafolder, 'patientDescribes_labels_%d_sample%d.pkl' % (labels, samples)),
                      'wb') as f:
                pickle.dump([self.patientDescribs, self.labels], f)
            with open(os.path.join(datafolder, 'patientWords_%d_sample%d.pkl' % (labels, samples)), 'wb') as f:
                pickle.dump(patientWords, f)

        # Sort the patientWords by frequency and add the most frequent max_size-2 words to the vocab
        patientWords = Counter(patientWords)
        sortedWords = patientWords.most_common(vocab_size - 2)

        self._word2id = {w: i + 2 for i, (w, c) in enumerate(sortedWords)}  # 索引从2开始
        self._word2id[PAD_TOKEN] = 0
        self._word2id[UNKNOWN_TOKEN] = 1
        self._id2word = {id: w for w, id in self._word2id.items()}


    def word2id(self, word):
        if word not in self._word2id:
            return self._word2id[UNKNOWN_TOKEN]
        return self._word2id[word]

    def id2word(self, word_id):
        if word_id not in self._id2word:
            raise ValueError('Id not found in vocab:%d' % word_id)
        return self._id2word[word_id]

    def size(self):
        return len(self._word2id)
