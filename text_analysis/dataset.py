from __future__ import division, unicode_literals

import json
import math
import multiprocessing
import os
import time
from nltk.corpus import stopwords

import constants
import logging
logging.basicConfig(filename='/home/karthik/PycharmProjects/cmps242/project/project_cmps242.log', level=logging.DEBUG)


class DatasetStat(object):
    def __init__(self):
        self._review_stats = []
        self._terms = {}

    def get_review_stats(self):
        return self._review_stats

    def merge(self, dataset_stat):
        for review_stat in dataset_stat.get_review_stats():
            self._review_stats.append(review_stat)
            self._merge_terms(review_stat.get_term_counts())

    def _merge_terms(self, term_counts):
        for term, count in term_counts.iteritems():
            self._terms[term] = count + self._terms.get(term, 0)

    def add(self, review_stats):
        for review_stat in review_stats:
            self._review_stats.append(review_stat)
            term_counts = review_stat.get_term_counts()
            for term, count in term_counts.iteritems():
                self._terms[term] = count + self._terms.get(term, 0)

    def n_containing(self, term):
        return sum(1 for review_stat in self._review_stats if review_stat.has_term(term))

    def inverse_doc_freq(self, term):
        return math.log(len(self._review_stats) / (1 + self.n_containing(term)))

    def top_term_freq_prod_inv_doc_freq(self, count):
        scores = {}
        for review_stat in self._review_stats:
            scores = {term: self.term_freq_inv_doc_freq(term, review_stat) for term in review_stat.get_terms()}
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:count]

    def term_freq_inv_doc_freq(self, term, review_stat):
        return review_stat.term_freq(term) * self.inverse_doc_freq(term)

    # def __repr__(self):
    #     return '%s' % str(self._terms)


class ReviewStat(object):
    def __init__(self, exclude_stop_words=False):
        self._term_count = {}
        self._total_words = 0
        self._exclude_stop_words = exclude_stop_words

    def add(self, token):
        stripped_token = token.rstrip().lstrip()
        if self._exclude_stop_words:
            if stripped_token not in constants.STOP_WORDS:
                self._term_count[stripped_token] = self._term_count.get(stripped_token, 0) + 1
                self._total_words += 1
        else:
            self._term_count[stripped_token] = self._term_count.get(stripped_token, 0) + 1
            self._total_words += 1

    def get_terms(self):
        return self._term_count.keys()

    def get_term_counts(self):
        return self._term_count

    def has_term(self, term):
        return term in self._term_count

    def term_freq(self, term):
        return self._term_count.get(term, 0) / self._total_words

    # def __repr__(self):
    #     return 'Total:%d (%s)' % (self._total_words, str(self._term_count))


DATASET_STAT = DatasetStat()


class Dataset(object):
    """Dataset class"""

    def __init__(self, dataset_dir, parallelism=4, batch_size=100, limit=-1):
        self._dataset_dir = dataset_dir
        self._pool = multiprocessing.Pool(parallelism)
        self._batch_size = batch_size
        self._limit = limit

    def load(self):
        """
        Loads the data from the review dataset file.
        """
        review_data_file_path = os.path.join(self._dataset_dir, constants.REVIEW_DATA_FILE)
        count = 0
        results = []
        with open(review_data_file_path, 'r') as review_data_file:
            lines = []
            for line in review_data_file:
                count += 1
                if self._limit != -1 and count > self._limit:
                    break
                if len(lines) == self._batch_size:
                    results.append(self._pool.apply_async(process, [lines]))
                    lines = []
                else:
                    lines.append(line)
        start_time = time.time()
        copy_of_results = list(results)
        while len(copy_of_results) != 0:
            for result in results:
                try:
                    dataset_stat = result.get(timeout=1)
                    logging.debug("Result: %s (%.2f secs)" % (dataset_stat, time.time() - start_time))
                    merge_results(dataset_stat)
                    copy_of_results.remove(result)
                except multiprocessing.TimeoutError:
                    pass
            results = copy_of_results
        self._pool.close()
        self._pool.join()
        return DATASET_STAT


def process(lines):
    dataset_stat = DatasetStat()
    for line in lines:
        tokens = parse(line)
        review_stat = ReviewStat()
        for token in tokens:
            review_stat.add(token)
        dataset_stat.add([review_stat])
    return dataset_stat


def parse(line):
    """Parse the given line as json and extract the review text from the supplied lines"""
    data = json.loads(line)
    cleaned_review_text = cleanse(data.get(constants.TEXT, ""))
    return tokenize(cleaned_review_text)


def _remove_punctuation(word):
    return ''.join(ch for ch in word if ch not in constants.PUNCTUATIONS)


def cleanse(data):
    processed_data = _remove_punctuation(data)
    processed_data = processed_data.lower()
    return processed_data


def tokenize(line):
    return line.split()


def merge_results(dataset_stat):
    logging.debug('Merging result %s' % (str(dataset_stat)))
    DATASET_STAT.merge(dataset_stat)
