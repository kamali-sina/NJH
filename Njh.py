import os
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import exp, log

MAKEHASHWITH = '$'

class Njh:
    def __init__(self, corpus_path, tuple_list=None, alpha=1, window_size=[1,0]):
        """
        Args:
            copus_path -> string, the path to the corpus
            tuple_list -> list of tuples, the pairs to include in the dict. if None includes all.
            alpha -> float, between 0 and 1
            window_size -> (i,j), shows the window of the words to look at when getting context.
        """
        self.alpha = alpha
        self.tuple_list = tuple_list
        self.window_size = window_size
        if (not os.path.exists(corpus_path)):
            print('path not found, exiting...')
            exit()
        corpus_file = open(corpus_path)
        self.corpus = corpus_file.read().split()
        #TODO: make to dict if needed
        self.uniques, self.uniques_count = list(np.unique(self.corpus, return_counts = True))
        print('doing preprocessing on corpus...')
        self._make_context_holder()
        print('preprocessing is done.')
        self._fill_ws_dict()

    def _make_context_holder(self):
        """makes self.context_holder that is a pandas df that holds the windows specified."""
        window = sum(self.window_size) + 1
        self.context_holder = np.array([self.corpus[0 : (1 - window)]])
        for i in range(1,window):
            if (i == window - 1):
                self.context_holder = np.append(self.context_holder, [self.corpus[i : ]], axis=0)
                continue
            self.context_holder = np.append(self.context_holder, [self.corpus[i : (i + 1 - window)]], axis=0)
        self.context_holder = np.transpose(self.context_holder)
        self.context_holder = pd.DataFrame(self.context_holder)
        self._make_groupby()
        self._make_unique_contexts_dict()

    def _make_unique_contexts_dict(self):
        center = self.window_size[0]
        cols = list(self.context_holder.columns)
        cols.pop(center)
        context_ref = self.context_holder[cols]
        # records = context_ref.to_records(index=False)
        # all_contexts = np.array(records)
        #TODO: choose which method is better, 2 list or dict
        # self.unique_contexts, self.unique_contexts_count = (np.unique(all_contexts, return_counts = True))
        self.unique_contexts_dict = self._make_dict_of_df(context_ref)
        self.context_with_words_counts = self._make_dict_of_df(self.context_holder)
        self._calculate_c()

    def _make_groupby(self):
        self.groupby_dict = {}
        cen = self.window_size[0]
        cols = list(self.context_holder.columns)
        center = cols.pop(cen)
        print('making groupby dict')
        for word,subdf in tqdm(self.context_holder.groupby(cen)):
            context_ref = subdf[cols]
            self.groupby_dict[word] = set(context_ref.agg(MAKEHASHWITH.join, axis=1))

    def _make_dict_of_df(self, df):
        """
        inputs:
            df is a dataframe of some words in each column
        output:
            dict is a dictionary, with keys of df rows and values of their counts
        """
        records = df.to_records(index=False)
        all_contexts = np.array(records)
        unique, unique_count = (np.unique(all_contexts, return_counts = True))
        output = list(map(MAKEHASHWITH.join, unique))
        #TODO: find faster hash function maybe? current takes 1.3 seconds
        return dict(zip(output, unique_count))

    def _calculate_c(self):
        self.c = 0
        number_of_contexts = len(self.context_holder)
        for count in self.unique_contexts_dict.values():
            self.c += (count/number_of_contexts)**self.alpha

    def _get_count_of_word(self, word):
        try:
            return self.uniques_count[self.uniques.index(word)]
        except:
            return 0

    def _fill_ws_dict(self):
        self.ws_dict = {}
        if (self.tuple_list):
            for x,y in tqdm(self.tuple_list):
                word_dict = self.ws_dict.get(x, {})
                word_dict[y] = self._calculate_similarity(x, y, self._get_count_of_word(x), self._get_count_of_word[y])
                self.ws_dict[x] = word_dict
        else:
            for word_index in tqdm(range(len(self.uniques))):
                word = self.uniques[word_index]
                word_dict = {}
                for second_word_index in range(len(self.uniques)):
                    second_word = self.uniques[second_word_index]
                    word_dict[second_word] = self._calculate_similarity(word, second_word, self.uniques_count[word_index], self.uniques_count[second_word_index])
                self.ws_dict[word] = word_dict

    def _get_N(self, w1, context):
        """Calculates N(w1|context)"""
        center = self.window_size[0]
        needed_row = context.split(MAKEHASHWITH)
        needed_row.insert(center, w1)
        N = self.context_with_words_counts.get(MAKEHASHWITH.join(needed_row), 0)
        return N

    def _calculate_conditional_P(self, w1, context, context_count):
        """Calculates P(w1|context)"""
        N_w1_context = self._get_N(w1, context)
        N_context = context_count
        return (N_w1_context / N_context)

    def _calculate_similarity(self, w1, w2, n_w1, n_w2):
        """Calculates S(w1,w2)"""
        p1 = n_w1 / len(self.corpus)
        p2 = n_w2 / len(self.corpus)
        number_of_contexts = len(self.context_holder)
        similarily = float(0)
        intersect = self.groupby_dict[w1].intersection(self.groupby_dict[w2] )
        for context in intersect:
            context_count = self.unique_contexts_dict[context]
            p_context = context_count / number_of_contexts
            cond_p1 = self._calculate_conditional_P(w1, context, context_count)
            cond_p2 = self._calculate_conditional_P(w2, context, context_count)
            similarily += ((cond_p1) * (cond_p2)) / ((p_context**(2-self.alpha)) * self.c * p1 * p2)
        return similarily