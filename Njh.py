import os
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import exp, log

class Njh:
    def __init__(self, corpus_path, tuple_list=None, alpha=1, window_size=[1,0]):
        """
        Args:
            copus_path -> string, the path to the corpus
            tuple_list -> list of tuples, the pairs to include in the dict. if None includes all.
            alpha -> int
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
        self._make_unique_contexts()

    def _make_unique_contexts(self):
        center = self.window_size[0]
        cols = list(self.context_holder.columns)
        cols.pop(center)
        context_ref = self.context_holder[cols]
        records = context_ref.to_records(index=False)
        all_contexts = np.array(records)
        self.unique_contexts, self.unique_contexts_count = (np.unique(all_contexts, return_counts = True))
        self.exp_sum_value = 0
        number_of_contexts = len(self.context_holder)
        for i in range(self.unique_contexts):
            self.exp_sum_value += self._calculate_exp_log(i, number_of_contexts)

    def _calculate_exp_log(self, context_index, number_of_contexts):
        p_context = self.unique_contexts_count[context_index] / number_of_contexts
        return exp(1) ** (log(p_context / self.alpha))

    def _get_count_of_word(self, word):
        try:
            return self.uniques_count[self.uniques.index(word)]
        except:
            return 0

    def _fill_ws_dict(self):
        self.ws_dict = {}
        if (self.tuple_list):
            for tup in tqdm(self.tuple_list):
                word_dict = self.ws_dict.get(tup[0], {})
                word_dict[tup[1]] = self._calculate_similarity(tup[0], tup[1], self._get_count_of_word(tup[0]), self._get_count_of_word[tup[1]])
                self.ws_dict[tup[0]] = word_dict
        else:
            for word_index in range(len(self.uniques)):
                word = self.uniques[word_index]
                word_dict = {}
                for second_word_index in tqdm(range(len(self.uniques))):
                    second_word = self.uniques[second_word_index]
                    word_dict[second_word] = self._calculate_similarity(word, second_word, self.uniques_count[word_index], self.uniques_count[second_word_index])
                self.ws_dict[word] = word_dict

    def _calculate_N(self, w1, context_index):
        """Calculates N(w1|context)"""
        center = self.window_size[0]
        needed_row = list(self.unique_contexts[context_index])
        needed_row.insert(center, w1)
        N = (self.context_holder == needed_row).all(1).sum()
        return N

    def _calculate_conditional_P(self, w1, context_index):
        """Calculates P(w1|context)"""
        N_w1_context = self._calculate_N(w1, context_index)
        N_context = self.unique_contexts_count[context_index]
        return (N_w1_context / N_context)
    
    def _calculate_similarity(self, w1, w2, n_w1, n_w2):
        """Calculates S(w1,w2)"""
        p1 = n_w1 / len(self.corpus)
        p2 = n_w2 / len(self.corpus)
        number_of_contexts = len(self.context_holder)
        similarily = float(0)
        for i in range(len(self.unique_contexts)):
            cond_p1 = self._calculate_conditional_P(w1, i)
            cond_p2 = self._calculate_conditional_P(w2, i)
            val1 = self._calculate_exp_log(i, number_of_contexts)
            similarily += (cond_p1/p1) * (cond_p2/p2) * (val1/self.exp_sum_value)
        return similarily

x = time()
Njh('./corpus.txt', window_size=[1,1])
print(f'{time() - x} seconds')