import os
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm

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

    def _fill_ws_dict(self):
        self.ws_dict = {}
        if (self.tuple_list):
            #TODO: fill here
            using_tuples = True
        else:
            for word_index in range(len(self.uniques)):
                word = self.uniques[word_index]
                word_dict = {}
                for second_word_index in tqdm(range(len(self.uniques))):
                    second_word = self.uniques[second_word_index]
                    word_dict[second_word] = self._calculate_similarity(word_index, second_word_index)
                self.ws_dict[word] = word_dict

    def _calculate_N(self, w1, context_index):
        """Calculates N(w1|context)"""
        center = self.window_size[0]
        needed_row = list(self.unique_contexts[context_index])
        needed_row.insert(center, w1)
        N = (self.context_holder == needed_row).all(1).sum()
        return N

    def _calculate_conditional_P(self, w1, context_index):
        #TODO: code here
        """Calculates P(w1|context)"""
        N_w1_context = self._calculate_N(w1, context_index)
        N_context = self.unique_contexts_count[context_index]
        return (N_w1_context / N_context)

    def _calculate_similarity(self, w1_index, w2_index):
        #TODO: code here
        """Calculates S(w1,w2)"""
        self._calculate_conditional_P(self.uniques[w1_index], 800)
        

x = time()
Njh('./corpus.txt', window_size=[1,1])
print(f'{time() - x} seconds')