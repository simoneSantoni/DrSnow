# !/usr/env/bin pyhon3
# -*- encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    search.py |  script to topic modeling the population of abstracts
------------------------------------------------------------------------------

Author   : simone.santoni.1@city.ac.uk

Synopsis : this script estimates a topic model on the population of abstracts
           published in a set of target journals

Status   : on going

"""

# %% import libraries
import numpy as np
import tomotopy as tp

# %% screen for plausible number of topics
def search_k(min_, max_, delta_, corpus_):
    """[summary]

    Args:
        min_ ([type]): [description]
        max_ ([type]): [description]
        delta_ ([type]): [description]
        corpus_ ([type]): [description]

    Returns:
        [type]: [description]
    """
    # take advantage of parallelization
    tp.ParallelScheme(0)
    # empty container to store coherence scores
    cs = {}
    # fit LDA model for different k
    ks = np.arange(min_, max_, delta_)
    for k in ks:
        lda_fit = tp.LDAModel(corpus=corpus_, k=k, rm_top=20, seed=000)
        # train model
        lda_fit.train(0)
        for i in range(0, 1000, 20):
            print("Iteration: {:04}, LL per word: {:.4}".format(i, lda_fit.ll_per_word))
            lda_fit.train(20)
        print("Iteration: {:04}, LL per word: {:.4}".format(1000, lda_fit.ll_per_word))
        # get coherences scores
        score = tp.coherence.Coherence(lda_fit, coherence="c_v")
        cs[k] = score.get_score()
    # return data
    return cs
