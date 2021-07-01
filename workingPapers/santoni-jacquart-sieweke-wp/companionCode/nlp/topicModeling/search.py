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
        for i in range(0, 100, 10):
            lda_fit.train(10)
            # print(
            #    """
            #      ==================================
            #      Retained number of topics: {}
            #      ----------------------------------
            #      Iteration:                 {:0.2f}
            #
            #      Log-likelihood:            {:0.2f}
            #      ----------------------------------
            #
            #      """.format(
            #        k, i, lda_fit.ll_per_word
            #    )
            # )
        # get coherences scores
        score = tp.coherence.Coherence(lda_fit, coherence="c_v")
        cs[k] = score.get_score()
    # return data
    return cs
