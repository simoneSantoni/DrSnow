# !/usr/env/bin pyhon3
# -*- encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    estimation.py |  script to topic modeling the population of abstracts
------------------------------------------------------------------------------

Author   : simone.santoni.1@city.ac.uk

Synopsis : this script estimates a topic model on the population of abstracts
           published in a set of target journals

Status   : on going

"""

# %% import libraries
import tomotopy as tp

# %% function to estimate and inspect a topic modeling
def tm_estimation(corpus_, k_):
    # estimatation
    tp.ParallelScheme(0)
    model = tp.LDAModel(min_df=5, rm_top=20, k=k_, seed=000, corpus=corpus_)
    model.train(0)
    # log
    print(
        "Num docs:{}, Num Vocabs:{}, Total Words:{}".format(
            len(model.docs), len(model.used_vocabs), model.num_words
        )
    )
    print("Removed Top words: ", *model.removed_top_words)
    # model training
    for i in range(0, 1000, 20):
        print("Iteration: {:04}, LL per word: {:.4}".format(i, model.ll_per_word))
        model.train(20)
    print("Iteration: {:04}, LL per word: {:.4}".format(1000, model.ll_per_word))
    # summary
    model.summary()
    # return
    return model
