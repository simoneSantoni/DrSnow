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
    return tp.LDAModel(corpus=corpus_, k=k_, rm_top=20, seed=000)
