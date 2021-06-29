# !/usr/env/bin pyhon3
# -*- encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    load_data.py |  script to load sample studies' full papers
------------------------------------------------------------------------------

Author   :

Synopsis :

To do    :

"""

# %% libraries
import os
import glob
import pandas as pd

# %% load population of abstracts
def get_papers(path_):
    # screen for files
    os.chdir(path_)
    in_files = glob.glob(os.path.join(".", "*.csv"))
    # load, concatenate, and return data
    return pd.concat((pd.read_csv(f) for f in in_files))


# %% load sample studies' full papers
# TODO
