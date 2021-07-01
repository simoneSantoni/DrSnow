#!/usr/bin/env python3
# coding: utf-8
"""
-------------------------------------------------------------------------------
    parsers.py   |    script to parse .bib and .md files
-------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : Function to parse: i) bibliographics entries in BibTeX format;
           ii) markdown files 

To do    : None

"""

# %% libraries
import bibtexparser
import numpy as np
import pandas as pd

# %% parser for BibTeX data


def df_from_bib(in_f):
    # load data
    with open(in_f) as bibtex_file:
        data = bibtexparser.load(bibtex_file)
    # get keys
    keys = list(data.entries_dict.keys())
    # arrange data within a Pandas data frame
    df = pd.DataFrame()
    # iterate over entries to get data from dictionary
    for i, key in enumerate(keys):
        to_append = pd.DataFrame.from_dict(data.entries_dict[keys[i]], orient="index")
        df = pd.concat([df, to_append.T])
    # columns to filter in
    cols = ["ID", "journal", "year", "author", "title","abstract"]
    if "abstract" in df.columns:
        df = df.loc[:, cols]
    else:
        df = df.loc[:, cols[:-1]]
    # rename column
    df.rename({"ID": "key"}, axis=1, inplace=True)
    # year as number
    df.loc[:, "year"] = df["year"].astype(int)
    # annotate forthcoming articles
    df.loc[df["year"] == 9999, "year"] = np.max(df.loc[df["year"] < 9999, "year"]) + 1
    # return df
    return df
