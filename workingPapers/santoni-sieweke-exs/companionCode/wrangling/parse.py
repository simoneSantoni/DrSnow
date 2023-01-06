#!/usr/bin/env python3
# coding: utf-8
"""
-------------------------------------------------------------------------------
    parsers.py   |    script to parse .bib and .md files
-------------------------------------------------------------------------------
Author   : Simone Santoni, simone.santoni.1@city.ac.uk
Synopsis : Function to parse: i) bibliographics entries in BibTeX format;
           ii) markdown files; iii) mulitple values included in individual
           columns of a Pandas DF
Status   : None
"""

# %% libraries
import bibtexparser
import pandas as pd
from matplotlib import rc

# %% parser for BibTeX data


def df_from_bib(in_file):
    """Function that parses BibTeX data and arranges them in a Pandas DF
    Args:
        in_file (string or buffer): The BibTex to parse
    Returns:
        Pandas DF: data frame containing selected attributes included
                   in BibTeX entries
    """
    # load data
    with open(in_file) as bibtex_file:
        data = bibtexparser.load(bibtex_file)
    # get keys
    keys = list(data.entries_dict.keys())
    # arrange data within a Pandas data frame
    df = pd.DataFrame()
    # iterate over entries to get data from dictionary
    for i, key in enumerate(keys):
        to_append = pd.DataFrame.from_dict(data.entries_dict[keys[i]],
                                           orient="index")
        df = pd.concat([df, to_append.T])
    # columns to filter out
    cols = ["ID", "journal", "year", "author", "title"]
    df = df.loc[:, cols]
    # rename column
    df.rename({"ID": "key"}, axis=1, inplace=True)
    # annotate forthcoming articles
    df.loc[df["year"].isnull(), "year"] = 2022
    # year as number
    df.loc[:, "year"] = df["year"].astype(int)
    # return object
    return df
