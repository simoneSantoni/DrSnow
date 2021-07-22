#!/usr/bin/env python3
# coding: utf-8
"""
-------------------------------------------------------------------------------
    clean_bib_items.py    |     clean BibLaTeX entries 
-------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : The script filters out BibTeX fields that are seldom used and
           normalize the values included in the retained fields

To do    : None

"""

# %% clean & write pipeline


def filter_and_norm(in_f, out_f, common_words):
    """Filter and normalize BibTeX fields

    Args:
        in_f (str): name of the file to clean
        out_f (str): name of the file containing clean data
        common_words (iterable): set of common words not to capitalize
    """
    # empty container
    refs = []
    # read data
    with open(in_f, "r") as pipe:
        for line in pipe.readlines():
            if "file =" in line:
                pass
            elif "mendeley-group" in line:
                pass
            elif "month " in line:
                pass
            elif "abstract " in line:
                pass
            elif "issn = " in line:
                pass
            elif "isbn = " in line:
                pass
            elif "doi = " in line:
                pass
            elif "url = " in line:
                pass
            elif "keywords = " in line:
                pass
            elif "publisher = " in line:
                pass
            else:
                refs.append(line)
    # apply correct BiBLaTeX item type
    for i, item in enumerate(refs):
        if "@misc" in item:
            refs[i] = item.replace("@misc", "@article")
        else:
            pass
    # apply correct BiBLaTeX item type
    for i, item in enumerate(refs):
        if "booktitle" in item:
            refs[i] = item.replace("booktitle", "journal")
        else:
            pass
    # item labels are all lower case
    for i, item in enumerate(refs):
        if "@article" in item:
            refs[i] = item.lower()
    # item titles have 'title' case
    for i, item in enumerate(refs):
        if "title = " in item:
            refs[i] = item.title()
            for word in common_words:
                refs[i] = refs[i].replace(word, word.lower())
            for word in common_words:
                old = ":{}".format(word.lower())
                new = ":{}".format(word)
                refs[i] = refs[i].replace(old, new)
        else:
            pass
    # &
    for i, item in enumerate(refs):
        refs[i] = refs[i].replace("&", "\&")
    # write file
    with open(out_f, "w") as pipe:
        for item in refs:
            pipe.write(item)
