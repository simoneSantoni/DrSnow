#!/usr/bin/env python3
# coding: utf-8
"""
-------------------------------------------------------------------------------
    clean_bib_items.py    |     clean BibLaTeX entries 
-------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : Sometimes, 'strange' carachters included in the abstract or other 
           sections prevent BibLaTeX from compiling the bibliography. This script
           gets rid of them.

To do    : TODO: create classes to read and manipulate bilio entries

"""

# %% read data
refs = []
with open('sample_of_studies.bib', 'r') as pipe:
    for line in pipe.readlines():
        if 'file =' in line:
            print('skip this!')
        elif 'mendeley-group' in line:
            print('skip this!')
        elif 'month = ' in line:
            print('skip this!')
        elif 'abstract' in line:
            print('skip this!')
        elif 'issn = ' in line:
            print('skip this!')
        elif 'doi = ' in line:
            print('skip this!')
        elif 'url = ' in line:
            print('skip this!')
        else:
            refs.append(line)

# %% write data
with open('sample_of_studies.bib', 'w') as pipe:
    for line in refs:
        pipe.write('{}'.format(line))

# %%
