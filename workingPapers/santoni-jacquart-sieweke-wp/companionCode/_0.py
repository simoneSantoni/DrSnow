#!/usr/bin/env python3
# coding: utf-8
"""
-------------------------------------------------------------------------------
    charts.py    |     definitions to reproduce the paper's scharts
-------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis :  

Notes    :
    -
"""

# %% libraries
import bibtexparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% load data
# all retrieved articles
in_file = '../review/'
with open(in_file) as bibtex_file:
    retrieved = bibtexparser.load(bibtex_file)

# sample
in_file = os.path.join(srv, path, folder, 'sample.bib')
with open(in_file) as bibtex_file:
    sample = bibtexparser.load(bibtex_file)

keys = list(sample.entries_dict.keys())


#  arrange data
# ---------------

# empty data frame
df = pd.DataFrame()

# iterate over entries to get data from dictionary
for i, key in enumerate(keys):
    to_append = pd.DataFrame.from_dict(sample.entries_dict[keys[i]],
                                       orient='index')
    df = pd.concat([df, to_append.T])

# columns of the data frame
cols = ['booktitle', 'keywords', 'month', 'publisher', 'url', 'file']
df.drop(cols, axis=1, inplace=True)

# year as number
df.loc[:, 'year'] = df['year'].astype(int)


#  - 0 - distribution of articles over time
# -----------------------------------------

# create data series

# --+ group the data
df.loc[:, 'count'] = 1
gr = df.groupby('year', as_index=False)
counts = gr['count'].agg(np.sum)

# --+ Get The Timespan
lb, ub = np.min(counts['year']), np.max(counts['year'])
timespan = pd.DataFrame({'year': np.append(np.arange(lb, ub, 1, dtype=int),
                                           2019)})

# --+ merge data
counts = pd.merge(counts, timespan, on='year', how='right')
counts.loc[counts['count'].isnull(), 'count'] = 0

# --+ x and y values to plot
counts.set_index('year', inplace=True)
counts.sort_index(inplace=True)
pos = np.arange(0, len(counts), 1)
y = counts['count'].values

# figure framewor
fig = plt.figure(figsize=(5, 3), frameon=True)
ax = fig.add_subplot(1, 1, 1)

# plot data
ax.bar(pos, y, color='grey', edgecolor='white')

# axis properties
ticks = np.append(np.arange(0, len(pos), 2), pos[-1])
tick_labels = ['%s' % i for i in np.arange(lb, ub, 2)]
tick_labels.append(r'2019$^{*}$')
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels, rotation='vertical')
ax.set_yticks([2, 4, 6, 8])
#ax.set_xlabel('Publication year')
#ax.set_ylabel('Counts of studies')
#ax.set_ylabel('Counts of articles')

# hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# grid
ax.grid(True, axis='y', ls=':', color='white')

# save plot to file
srv, path = srv, path
folder = 'ss_1/exhibits'
out_file = '_0.pdf'
plt.savefig(os.path.join(srv, path, folder, out_file),
            transparent=True,
            bbox_inches='tight',
            pad_inches=0,
            dpi=600)


# - 1 - distribution of articles over time and role of the exogenous variation
# ----------------------------------------------------------------------------

plt.rcParams["font.family"] = "Times New Roman"

# create figure frameworkk
fig = plt.figure(figsize=(3, 3))

# plot
ax = fig.add_subplot(1, 1, 1)

options = ['Methodological', 'Substantive', 'Both']
pos = [0, 1, 2]

# panel A: plot dat
ax.bar(0, 63, color='silver', width=0.6)
ax.bar(1, 7, color='silver', width=0.6)
ax.bar(2, 30, color='silver', width=0.6)

# axis
ax.set_xticks(pos)
ax.set_xticklabels(options)
ax.set_ylabel(' of studies')
ax.grid(axis='y', ls='--')

# hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# save plot to file
srv, path = srv, path
folder = 'ss_1/exhibits'
out_file = '_2.png'
plt.savefig(os.path.join(srv, path, folder, out_file),
            transparent=True,
            bbox_inches='tight',
            pad_inches=0,
            dpi=600)

# show plot
plt.show()



# -----------------------------------------------------------------------------
#    On going
# -----------------------------------------------------------------------------

ticks
