# !/usr/env/bin python3
# -*- encoding utf-8 -*-

"""
----------------------------------------------------
                 DESCRIPTIVE STATS

      ~ Leadership studies using NE designs ~
----------------------------------------------------

Author: Simone Santoni, simone.santoni.1@city.ac.uk

Dates: created Fri Apr 30 2019 08:03:12 BST;
       last change Sat 14 Sep 2019 22:19:33 BST

Notes: --

"""


# %% libraries
# ------------

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %% figure preferences
# ---------------------

plt.style.use('seaborn-bright')


# %% read data
# ------------

CWD = os.getcwd()
FDR = 'data'
FILE0 = 'ne_instances.csv'
FILE1 = 'ne_in_ldr.csv'

# ne instances
DF0 = pd.read_csv(os.path.join(CWD, FDR, FILE0))

# ne studies in the field of leaderhip
DF1 = pd.read_csv(os.path.join(CWD, FDR, FILE1))

# merge
COLS = ['id', 'title', 'journal', 'year', 'abstract', 'busman', 'econ',
        'mult', 'psych', 'socsci']

DF = pd.merge(DF0[COLS], DF1, on='id', how='right')

del(DF0, DF1)

# gardner et al. 2010 categories
FILE2 = 'gardner_et_al_cat.csv'
DF2 = pd.read_csv(os.path.join(CWD, FDR, FILE2))


# %% slice data
# -------------

'''
Our review focuses on the 2000 - 2019 timespan
'''

DF = DF.loc[DF['year'] > 1999]


# %% Fig. 1: distribution of articles by form of NE and publication year
# ----------------------------------------------------------------------

'''
The chart has two panels. Panel A shows the distribution of
NE studies with respect to the form of NE. Panel B shows
the distribution with respect to form and publication year
'''

# create framework of the figure
FIG, [AX0, AX1] = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

# NE forms
FORMS = ['standard_ne', 'iv', 'rdd']

# TIMESPAN
T = 20

# panel A: data series
SNE = np.sum(DF[FORMS[0]])
IV = np.sum(DF[FORMS[1]])
RDD = np.sum(DF[FORMS[2]])
COUNTS = [SNE, IV, RDD]
FORM_LABELS = ['`Standard\' NE', 'IV', 'RD']
POS = [0, 1, 2]

# panel B: data series
DF.loc[:, 'count'] = 1
X = np.arange(2000, 2020, 1)
Y_SNE = dict(zip(X, np.repeat(0, len(X))))
Y_IV = dict(zip(X, np.repeat(0, len(X))))
Y_RDD = dict(zip(X, np.repeat(0, len(X))))
DICTS = [Y_SNE, Y_IV, Y_RDD]
for form, dictionary in zip(FORMS, DICTS):
    gr = DF.loc[DF[form] == 1].groupby(['year', form])
    series = gr['count'].agg(np.sum)
    for year in X:
        if year in series.keys():
            dictionary[year] = series[year].values[0]
        else:
            pass


BOTTOM_SNE = np.repeat(0, T)
BOTTOM_IV = list(Y_SNE.values())
BOTTOM_RDD = [i + j for i, j in zip(Y_SNE.values(), Y_IV.values())]

# panel A: plot data
AX0.bar(0, SNE, color='black', width=0.6)
AX0.bar(1, IV, color='grey', width=0.6)
AX0.bar(2, RDD, color='silver', width=0.6)
AX0.set_xticks(POS)
AX0.set_xticklabels(FORM_LABELS)
AX0.set_yticks(np.arange(0, np.max([SNE, IV, RDD]), 10))
AX0.set_ylabel('Counts of studies')
AX0.grid(axis='y', ls='--')
AX0.text(2, 38, 'A', fontweight='bold', fontsize=14, color='k')
AX0.text(0, -8, 'Timespan: Jan. 2000 - Mar. 2019')

# panel B: plot data
AX1.bar(X, Y_SNE.values(),
        bottom=BOTTOM_SNE,
        color='black', label=FORM_LABELS[0])
AX1.bar(X, Y_IV.values(),
        bottom=BOTTOM_IV,
        color='grey', label=FORM_LABELS[1])
AX1.bar(X, Y_RDD.values(),
        bottom=BOTTOM_RDD,
        color='silver', label=FORM_LABELS[2])
TICKS = np.arange(2000, 2020, 2)
TICK_LABELS = ['%s' % i for i in TICKS]
TICK_LABELS.append('2019$^{*}$')
TICKS = np.append(TICKS, 2019)
AX1.set_xticks(TICKS)
AX1.set_xticklabels(TICK_LABELS, rotation='vertical')
AX1.set_xlabel('Publication year')
STACK = [i + j + w for i, j, w in zip(Y_SNE.values(),
                                      Y_IV.values(),
                                      Y_RDD.values())]
TICKS = np.arange(0, np.max(STACK) + 2, 2)
AX1.set_yticks(TICKS)
AX1.set_ylabel('Counts of studies')
#AX1.grid(True, ls='--')
AX1.legend(loc='upper center')
AX1.text(2001, 14.2, 'B', fontweight='bold', fontsize=14, color='k')

# save plot to file
CWD = CWD
os.mkdir('exhibits')
FDR = 'exhibits'
FILE = '_0.pdf'
plt.savefig(os.path.join(CWD, FDR, FILE),
            transparent=True,
            bbox_inches='tight',
            pad_inches=0)

# send plot to the screen
plt.show()


# %% Fig. A1: distribution of articles by year and subject
# --------------------------------------------------------

# panels
SUBJS = ['busman', 'econ', 'mult', 'psych', 'socsci']
SUBJS_LBLS = ['Business and Management', 'Economics',
              'Multidisciplinary', 'Psychology',
              'Social Sciences']
ALPHAS = [1, 0.6, 0.4, 0.25, 0.15]

# create framework
FIG = plt.figure(figsize=(6, 4))

# add plot
AX = FIG.add_subplot(1, 1, 1)

# timespan
T = T

# time periods
X = np.arange(2000, 2020, 1)

# year as positions
POS = pd.DataFrame({'year': X, 'pos': np.arange(0, T, 1)})
DF = pd.merge(DF, POS, on='year', how='inner')

# plot data series
BOTTOM = np.repeat(0, T)
for subj, label, alpha in zip(SUBJS, SUBJS_LBLS, ALPHAS):
    gr = DF.loc[DF[subj] == 1].groupby('pos', as_index=False)
    df = pd.merge(POS, pd.DataFrame(gr['count'].agg(np.sum)),
                  on='pos', how='left')
    df.loc[df['count'].isnull(), 'count'] = 0
    # get positions and data points
    pos, y = df['pos'], df['count']
    # plot data
    AX.bar(pos, y, color='k', alpha=alpha, label=label, bottom=BOTTOM)
    # increase bottom
    for i in pos:
        for j in POS.pos:
            if i == j:
                to_add = df.loc[df['pos'] == i, 'count']
                BOTTOM[i] += to_add
            else:
                pass



# axis
AX.set_ylabel('Counts of study - subject occurrences')
TICKS = np.arange(0, T - 1, 2)
TICK_LABELS = ['%s' % i for i in np.arange(2000, 2020, 2)]
TICK_LABELS.append('2019$^{*}$')
TICKS = np.append(TICKS, 19)
AX.set_xticklabels(TICK_LABELS, rotation='vertical')
AX.set_xticks(TICKS)
AX.set_xlabel('Publication year')


# grid
#AX.grid(True, ls='--')

# legend
AX.legend(loc='best')

# save plot to file
CWD, FDR = CWD, FDR
FILE = 'A_0.pdf'
plt.savefig(os.path.join(CWD, FDR, FILE),
            transparent=True,
            bbox_inches='tight',
            pad_inches=0)

# send plot to the screen
plt.show()


# %% Tab. A1: distribution of the articles across journals
# --------------------------------------------------------

GR = DF.groupby('journal')
JRN = GR['count'].agg(np.sum)

CWD, FDR = CWD, FDR
FILE = 'A_1.tex'
JRN.to_latex(os.path.join(CWD, FDR, FILE))

