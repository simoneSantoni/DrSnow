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
import spacy
import tomotopy as tp
import pandas as pd

# %% load data
sample = "/home/simone/.data/academy_of_management_journal.csv"
df = pd.read_csv(sample)

# %% rendering of text
nlp = spacy.load("en_core_web_lg")
tkn_docs = []
for doc in df.loc[:, "Abstract"].to_list():
    tkn_docs.append(
        [
            tkn.lemma_.lower()
            for tkn in nlp(doc)
            if not tkn.is_stop and tkn.is_alpha and not tkn.is_digit
        ]
    )

# %% create Tomotopy's corpus class and populate it
corpus = tp.utils.Corpus()
for doc in tkn_docs:
    corpus.add_doc(words=doc)

# %% screen for plausible number of topics
# number of topics to retain
# fit LDA model
lda_fit = tp.LDAModel(corpus=corpus, k=10, rm_top=5, seed=000)
# train model
for i in range(0, 100, 10):
    lda_fit.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, lda_fit.ll_per_word))

# %% preview topics
for i in range(lda_fit.k):
    print('Top 10 words of topic #{}'.format(i))
    print(lda_fit.get_topic_words(i, top_n=10))
    
# %% get coherences scores
for preset in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
    coh = tp.coherence.Coherence(lda_fit, coherence=preset)
    average_coherence = coh.get_score()
    coherence_per_topic = [coh.get_score(topic_id=k) for k in range(lda_fit.k)]
    print('==== Coherence : {} ===='.format(preset))
    print('Average:', average_coherence, '\nPer Topic:', coherence_per_topic)
    print()

# %%
