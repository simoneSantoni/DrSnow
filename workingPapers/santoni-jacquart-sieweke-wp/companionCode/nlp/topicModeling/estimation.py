# %% import libraries
import sys
import spacy
import tomotopy as tp
import pandas as pd

# %% load data
df = pd.read_csv("~/Downloads/academy_of_management_journal.csv")

# %% rendering of text
nlp = spacy.load("en_core_web_sm")
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
lda_fit = tp.LDAModel(corpus=corpus, k=4, rm_top=1)
# train model
for i in range(0, 100, 10):
    lda_fit.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, lda_fit.ll_per_word))

# %% get coherence score
for i in range(4):
        print(lda_fit.get_topic_words(topic_id=i, top_n=10))

# %%
for k in range(lda_fit.k):
    print('Top 10 words of topic #{}'.format(k))
    print(lda_fit.get_topic_words(k, top_n=10))
# %%
