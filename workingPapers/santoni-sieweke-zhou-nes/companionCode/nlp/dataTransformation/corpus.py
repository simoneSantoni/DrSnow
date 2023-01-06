# %% import libraries
from tomotopy.utils import Corpus

# %% create Tomotopy's 'corpus' class and populate it
def tp_corpus(tkn_docs):
    # initialize object of class Tomotopy corpus
    corpus = Corpus()
    # populate the corpus
    for doc in tkn_docs:
        corpus.add_doc(words=doc)
    # return corpus
    return corpus
