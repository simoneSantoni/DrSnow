# %% libraries
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

# %% train a toy phrase model on our training corpus
def extract_ngrams(tkn_docs):
    """[summary]

    Args:
        tkn_docs ([type]): [description]
        common_terms ([type]): [description]
    """    
    bigrams = Phrases(
        tkn_docs,
        min_count=50,
        threshold=5,
        max_vocab_size=50000,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    trigrams = Phrases(
        bigrams[tkn_docs],
        min_count=50,
        threshold=5,
        max_vocab_size=50000,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    # return
    return [trigrams[bigrams[line]] for line in tkn_docs]
