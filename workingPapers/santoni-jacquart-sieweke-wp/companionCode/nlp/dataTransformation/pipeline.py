# %% libraries
import spacy

# %% rendering of text
def spacy_pipeline(model_, raw_corpus):
    """[summary]

    Args:
        model_ ([type]): [description]
        raw_corpus ([type]): [description]
    """
    # load model
    nlp = spacy.load(model_)
    # empty container
    tkn_docs = []
    # iterate over documents
    for doc in raw_corpus:
        tkn_docs.append(
            [
                tkn.lemma_.lower()
                for tkn in nlp(doc)
                if not tkn.is_stop and tkn.is_alpha and not tkn.is_digit
            ]
        )
    # return tokenized docs
    return tkn_docs