# %% libraries
import numpy as np
import tomotopy as tp

# %% function to create itneractive pyLDAvis viz
def get_vectors(model_):
    topic_term_dists = np.stack(
        [model_.get_topic_word_dist(k) for k in range(model_.k)]
    )
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in model_.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in model_.docs])
    vocab = list(model_.used_vocabs)
    term_frequency = model_.used_vocab_freq
    return topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
