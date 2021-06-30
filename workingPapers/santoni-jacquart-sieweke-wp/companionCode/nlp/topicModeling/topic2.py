# %% libraries
import numpy as np
import pyLDAvis

# %% function to create itneractive pyLDAvis viz
def lda_viz(model_, topic_2_term, topic_2_doc):
    # data
    doc_lengths = np.array([len(doc.words) for doc in model_.docs])
    vocab = list(model_.used_vocabs)
    term_frequency = model_.used_vocab_freq
    # create pyLDAvis object 
    prepared_data = pyLDAvis.prepare(
        topic_term_dists=topic_2_term, 
        doc_topic_dists=topic_2_doc, 
        doc_lengths, 
        vocab, 
        term_frequency,
        start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
        sort_topics=False # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
    )
    pyLDAvis.save_html(prepared_data, 'ldavis.html')