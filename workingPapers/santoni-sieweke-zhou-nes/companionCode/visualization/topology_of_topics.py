# %% libraries
import numpy as np
import pyLDAvis

# %% function to create itneractive pyLDAvis viz
def lda_viz(topic_2_term, topic_2_doc, doc_lengths, vocab_, term_frequency):
    # create pyLDAvis object
    prepared_data = pyLDAvis.prepare(
        topic_term_dists=topic_2_term,
        doc_topic_dists=topic_2_doc,
        doc_lengths=doc_lengths,
        vocab=vocab_,
        term_frequency=term_frequency,
        start_index=0,
        sort_topics=False,
    )
    pyLDAvis.save_html(prepared_data, "ldaviz.html")
