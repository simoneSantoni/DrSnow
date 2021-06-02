# Sieweke & Santoni 2020 － README

- [Sieweke & Santoni 2020 － README](#sieweke--santoni-2020--readme)
  - [Contents ― Overview](#contents--overview)
    - [File Tree](#file-tree)
  - [Folders' content](#folders-content)
    - [`companionCode`](#companioncode)
      - [Python Script `_0.py`](#python-script-_0py)
      - [Jupyter Notebook `_1`](#jupyter-notebook-_1)
    - [`data`](#data)
    - [`transformedData`](#transformeddata)
  - [Project status](#project-status)

## Contents ― Overview

This folder include the code, data, and $\LaTeX$ project for the published
research paper  ["Natural experiments in leadership 
  research: An introduction, review, and guidelines"](https://www.sciencedirect.com/science/article/pii/S1048984318308476?casa_token=-3OC5QMgU6cAAAAA:qlBDw-17VkrxRsYA5HtXyiYOuQHNepujSU0x44abPMrBEer5gEmX5GcNrywvaJjwJFB14J5_BSs).

In order to ensure the full reproducibility of the natural language processing
analyses (see Appendix B), we include the transformed text corpora associated to
the topic model.

### File Tree

```
├── companionCode
│   ├── _0.py
│   ├── _1.html
│   ├── _1.ipynb
├── data
│   ├── gardner_et_al_cat.csv
│   ├── lq_instances.csv
│   ├── my_stopwords.pickle
│   ├── ne_in_ldr.csv
│   ├── ne_instances.csv
│   └── transformed_corpus.csv
├── manuscript
|   ├── *.pdf
|   ├── wp.*
├── README.md
└── transformedData
    ├── mds_positions.csv
    ├── transformed_corpus.csv
    └── transformed_newdocs.csv
```
## Folders' content

### `companionCode`

#### Python Script `_0.py`

This script creates the following exhibits:

+ Figure 1: Counts of Retrieved Studies across Forms of Natural Experiment and 
  Time
+ Figure A.1: Counts of Retrieved Studies - Disciplinary Subjects Occurrences 
  over Time
+ Table A.1: Counts of Retrieved Studies by Journal (Alphabetical Order)

Note running `_0.py` will create and populate the `exhibits` sub-folder.


#### Jupyter Notebook `_1`

This Jupyter notebook `_1` creates the below displayed list of exhibits:

+ Table 1: Term-Topic Matrix
+ Table 2: Standard Natural Experiment Designs―Substantive Focus
+ Table 3: Instrumental Variable Designs―Substantive Focus
+ Table 4: RD Designs―Substantive Focus
+ Figure 3: Standard Natural Experiments—Topic Characterization
+ Figure 5: Instrumental Variable Designs—Topic Characterization
+ Figure 7: Regression Discontinuity Designs—Topic Characterization


### `data` 

The `data` folder contains the following data tables:

+ `gardner_et_al_cat.csv`: unique identifiers and labels for the 29 leadership
  topics reported in Gardner, W. L., Lowe, K. B., Moss, T. W., Mahoney, K. T., &
  Cogliser, C. C. (2010). [Scholarly leadership of the study of leadership: A
  review of The Leadership Quarterly’s second decade](https://www.sciencedirect.com/science/article/pii/S1048984310001402?via%3Dihub),
  2000-2009. *Leadership Quarterly*, 21 (6), 922-958.
+ `lq_instances.csv`: meta-data about the 1,156 articles published in the Leadership 
  Quartyerly between January 2000 and March 2019.
+ `my_stopwords.pickle`: pickle file containing the stop-words to pass in the
  natural language processing pipeline ― as per spaCy [`nlp`](https://spacy.io/usage/processing-pipelines)
+ `ne_in_ldr.csv`: sample of leadership studies using a natural experiment 
  design along with key study attributes such as the reference topic (see
  `data/gardner_et_al.csv`) and the type of natural experiment form.
+ `ne_instances.csv`: meta-data about the articles retrieved with Scopus or
    Google Scholar/hand-curated search. Note `id` values equal or greater than
    10,000 are associated to articles identified with the Google
    Scholar/hand-curated search. 

### `transformedData`

The `transformedData` folder contains the following data tables:

+ `transformed_corpus.csv`: transformed corpus of text (abstracts) concerning 
  the 1,156 Leadership Quarterly articles (see `data/ne_in_ldr.csv`)
+ `transformed_newdocs.csv`: transformed corpus of text (abstracts) concerning 
  the 87studies that adopt a natural experiment design to address a leadership 
  topic (see `data/ne_in_ldr.csv`)
+ `mds_positions.csv`: outcome of the MDS model underlying Figure 3, Figure 5,
    and Figure 7.


## Project status

Closed - no updates expected.
