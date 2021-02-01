# CohPy
CohPy aims to generate a cohesion determination for given texts.
Therefore a multitude of scores are implemented and applied on most of the Gutenberg Corpus.
As well as some other well known books, which have a known "dificulty".

## Get Started:
- Clone github repository
- install all necessary dependencies (using conda)\
        conda env create -f environment.yml
- run the application pyhon main.py YOUR_TXT_FILE.txt
    - necessary files:
        - vector space model (here used SentiArt -> retrain for another model)
        - Word Conretness Scores (for your language)
        - for retraining, the corpora to train on and some evaluation data
