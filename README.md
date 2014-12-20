Evaluation of word embeddings
=============================

Code for the blog post evaluating word2vec, GloVe, SPPMI and SPPMI-SVD methods:

[Making sense of word2vec](http://radimrehurek.com/2014/12/making-sense-of-word2vec/).

Run `run_all.sh` to run all experiments. Logs with results will be stored in the data directory.

To replicate my results from the blog article, download and preprocess Wikipedia using [this code](https://github.com/piskvorky/sim-shootout).
You can use your own corpus though (the corpus path is a parameter to `run_all.sh`).
