#!/bin/bash

EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
  # when run without params, print help and exit
  echo "Run as: run_all.sh /data/shootout/title_tokens.txt.gz /data/embeddings/questions-words.txt ./results"
  exit $E_BADARGS
fi

input_corpus=$1
questions_file=$2
datadir=$3
shift 3

time python ./run_word2vec.py /data/shootout/title_tokens.txt.gz /data/embeddings/questions-words.txt ./results &> ./results/word2vec.log
time python ./run_glove.py /data/shootout/title_tokens.txt.gz /data/embeddings/questions-words.txt ./results &> ./results/glove.log
time python ./run_ppmi.py /data/shootout/title_tokens.txt.gz /data/embeddings/questions-words.txt ./results &> ./results/ppmi.log
time python ./run_svd.py /data/shootout/title_tokens.txt.gz /data/embeddings/questions-words.txt ./results &> ./results/svd.log
