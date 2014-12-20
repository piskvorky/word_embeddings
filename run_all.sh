#!/bin/bash

EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
  # when run without params, print help and exit
  echo "Run as: ./run_all.sh /data/shootout/title_tokens.txt.gz /data/embeddings/questions-words.txt ./results_dim300_vocab30k"
  exit $E_BADARGS
fi

input_corpus=$1
questions=$2
outdir=$3

mkdir -p $outdir 2> /dev/null

time python ./run_word2vec.py $input_corpus $questions $outdir &> $outdir/word2vec.log
time python ./run_glove.py $input_corpus $questions $outdir &> $outdir/glove.log
time python ./run_ppmi.py $input_corpus $questions $outdir &> $outdir/ppmi.log
time python ./run_svd.py $input_corpus $questions $outdir &> $outdir/svd.log
