#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s INPUT_FILE QUESTIONS OUTPUT_DIR

Compare various word embedding techniques on the analogy task.

Example: python ./run_word2vec.py /data/shootout/title_tokens.txt.gz /data/embeddings/questions-words.txt ./results_dim300_vocab30k

"""


import os
import sys
import logging
import itertools
from collections import defaultdict

import numpy
import scipy.sparse

import gensim
from gensim import utils, matutils

import glove  # https://github.com/maciejkula/glove-python

# parameters controlling what is to be computed: how many dimensions, window size etc.
DIM = 600
DOC_LIMIT = None  # None for no limit
TOKEN_LIMIT = 30000
WORKERS = 8
WINDOW = 10
DYNAMIC_WINDOW = False
NEGATIVE = 10  # 0 for plain hierarchical softmax (no negative sampling)

logger = logging.getLogger("run_embed")

import pyximport; pyximport.install(setup_args={'include_dirs': numpy.get_include()})
from cooccur_matrix import get_cooccur


def most_similar(model, positive=[], negative=[], topn=10):
    """
    Find the top-N most similar words. Positive words contribute positively towards the
    similarity, negative words negatively.

    `model.word_vectors` must be a matrix of word embeddings (already L2-normalized),
    and its format must be either 2d numpy (dense) or scipy.sparse.csr.

    """
    if isinstance(positive, basestring) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) if isinstance(word, (basestring, numpy.ndarray)) else word
        for word in positive]
    negative = [
        (word, -1.0) if isinstance(word, (basestring, numpy.ndarray)) else word
        for word in negative]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, numpy.ndarray):
            mean.append(weight * word)
        elif word in model.word2id:
            word_index = model.word2id[word]
            mean.append(weight * model.word_vectors[word_index])
            all_words.add(word_index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    if scipy.sparse.issparse(model.word_vectors):
        mean = scipy.sparse.vstack(mean)
    else:
        mean = numpy.array(mean)
    mean = matutils.unitvec(mean.mean(axis=0)).astype(model.word_vectors.dtype)

    dists = model.word_vectors.dot(mean.T).flatten()
    if not topn:
        return dists
    best = numpy.argsort(dists)[::-1][:topn + len(all_words)]

    # ignore (don't return) words from the input
    result = [(model.id2word[sim], float(dists[sim])) for sim in best if sim not in all_words]

    return result[:topn]


def log_accuracy(section):
    correct, incorrect = section['correct'], section['incorrect']
    if correct + incorrect > 0:
        logger.info("%s: %.1f%% (%i/%i)" %
            (section['section'], 100.0 * correct / (correct + incorrect),
            correct, correct + incorrect))


def accuracy(model, questions, ok_words=None):
    """
    Compute accuracy of the word embeddings.

    `questions` is a filename where lines are 4-tuples of words, split into
    sections by ": SECTION NAME" lines.
    See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

    The accuracy is reported (=printed to log and returned as a list) for each
    section separately, plus there's one aggregate summary at the end.

    Only evaluate on words in `word2id` (such as 30k most common words), ignoring
    any test examples where any of the four words falls outside `word2id`.

    This method corresponds to the `compute-accuracy` script of the original C word2vec.

    """
    if ok_words is None:
        ok_words = model.word2id

    sections, section = [], None
    for line_no, line in enumerate(utils.smart_open(questions)):
        line = utils.to_unicode(line)
        if line.startswith(': '):
            # a new section starts => store the old section
            if section:
                sections.append(section)
                log_accuracy(section)
            section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
        else:
            if not section:
                raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
            try:
                a, b, c, expected = [word.lower() for word in line.split()]  # TODO assumes vocabulary preprocessing uses lowercase, too...
            except:
                logger.info("skipping invalid line #%i in %s" % (line_no, questions))
            if a not in ok_words or b not in ok_words or c not in ok_words or expected not in ok_words:
                logger.debug("skipping line #%i with OOV words: %s" % (line_no, line.strip()))
                continue

            ignore = set(model.word2id[v] for v in [a, b, c])  # indexes of words to ignore
            predicted = None

            # find the most likely prediction, ignoring OOV words and input words
            sims = most_similar(model, positive=[b, c], negative=[a], topn=False)
            for index in numpy.argsort(sims)[::-1]:
                if model.id2word[index] in ok_words and index not in ignore:
                    predicted = model.id2word[index]
                    if predicted != expected:
                        logger.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                    break

            section['correct' if predicted == expected else 'incorrect'] += 1
    if section:
        # store the last section, too
        sections.append(section)
        log_accuracy(section)

    total = {'section': 'total', 'correct': sum(s['correct'] for s in sections), 'incorrect': sum(s['incorrect'] for s in sections)}
    log_accuracy(total)
    sections.append(total)
    return sections


def raw2ppmi(cooccur, word2id, k_shift=1.0):
    """
    Convert raw counts from `get_coccur` into positive PMI values (as per Levy & Goldberg),
    in place.

    The result is an efficient stream of sparse word vectors (=no extra data copy).

    """
    logger.info("computing PPMI on co-occurence counts")

    # following lines a bit tedious, as we try to avoid making temporary copies of the (large) `cooccur` matrix
    marginal_word = cooccur.sum(axis=1)
    marginal_context = cooccur.sum(axis=0)
    cooccur /= marginal_word[:, None]  # #(w, c) / #w
    cooccur /= marginal_context  # #(w, c) / (#w * #c)
    cooccur *= marginal_word.sum()  # #(w, c) * D / (#w * #c)
    numpy.log(cooccur, out=cooccur)  # PMI = log(#(w, c) * D / (#w * #c))

    logger.info("shifting PMI scores by log(k) with k=%s" % (k_shift, ))
    cooccur -= numpy.log(k_shift)  # shifted PMI = log(#(w, c) * D / (#w * #c)) - log(k)

    logger.info("clipping PMI scores to be non-negative PPMI")
    cooccur.clip(0.0, out=cooccur)  # SPPMI = max(0, log(#(w, c) * D / (#w * #c)) - log(k))

    logger.info("normalizing PPMI word vectors to unit length")
    for i, vec in enumerate(cooccur):
        cooccur[i] = matutils.unitvec(vec)

    return matutils.Dense2Corpus(cooccur, documents_columns=False)


class PmiModel(object):
    def __init__(self, corpus):
        # serialize PPMI vectors into an explicit sparse CSR matrix, in RAM, so we can do
        # dot products more easily
        self.word_vectors = matutils.corpus2csc(corpus).T


class SvdModel(object):
    def __init__(self, corpus, id2word, s_exponent=0.0):
        logger.info("calculating truncated SVD")
        lsi = gensim.models.LsiModel(corpus, id2word=id2word, num_topics=DIM, chunksize=1000)
        self.singular_scaled = lsi.projection.s ** s_exponent
        # embeddings = left singular vectors scaled by the (exponentiated) singular values
        self.word_vectors = lsi.projection.u * self.singular_scaled


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    from run_embed import PmiModel, SvdModel  # for pickle

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    in_file = gensim.models.word2vec.LineSentence(sys.argv[1])
    # in_file = gensim.models.word2vec.Text8Corpus(sys.argv[1])
    q_file = sys.argv[2]
    outf = lambda prefix: os.path.join(sys.argv[3], prefix)
    logger.info("output file template will be %s" % outf('PREFIX'))

    sentences = lambda: itertools.islice(in_file, DOC_LIMIT)

    # use only a small subset of all words; otherwise the methods based on matrix
    # decomposition (glove, ppmi) take too much RAM (quadratic in vocabulary size).
    if os.path.exists(outf('word2id')):
        logger.info("dictionary found, loading")
        word2id = utils.unpickle(outf('word2id'))
    else:
        logger.info("dictionary not found, creating")
        id2word = gensim.corpora.Dictionary(sentences(), prune_at=10000000)
        id2word.filter_extremes(keep_n=TOKEN_LIMIT)  # filter out too freq/infreq words
        word2id = dict((v, k) for k, v in id2word.iteritems())
        utils.pickle(word2id, outf('word2id'))
    id2word = gensim.utils.revdict(word2id)

    # filter sentences to contain only the dictionary words
    corpus = lambda: ([word for word in sentence if word in word2id] for sentence in sentences())

    if 'word2vec' in program:
        if os.path.exists(outf('w2v')):
            logger.info("word2vec model found, loading")
            model = utils.unpickle(outf('w2v'))
        else:
            logger.info("word2vec model not found, creating")
            if NEGATIVE:
                model = gensim.models.Word2Vec(size=DIM, min_count=0, window=WINDOW, workers=WORKERS, hs=0, negative=NEGATIVE)
            else:
                model = gensim.models.Word2Vec(size=DIM, min_count=0, window=WINDOW, workers=WORKERS)
            model.build_vocab(corpus())
            model.train(corpus())  # train with 1 epoch
            model.init_sims(replace=True)
            model.word2id = dict((w, v.index) for w, v in model.vocab.iteritems())
            model.id2word = utils.revdict(model.word2id)
            model.word_vectors = model.syn0norm
            utils.pickle(model, outf('w2v'))

    if 'glove' in program:
        if os.path.exists(outf('glove')):
            logger.info("glove model found, loading")
            model = utils.unpickle(outf('glove'))
        else:
            if os.path.exists(outf('glove_corpus')):
                logger.info("glove corpus matrix found, loading")
                cooccur = utils.unpickle(outf('glove_corpus'))
            else:
                logger.info("glove corpus matrix not found, creating")
                cooccur = glove.Corpus(dictionary=word2id)
                cooccur.fit(corpus(), window=WINDOW)
                utils.pickle(cooccur, outf('glove_corpus'))
            logger.info("glove model not found, creating")
            model = glove.Glove(no_components=DIM, learning_rate=0.05)
            model.fit(cooccur.matrix, epochs=10, no_threads=WORKERS, verbose=True)
            model.add_dictionary(cooccur.dictionary)
            model.word2id = dict((utils.to_unicode(w), id) for w, id in model.dictionary.iteritems())
            model.id2word = gensim.utils.revdict(model.word2id)
            utils.pickle(model, outf('glove'))

    if 'pmi' in program:
        if os.path.exists(outf('pmi')):
            logger.info("PMI model found, loading")
            model = utils.unpickle(outf('pmi'))
        else:
            if not os.path.exists(outf('pmi_matrix.mm')):
                logger.info("PMI matrix not found, creating")
                if os.path.exists(outf('cooccur.npy')):
                    logger.info("raw cooccurrence matrix found, loading")
                    raw = numpy.load(outf('cooccur.npy'))
                else:
                    logger.info("raw cooccurrence matrix not found, creating")
                    raw = get_cooccur(corpus(), word2id, window=WINDOW, dynamic_window=False)
                    numpy.save(outf('cooccur.npy'), raw)
                # store the SPPMI matrix in sparse Matrix Market format on disk
                gensim.corpora.MmCorpus.serialize(outf('pmi_matrix.mm'), raw2ppmi(raw, word2id, k_shift=NEGATIVE or 1))
                del raw
            logger.info("PMI model not found, creating")
            model = PmiModel(gensim.corpora.MmCorpus(outf('pmi_matrix.mm')))
            model.word2id = word2id
            model.id2word = id2word
            utils.pickle(model, outf('pmi'))

    if 'svd' in program:
        if os.path.exists(outf('svd')):
            logger.info("SVD model found, loading")
            model = utils.unpickle(outf('svd'))
        else:
            logger.info("SVD model not found, creating")
            model = SvdModel(gensim.corpora.MmCorpus(outf('pmi_matrix.mm')), id2word, s_exponent=0.0)
            model.word2id = word2id
            model.id2word = id2word
            utils.pickle(model, outf('svd'))

    logger.info("evaluating accuracy")
    print accuracy(model, q_file, word2id)  # output result to stdout as well

    logger.info("finished running %s" % program)
