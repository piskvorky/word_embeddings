#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
import cython
import numpy
cimport numpy as np

logger = logging.getLogger(__name__)


def get_cooccur(corpus, word2id, int window, dynamic_window=False):
    """
    Get raw (word x context) => int cooccurence counts, from the `corpus` stream of
    sentences (generator), as a dense NumPy matrix.

    """
    cdef int sentence_no, s_len, id1, id2, reduced_window, pos, pos2
    cdef list sentence
    logger.info("counting raw co-occurrence counts")
    cdef np.ndarray[np.float32_t, ndim=2] cooccur = numpy.zeros((len(word2id), len(word2id)), dtype=numpy.float32)
    for sentence_no, sentence in enumerate(corpus):
        if sentence_no % 100000 == 0:
            logger.info("processing sentence #%i" % sentence_no)
        s_len = len(sentence)
        for pos in range(s_len):
            id1 = word2id.get(sentence[pos], -1)
            if id1 == -1:
                continue  # OOV word in the input sentence => skip
            reduced_window = numpy.random.randint(window) if dynamic_window else 0
            for pos2 in range(max(0, pos - window + reduced_window), min(s_len, pos + window + 1 - reduced_window)):
                id2 = word2id.get(sentence[pos2], -1)
                if id2 == -1 or pos2 == pos:
                    continue  # skip OOV and the target word itself
                cooccur[id1, id2] += 1.0
    logger.info("%i total count, %i non-zeros in raw co-occurrence matrix" %
        (cooccur.sum(), numpy.count_nonzero(cooccur)))
    return cooccur
