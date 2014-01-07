#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Modified by Sébastien Jean

#Make neu1 replicate the behavior of work

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset

from cpython cimport PyCObject_AsVoidPtr
from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t


ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil

ctypedef void (*fast_sentence_ptr) (
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[1000],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1, const int size,
    np.uint32_t indexes[1000], const REAL_t alpha, REAL_t *work, int i, int bags_before, int bags_after, int words_per_bag, int sentence_len) nogil

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef fast_sentence_ptr fast_sentence


DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0


cdef void fast_sentence0(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[1000],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1, const int size,
    np.uint32_t indexes[1000], const REAL_t alpha, REAL_t *work, int i, int bags_before, int bags_after, int words_per_bag, int sentence_len) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g
    cdef int m

    cdef int count

    cdef int bag_index
    cdef int start
    cdef int end

    for bag_index in range(-bags_before,0):
        start = max(0, i + bag_index*words_per_bag)
        end = i + (bag_index+1)*words_per_bag

        memset(neu1, 0, size * cython.sizeof(REAL_t)) #set work to zero?

        count = 0
        for m in range(start, end):
            if m == i or codelens[m] == 0:
                continue
            else:
                count = count + 1
                saxpy(&size,&ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

        if count > 0:
            for c in range(size):
                neu1[c] = neu1[c] / count

        memset(work, 0, size * cython.sizeof(REAL_t)) #set work to zero?
        for b in range(codelens[i]):
            row2 = word_point[b] * size
            f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1[row2], &ONE) #dsdot
            #if f <= -MAX_EXP or f >= MAX_EXP:
            #    continue
            #f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            f = (<REAL_t>1.0)/(<REAL_t>1.0 + <REAL_t>exp(-f))
            g = (1 - word_code[b] - f) * alpha
            saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
            saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

        for m in range(start,end):
            if m == i or codelens[m] == 0:
                continue
            else:
                saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

    for bag_index in range(0, bags_after):
        start = i + bag_index*words_per_bag + 1
        end = min(sentence_len, i + (bag_index+1)*words_per_bag + 1)

        memset(neu1, 0, size * cython.sizeof(REAL_t))

        count = 0
        for m in range(start, end):
            if m == i or codelens[m] == 0:
                continue
            else:
                count = count + 1
                saxpy(&size,&ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

        if count > 0:
            for c in range(size):
                neu1[c] = neu1[c] / count

        memset(work, 0, size * cython.sizeof(REAL_t))
        for b in range(codelens[i]):
            row2 = word_point[b] * size
            f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1[row2], &ONE) #dsdot
            #if f <= -MAX_EXP or f >= MAX_EXP:
            #    continue
            #f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            f = (<REAL_t>1.0)/(<REAL_t>1.0 + <REAL_t>exp(-f))
            g = (1 - word_code[b] - f) * alpha
            saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
            saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

        for m in range(start,end):
            if m == i or codelens[m] == 0:
                continue
            else:
                saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)


cdef void fast_sentence1(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[1000],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1, const int size,
    np.uint32_t indexes[1000], const REAL_t alpha, REAL_t *work, int i, int bags_before, int bags_after, int words_per_bag, int sentence_len) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g
    cdef int m

    cdef int count

    cdef int bag_index
    cdef int start
    cdef int end

    for bag_index in range(-bags_before,0):
        start = max(0, i + bag_index*words_per_bag)
        end = i + (bag_index+1)*words_per_bag

        memset(neu1, 0, size * cython.sizeof(REAL_t)) #set work to zero?

        count = 0
        for m in range(start, end):
            if m == i or codelens[m] == 0:
                continue
            else:
                count = count + 1
                saxpy(&size,&ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

        if count > 0:
            for c in range(size):
                neu1[c] = neu1[c] / count

        memset(work, 0, size * cython.sizeof(REAL_t))
        for b in range(codelens[i]):
            row2 = word_point[b] * size
            f = <REAL_t>sdot(&size, neu1, &ONE, &syn1[row2], &ONE) #sdot
            #if f <= -MAX_EXP or f >= MAX_EXP:
            #    continue
            #f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            f = (<REAL_t>1.0)/(<REAL_t>1.0 + <REAL_t>exp(-f))
            g = (1 - word_code[b] - f) * alpha
            saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
            saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

        for m in range(start,end):
            if m == i or codelens[m] == 0:
                continue
            else:
                saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

    for bag_index in range(0, bags_after):
        start = i + bag_index*words_per_bag + 1
        end = min(sentence_len, i + (bag_index+1)*words_per_bag + 1)

        memset(neu1, 0, size * cython.sizeof(REAL_t)) #set work to zero?

        count = 0
        for m in range(start, end):
            if m == i or codelens[m] == 0:
                continue
            else:
                count = count + 1
                saxpy(&size,&ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

        if count > 0:
            for c in range(size):
                neu1[c] = neu1[c] / count

        memset(work, 0, size * cython.sizeof(REAL_t))
        for b in range(codelens[i]):
            row2 = word_point[b] * size
            f = <REAL_t>sdot(&size, neu1, &ONE, &syn1[row2], &ONE) #sdot
            #if f <= -MAX_EXP or f >= MAX_EXP:
            #    continue
            #f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            f = (<REAL_t>1.0)/(<REAL_t>1.0 + <REAL_t>exp(-f))
            g = (1 - word_code[b] - f) * alpha
            saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
            saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

        for m in range(start,end):
            if m == i or codelens[m] == 0:
                continue
            else:
                saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

"""
cdef void fast_sentence2(
    #const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[1000],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1, const int size,
    #const REAL_t alpha, REAL_t *work, int i, int j, int k) nogil:
    #np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work, int i, int j, int k) nogil:
    np.uint32_t indexes[1000], const REAL_t alpha, REAL_t *work, int i, int bags_before, int bags_after, int words_per_bag, int sentence_len) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g
    cdef int m

    cdef int count = 0

    memset(neu1, 0, size * cython.sizeof(REAL_t)) #set work to zero?

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count = count + 1
            #saxpy(&size,&ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
            for c in range(size):
                neu1[c] = neu1[c] + syn0[indexes[m] * size + c]
            #    neu1_copy[c] = neu1_copy[c] + syn0[indexes[m] * size + c]

#    if count > 0: #divide or not?
#        for c in range(size):
#            neu1[c] = neu1[c] / count

    for a in range(size):
        work[a] = <REAL_t>0.0
#        neu1[a] = <REAL_t>0.0
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        #f = <REAL_t>0.2
        for a in range(size):
            f += neu1[a] * syn1[row2 + a]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        for a in range(size):
            work[a] += g * syn1[row2 + a]
        for a in range(size):
            syn1[row2 + a] += g * neu1[a]

    #for a in range(size):
    #    neu1[a] += work[a]    
    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue
        else:
    #        #saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)
            for a in range(size):
                syn0[indexes[m] * size + a] = syn0[indexes[m] * size + a] + work[a]
"""


DEF MAX_SENTENCE_LEN = 1000

def train_sentence(model, sentence, alpha, _work, _neu1):
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size
    cdef int reduce = model.reduce
    cdef int direction = model.direction

    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.int32_t reduced_half_bags[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int half_bags = model.half_bags
    cdef int words_per_bag = model.words_per_bag

    cdef int i, m
    cdef long result = 0
    cdef int c, count #Added this line

    cdef int bags_before
    cdef int bags_after

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            if reduce > 0:
                reduced_half_bags[i] = np.random.randint(half_bags)
            else:
                reduced_half_bags[i] = 0
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0: #out of vocabulary
                continue

            if direction <= 0:
                bags_before = min(half_bags - reduced_half_bags[i], (i-1)/words_per_bag + 1)
            else:
                bags_before = 0
            if direction >= 0:
                bags_after = min(half_bags - reduced_half_bags[i], (sentence_len-i-2)/words_per_bag + 1)
            else:
                bags_after = 0

            fast_sentence(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, bags_before, bags_after, words_per_bag, sentence_len) #need a way to access stuff                         

    return result


def init():
    
    #Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    #into table EXP_TABLE.

    global fast_sentence
    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res

    if (abs(d_res - expected) < 0.0001):
        fast_sentence = fast_sentence0
        print "0"
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        fast_sentence = fast_sentence1
        print "1"
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        fast_sentence = fast_sentence1 #modified (and false) The last optimization has not been implemented.
        "print 2"
        return 2


FAST_VERSION = init()  # initialize the module