import numpy as np
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
from numpy import zeros

def bin2grayMap(order):
    assert order != 8
    k = int(round(np.log2(order)))
    k2 = int(k/2)
    mapping = np.arange(order, dtype=int)

    symbolI = mapping >> k2
    symbolQ = np.bitwise_and(mapping, (order-1)>>k2)

    i = 1
    while i < k2:
        tempI = symbolI
        tempI = tempI >> int(i)
        symbolI = symbolI ^ tempI

        tempQ = symbolQ
        tempQ = tempQ >> int(i)
        symbolQ = symbolQ ^ tempQ
        i = i + 1

    symbolIndex = (symbolI<<k2) + symbolQ
    mapping = symbolIndex
    return mapping

def gray2binMap(order):
    assert order != 8
    mapping = bin2grayMap(order)
    mapping2 = ismember(np.arange(order), mapping)[1]
    return mapping2


def bin2gray(x, order):
    if order == 8:
        X = np.array([0,1,3,2,7,6,4,5])
        return X[x]


    k = int(round(np.log2(order)))
    k2 = int(k/2)
    mapping = np.arange(order, dtype=int)

    symbolI = mapping >> k2
    symbolQ = np.bitwise_and(mapping, (order-1)>>k2)

    i = 1
    while i < k2:
        tempI = symbolI
        tempI = tempI >> int(i)
        symbolI = symbolI ^ tempI

        tempQ = symbolQ
        tempQ = tempQ >> int(i)
        symbolQ = symbolQ ^ tempQ
        i = i + 1

    symbolIndex = (symbolI<<k2) + symbolQ
    mapping = symbolIndex
    output = mapping[x]

    return output

def gray2bin(x, order):
    if order == 8:
        X = np.array([0, 1, 3, 2, 6, 7, 5, 4])
        return X[x.astype(int)]

    k = int(round(np.log2(order)))
    k2 = int(k/2)
    mapping = np.arange(order, dtype=int)

    symbolI = mapping >> k2
    symbolQ = np.bitwise_and(mapping, (order-1)>>k2)

    i = 1
    while i < k2:
        tempI = symbolI
        tempI = tempI >> int(i)
        symbolI = symbolI ^ tempI

        tempQ = symbolQ
        tempQ = tempQ >> int(i)
        symbolQ = symbolQ ^ tempQ
        i = i + 1

    symbolIndex = (symbolI<<k2) + symbolQ
    mapping = symbolIndex

    mapping2 = ismember(np.arange(order), mapping)[1]

    output = mapping2[x]

    return output



def ismember(a, b):
    tf = np.in1d(a,b) # for newer versions of numpy
    # tf = np.array([i in b for i in a])
    u = np.unique(a[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
    return tf, index
