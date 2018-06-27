import graycode
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
import numpy as np
from numpy import zeros, repeat
import itertools

def bin2int(bitTuple):
    out = 0
    for bit in bitTuple:
        out = (out << 1) | bit
    return out


def int2bin(v, scheme):
    out = []
    v = int(v)
    while v > 0:
        out.append(v & 1)
        v = v >> 1
    return (0,)*(scheme - len(out)) + tuple(reversed(out))


def bitsToInts(bits, scheme):
    assert(len(bits) % scheme == 0)

    res = [bin2int(bits[scheme*i:scheme*(i+1)]) for i in range(len(bits)/scheme)]
    return res


def intsToBits(ints, scheme):
    if len(ints) == 1:
        return int2bin(ints, scheme)
    return np.array(tuple(itertools.chain(*(int2bin(i, scheme) for i in ints))))
    # return sum((int2bin(i, scheme) for i in ints), ())



def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


symbols_8psk = np.array([np.exp(2j*np.pi*x/8) for x in np.arange(8)])
def get_modulation_symbols(modulation):
    if modulation == 1:
        syms = np.array([-1, 1])
    elif modulation == 3:
        return symbols_8psk
    elif modulation % 2 == 0:
        u = 2**(modulation/2)
        r = np.arange(-u+1, u+1, 2)
        i = 1j*r.T
        c = cartesian((i, r))
        syms = np.sum(c, axis=1)
        syms = syms / np.sqrt(2.*(u**2-1.)/3.)
    else:
        raise NotImplementedError()
    return syms


def qammod(vals, scheme):
    vals = graycode.bin2gray(vals, 2**scheme)
    syms = -1j*get_modulation_symbols(scheme)
    return syms[vals]


def qamdemod(y, scheme):
    if scheme == 3:
        # 8PSK
        angles = np.angle(1j*y) * 8 / (2*np.pi)
        V = np.round(angles)
        V2 = V % 8
        return graycode.gray2bin(V2, 8)

    M = 2**scheme
    sqrtM = np.sqrt(M)

    u = sqrtM
    normfac =  np.sqrt(2.*(u**2-1.)/3.)
    y = y * normfac

    rIdx = np.round(((np.real(y) + (sqrtM-1)) / 2))
    rIdx[rIdx <= -1] = 0
    rIdx[rIdx > sqrtM-1] = sqrtM-1
    iIdx = np.round(((np.imag(y) + (sqrtM-1)) / 2))
    iIdx[iIdx <= -1] = 0
    iIdx[iIdx > sqrtM-1] = sqrtM-1

    z = sqrtM-iIdx-1+sqrtM*rIdx

    z = graycode.gray2bin(z.astype(int), M)
    return z

def de2bi(ints, numBits):
    result = np.zeros((len(ints), numBits), dtype=int)
    for b in range(numBits):
        result[:,b] = (np.bitwise_and(ints, 1 << b) >> b)
    return result

def bi2de(bits):
    L = bits.shape[1]
    values = 2**np.arange(L)[::1]
    return bits.dot(values)
    return np.zeros(bits.shape[0])

def bits2qam_8PSK(bits):
    vals = bi2de(bits)
    return qammod(vals, 3)

def bits2qam(bits, M):
    if M == 8:
        return bits2qam_8PSK(bits)
    L = np.round(np.log2(M))
    bpwr = 2 ** np.arange(L/2, dtype=int)

    mid = int(L/2)
    bits_real = bits[:,:mid]
    bits_imag = bits[:, mid:]

    symb_real = bits_real.dot(bpwr)
    symb_imag = bits_imag.dot(bpwr)

    if int(M) not in g_gray2binMap:
        maxconstell = int(np.sqrt(M))

        g_gray2binMap[int(M)] = graycode.gray2binMap(int(M))
        grayMap = g_gray2binMap[int(M)]
        grayMap = grayMap[:maxconstell]

        const_bitmap = np.argsort(grayMap)
        const_value = np.arange(-maxconstell+1,maxconstell,2)
        const_map = const_value[const_bitmap]
        g_const_map[int(M)] = const_map / qamnormfac(L)

    const_map = g_const_map[int(M)]
    symb = const_map[symb_real] - 1j*const_map[symb_imag]
    return symb

def qam2bits_8PSK(qam):
    vals = qamdemod(qam, 3)
    return de2bi(vals, 3)

def qam2bits(qam, M):
    if M == 8:
        return qam2bits_8PSK(qam)
    maxconstell = np.sqrt(M)
    L = int(np.round(np.log2(M)))
    offset = (maxconstell-1) + 1j*(maxconstell-1)
    qam = np.round((qam.conj() * qamnormfac(L)+offset)/2)

    grayMap = graycode.gray2binMap(int(M))
    grayMap = grayMap[:maxconstell]

    decR = grayMap[qam.real.astype(int)]
    decI = grayMap[qam.imag.astype(int)]
    return np.hstack([de2bi(decR, L/2), de2bi(decI, L/2)])

def llrToSoftSymbol(llrs, mu, offset=0, calcVariance=False):
    L = np.exp(-llrs)
    p1 = L / (1+L)
    p0 = 1-p1;

    P = np.dstack([p0, p1])

    syms = np.arange(2**mu)

    bits = de2bi(syms, mu)
    qam = bits2qam(bits, 2**mu)

    # online algorithm for simultaneous mean and variance calculation, adapted from wikipedia
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
    mean = 0
    sumweight=0
    M2 = 0
    for iq in range(len(qam)):
        bVec = bits[iq, :]
        bbVec = np.tile(bVec.astype(bool), (p1.shape[0], 1))
        pQ = p0.copy()
        pQ[bbVec] = p1[bbVec]

        val = qam[iq] - offset

        probQ = np.prod(pQ, axis=1)
        temp = probQ + sumweight
        delta = val - mean
        R = delta * probQ / temp
        mean = mean + R
        M2 = M2 + sumweight * delta * R.conj()
        sumweight = temp

    if calcVariance:
        var = M2 / sumweight
        return mean, var
    else:
        return mean

def llrToSymbolVariance(llrs, mu, offset=0):
    L = np.exp(llrs)
    p1 = L / (1+L)
    p0 = 1-p1;

    P = np.dstack([p0, p1])

    syms = np.arange(2**mu)

    bits = de2bi(syms, mu)
    qam = bits2qam(bits, 2**mu)
    mean = 0
    for iq in range(len(qam)):
        bVec = bits[iq, :]
        bbVec = np.tile(bVec.astype(bool), (p1.shape[0], 1))
        pQ = p0.copy()
        pQ[bbVec] = p1[bbVec]

        probQ = np.prod(pQ, axis=1)
        mean = mean + (qam[iq]-offset) * probQ
    return mean


def clipLLR(llrs, clip):
    res = llrs.copy()
    res[res > clip] = clip
    res[res < -clip] = -clip
    return res

def qamnormfac(mu):
    assert mu != 8
    return np.sqrt(2./3. * (2**mu-1))

g_gray2binMap = dict()
g_const_map = dict()
