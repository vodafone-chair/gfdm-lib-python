from fractions import gcd
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
import numpy as np
from numpy.fft import fft, ifft
from numpy import exp, floor


def zak(g, a):
    assert(len(g.shape) == 1 or g.shape[1] == 1)
    assert(g.shape[0] % a == 0)

    gg = g.reshape((a, -1), order='F')
    GG = fft(gg, axis=1)
    return GG / np.sqrt(a)

def izak(Z, a):
    ZZ = np.sqrt(a) * ifft(Z, axis=1)
    return ZZ.flatten(order='F')

def zakm(M, K):
    # return the matrix to doing ZAK transform
    res = np.zeros((M*K, M*K), dtype=complex)
    w = exp(1j*2*np.pi/M)
    for m in range(M):
        for k in range(K):
            res[k+m*K,k+K*np.arange(M)] = w**(m*np.arange(M))
    return res/np.sqrt(K)


def zakconvolve(Z1, Z2):
    assert(Z1.shape == Z2.shape)
    K, M = Z1.shape
    Zc = np.array([[sum(exp(1j*2*np.pi*floor((1.0*n-m)/K)*k/M)*Z1[(n-m) % K,k]*Z2[m,k] for m in range(K)) for n in range(K)] for k in range(M)]).T
    return Zc * np.sqrt(K)


# Copied from ltfat-lib
def gabdual(g, a, M):
    return np.real(comp_gabdual_long(g, a, M).flatten())


def comp_gabdual_long(g, a, M):
    L = len(g)
    R = 1
    assert(len(g.shape) == 1 or g.shape[1] == 1)

    gf = comp_wfac(g, a, M)

    b = L / M
    N = L / a
    c = gcd(a, M)
    d = gcd(b, N)

    p = b / d
    q = N / d

    gdf = np.zeros((p*q*R, c*d), dtype=complex)
    G = np.zeros((p, q*R), dtype=complex)
    assert(p == 1)
    assert(q == 1)
    for ii in range(c*d):
        G[:] = gf[:, ii]
        S = np.dot(G, G.T.conj())
        Gpinv = 1 / S * G

        gdf[:, ii] = Gpinv[:]

    gd = comp_iwfac(gdf, L, a, M)
    return np.real(gd)


def comp_wfac(g, a, M):
    L = g.shape[0]
    R = 1
    assert(len(g.shape) == 1 or g.shape[1] == 1)

    N = L / a

    c = gcd(a, M)
    p = a / c
    q = M / c
    d = N / q

    gf = np.zeros((p, q*R, c, d), dtype=complex)

    assert(p == 1)
    assert(c != 1 or d != 1)

    for w in range(R):
        for s in range(d):
            for l in range(q):
                gf[0, l+q*w,:,s]=g[np.arange(c)+(-l*a+s*p*M % L)]

    assert(d > 1)
    gf = fft(gf, axis=3)
    gf *= np.sqrt(M)
    return gf.reshape((p*q*R, c*d), order='F')


def comp_iwfac(gf, L, a, M):
    R = (gf.shape[0]*gf.shape[1])/L

    N = L / a
    b = L / M

    c = gcd(a, M)
    p = a / c
    q = M / c
    d = N / q
    gf = gf.reshape((p, q*R, c, d), order='F')
    gf /= np.sqrt(M)

    if d > 1:
        gf = ifft(gf, axis=3)

    g = np.zeros((L, R), dtype=complex)
    for w in range(R):
        for s in range(d):
            for l in range(q):
                for k in range(p):
                    g[np.arange(c)+(k*M-l*a+s*p*M % L),w] = gf[k,l+q*w,:,s].reshape((c,), order='F')

    return g
