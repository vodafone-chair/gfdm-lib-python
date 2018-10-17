# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
import numpy as np
from numpy.fft import fft, ifft, fftshift
from gfdmutil import get_transmitter_pulse


class Modulator(object):
    def modulate(self, block):
        raise NotImplementedError()


class DefaultModulator(Modulator):
    def __init__(self, pnew):
        self._pnew = pnew

    def modulate(self, block):
        return self._modulate_time(block)

    def _modulate_freq(self, block):
        L, Kon, M, K = self._pnew.L, self._pnew.Kon, self._pnew.M, self._pnew.K
        N = M*K

        block = block.copy().T
        g = get_transmitter_pulse(self._pnew)
        g2 = g[::K/L]; G2 = fft(g2)
        DD = np.tile(fft(block, axis=0), (L, 1))

        X = np.zeros((N,))
        offset = 0

        for k in range(K):
            # iterate over carriers. Since this is FFT based,
            # K samples per symbol produce K subcarriers.
            carrier = np.zeros((N,), dtype=complex)
            carrier[0:L*M] = fftshift(DD[:, k] * G2)

            # shift data to the correct frequency position:
            # - offset moves lowest SC (subcarrier 1) to lowest frequency,
            # - L*M/2 equals half of the pulse shape bandwidth
            # - M*(k-1) shifts to the correct carrier
            carrier = np.roll(carrier, offset + -L*M/2 + M*k)
            X = X + carrier
        X = (K/L) * X.flatten(1)
        x = ifft(X)
        signal = x
        return signal

    def _modulate_time(self, block):
        g = get_transmitter_pulse(self._pnew)
        K, M = self._pnew.K, self._pnew.M
        N = K * M

        B = ifft(block, axis=0) * K

        signal = np.zeros((N, M), dtype=complex)
        for m in range(M):
            b = np.tile(B[:,m], M)
            signal[:, m] = b * np.roll(g, m*K)
        return signal.sum(axis=1)


def do_modulate(p, D):
    return DefaultModulator(p).modulate(D)
