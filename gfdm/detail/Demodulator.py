import numpy as np
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
from numpy.fft import fft, ifft, fftshift
import gfdmutil as util

class Demodulator(object):
    def demodulate(self, signal):
        raise NotImplementedError()

class DefaultDemodulator(Demodulator):
    def __init__(self, pnew, kind):
        self._pnew = pnew
        self._kind = kind

        assert(kind in ['ZF', 'MF', 'FFT', 'ZFFFT', 'custom'])
        self._prepareDemodulation()

    def _prepareDemodulation(self):
        if self._kind == 'ZF':
            self._B = util.get_receiver_matrix(self._pnew, 'ZF')
        elif self._kind == 'ZFFFT':
            L = min(16, self._pnew.K)
            gzf = util.get_receiver_pulse(self._pnew, 'ZF')
            self._Gzf = fft(gzf[::self._pnew.K/L], axis=0)
        elif self._kind in ['MF', 'FFT']:
            L, K = self._pnew.L, self._pnew.K
            g = util.get_transmitter_pulse(self._pnew)
            g2 = g[::K/L]; G2 = fft(g2, axis=0)
            self._Gmf = G2.conj()

    def demodulate(self, signal):
        if self._kind in ['MF', 'FFT']:
            return self.__demodulateFFT(signal)
        elif self._kind == 'ZF':
            return self.__demodulateZF(signal)
        elif self._kind == 'ZFFFT':
            return self.__demodulateZFFFT(signal)
        elif self._kind == 'custom':
            return self.__demodulateInFD(signal, self._Gcustom)
        else:
            raise NotImplementedError("%s receiver type not implemented!" % self._kind)

    def setFilter(self, pulse, L):
        assert(self._kind == 'custom')
        K = self._pnew.K
        L = min(K, L)
        gg = pulse[::K/L]; GG = fft(gg, axis=0)
        self._Gcustom = GG

    def __demodulateFFT(self, signal):
        return self.__demodulateInFD(signal, self._Gmf)

    def __demodulateZF(self, signal):
        dhat = np.dot(self._B, signal)
        return dhat.reshape(self._pnew.Kon, self._pnew.M, order='F')

    def __demodulateZFFFT(self, signal):
        return self.__demodulateInFD(signal, self._Gzf)

    def __demodulateInFD(self, signal, G):
        M, K, Kon = self._pnew.M, self._pnew.K, self._pnew.Kon
        L = len(G) / M
        Xhat = fft(signal, axis=0)

        Dhat = np.zeros((K, M), dtype=complex)
        offset = 0
        for k in range(K):
            # do processing from Modulator backwards
            carrier = np.roll(Xhat, -offset + L*M/2 - M*k)
            carrier = fftshift(carrier[:L*M])
            carrierMatched = carrier * G
            dhat = ifft(carrierMatched, axis=0)[::L]  # downsampling
            Dhat[k, :] = dhat
        return Dhat

def do_demodulate(p, x, recType):
    return Demodulator(p, recType).demodulate(x);
