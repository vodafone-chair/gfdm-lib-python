import numpy as np
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
from numpy.fft import fftshift, fft, ifft
import filters
import gabor
from copy import deepcopy


def tfshifted_filter_matrix(p, g00):
    N = p.K * p.M
    res = np.zeros((N, N), dtype=complex)
    n = np.arange(N)
    w = np.exp(1j*2*np.pi/p.K)

    for k in range(p.K):
        for m in range(p.M):
            res[:,m*p.K+k] = np.roll(g00, m*p.K) * w**(k*n)
    return res

def ftshifted_filter_matrix(p, g00):
    N = p.K * p.M
    res = np.zeros((N, N), dtype=complex)
    n = np.arange(N)
    w = np.exp(1j*2*np.pi/p.K)

    for k in range(p.K):
        for m in range(p.M):
            res[:,k*p.M+m] = np.roll(g00, m*p.K) * w**(k*n)
    return res

def get_transmitter_matrix(pnew, deltaNMult=None, Krange=None):
    assert(deltaNMult is None)
    assert(Krange is None)
    return tfshifted_filter_matrix(pnew, get_transmitter_pulse(pnew))

def get_truncated_transmitter_matrix(p):
    A = get_transmitter_matrix(p)
    kset,mset=get_kmset(p)

    idx = []
    for m in mset:
        for k in kset:
            idx.append(k + m*p.K)
    return A[:,idx]


def get_receiver_matrix(pnew, recType):
    return tfshifted_filter_matrix(pnew, get_receiver_pulse(pnew, recType)).T.conj()


def get_ambgfun(params, pulse=None, full=False, absOnly=False, pulse2=None):
    if pulse is None:
        pulse = get_transmitter_pulse(params)
    if pulse2 is None:
        pulse2 = pulse

    assert(pulse.shape == pulse2.shape)

    if full:
        nF = params.K*params.M
        nT = nF
        sF = 1
        sT = 1
        dtype = np.complex
    else:
        nF = params.K
        nT = params.M
        sF = params.M
        sT = params.K
        dtype = np.float

    A = np.zeros((nT, nF), dtype=dtype)

    for tau in np.arange(nT):
        Atf = fft(pulse2 * np.roll(pulse.conj(), tau*sT))
        x = Atf[::sF]

        if full:  # full calculation allows complex data
            A[tau, :] = x
        else:     # check if only real or imaginary
            val = np.real(x)
            if absOnly:
                x = abs(x)
            significant = abs(x) > 1e-10

            isR = abs(np.real(x)) > 1e-10
            isI = abs(np.imag(x)) > 1e-10
            both = np.logical_and(isR, isI, significant)
            if np.count_nonzero(both) > 0:
                raise RuntimeError('Ambiguity function is not only real or imaginary!')
            val[isI] = np.imag(x[isI])
            val[isR] = np.real(x[isR])
            # val = np.abs(x);#np.maximum(np.imag(x), np.real(x))
            A[tau, :] = val
    return A.T


def get_transmitter_pulse(pnew):
    rolloff = getattr(pnew, 'a', None)
    p = filters.filterObject(pnew.pulse, rolloff, None).getPulse(pnew.K, pnew.M)
    return fftshift(p)


def get_receiver_pulse(pnew, kind, snr=None):
    if kind == 'MF':
        # return get_transmitter_pulse(pnew)[::-1].conj()
        return get_transmitter_pulse(pnew)
    elif kind == 'ZF':
        return gabor.gabdual(get_transmitter_pulse(pnew), pnew.K, pnew.K)
    elif kind == 'MMSE':
        assert(snr is not None)
        g = get_transmitter_pulse(pnew)
        sigma2 = 10**(-snr/10.)
        K = pnew.K; sK = np.sqrt(K)

        Zg = sK*gabor.zak(g, pnew.K)
        Zg2 = Zg.copy()
        Zg2[abs(Zg) < 1e-6] = 1
        Zgd = 1/(np.conj(Zg2))
        Zgm = 1 / (np.conj(Zg+sigma2*Zgd/K))

        Zgm /= (K * sK)

        Zgm[abs(Zg) < 1e-6] = 0
        gmm = gabor.izak(Zgm, pnew.K)

        s = (gmm * g).sum()
        return gmm/s

        # gd = gabor.gabdual(g, pnew.K, pnew.K)

        # return gabor.gabdual(g+sigma2*gd, pnew.K, pnew.K)
    else:
        raise NotImplementedError("Unknown receiver type!")

def get_dirichlet_pulse(K, M):
    g = filters.filterObject('dirichlet', 0, None).getPulse(K, M)
    g = fftshift(g)
    return g / g[0]


def get_noise_enhancement_factor(p):
    invPulse = get_receiver_pulse(p, 'ZF')
    return sum(abs(invPulse)**2)


def get_effective_transmitter_pulse(p):
    g = get_transmitter_pulse(p)
    K, L = p.K, p.L
    g2 = g[::K/L]
    n = len(g2)

    G2 = fft(g2)
    GG = np.zeros(len(g), dtype=np.complex)
    GG[:n/2] = G2[:n/2]
    GG[-n/2:] = G2[n/2:]
    if n % 2 == 0:
        GG[n/2] = 0.5 * G2[n/2]
        GG[-n/2] = 0.5 * G2[n/2]
    else:
        raise NotImplementedError()
    g3 = ifft(np.real(GG)) * K / L
    if sum(abs(np.imag(g))) < 1e-6:
        return np.real(g3)
    else:
        return g3


def get_shifted_pulses(p, withCP=False, g=None):
    if g is None:
        g = get_transmitter_pulse(p)
    else:
        assert(g.shape == (p.K*p.M, ))
    mset = get_mset(p)

    gi = np.zeros((len(g), len(mset)))
    for i, m in enumerate(mset):
        gi[:, i] = np.roll(g, m*p.K)

    if withCP:
        gi = do_addcp(p, gi)
        gi = do_apply_blockwindow(p, gi)

    return gi


def get_block_window(p):
    l = p.K * p.M + p.NCP
    if p.b == 0 or p.window == 'rect':
        return np.ones(l)

    f = filters.IvanRamp('', 1.0)
    rise, fall = f._getRamp(p.b, p.window)
    c = np.ones((l - 2*p.b, ))
    return np.hstack([rise, c, fall])


def do_addcp(p, signal):
    if p.NCP == 0:
        return signal

    # import ipdb; ipdb.set_trace()

    shape = list(signal.shape)
    shape[0] += p.NCP
    res = np.zeros(tuple(shape), dtype=complex)
    if len(shape) == 1:
        res[p.NCP:] = signal
        res[:p.NCP] = signal[-p.NCP:]
    else:
        res[p.NCP:, :] = signal
        res[:p.NCP, :] = signal[-p.NCP:, :]

    return do_apply_blockwindow(p, res)


def do_apply_blockwindow(p, signal):
    # import ipdb; ipdb.set_trace()
    w = get_block_window(p)
    if len(signal.shape) > 1:
        w = np.tile(w, (signal.shape[1], 1)).T
    res = signal * w
    # if hasattr(p, 'window'):
    #     import ipdb; ipdb.set_trace()
    return res


def get_mset(pnew):
    if hasattr(pnew, 'Mset'):
        return pnew.Mset
    elif hasattr(pnew, 'Mon'):
        assert(pnew.Mon <= pnew.M)
        return np.arange(pnew.M - pnew.Mon, pnew.M)
    else:
        return np.arange(pnew.M)


def get_kset(pnew):
    if hasattr(pnew, 'Kset'):
        return pnew.Kset
    elif hasattr(pnew, 'Kon'):
        assert(pnew.Kon <= pnew.K)
        return np.arange(pnew.Kon)
    else:
        return np.arange(pnew.K)


def get_kmset(pnew):
    return get_kset(pnew), get_mset(pnew)

def copyAndChange(p, **kwargs):
    pnew = deepcopy(p)
    for key, value in kwargs.items():
        setattr(pnew, key, value)
    return pnew

def equivalentOFDM(p):
    ofdm = {}
    ofdm['carriers'] = p.Kon * p.Mon
    ofdm['fftLen'] = p.K * p.Mon
    ofdm['cpLen'] = p.NCP
    ofdm['scheme'] = p.mu
    return ofdm
