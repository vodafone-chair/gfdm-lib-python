from numpy import sin, cos, flatnonzero, ix_, linspace, sqrt, r_, zeros, exp, real, imag
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
from numpy.fft import fft, ifft, fftshift, ifftshift
from math import pi
import warnings
import numpy as np


def rc(a, nsamples, nperiods):
    # proakis p. 560, 608

    # a         rolloff factor
    # nperiods  number of periods
    # nsamples 	number of samples per period, EVEN number!

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        t = linspace( -nperiods/2., nperiods/2., nsamples*nperiods, endpoint=False )

        if a == 0:
            g = np.sinc(t)
        else:
            g = sin(pi*t) / (pi*t) * cos( pi*a*t ) / ( 1-(2*a*t)**2 );

            g[ t==0 ] = 1;
            g[ abs(t)==1/(2*a) ] = sin( pi/(2*a) ) / (pi/(2*a)) * pi/4;

        g = 1/sqrt( sum( g**2 ) ) * g;

    return (g, t, 'rc')


def rootrc(a, nsamples, nperiods):
    # proakis p. 560
    #
    # a         rolloff factor
    # nperiods  number of periods
    # nsamples 	number of samples per period, EVEN number!

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        t = linspace( -nperiods/2., nperiods/2., nsamples*nperiods, endpoint=False )

        g = ( sin( pi*t*(1-a) ) + 4*a*t*cos( pi*t*(1+a) ) ) / ( pi*t*( 1-(4*a*t)**2 ) )

        g[ t==0 ] = 1-a+4*a/pi
        g[ abs(t)==1/(4*a) ] = a/sqrt(2)*( (1+2/pi)*sin( pi/(4*a) ) + (1-2/pi)*cos( pi/(4*a) ) )

        infi = flatnonzero (abs(t-(1/(4*a)))<0.000001)
        infi = r_[infi,flatnonzero(abs(t+1/(4*a))<0.0000001)]
        ti = t[ix_(infi)]
        vals =  (pi*(1-a)*cos(pi*ti*(1-a))+4*a*(cos(pi*ti*(1+a))-pi*(1+a)*ti*sin(pi*ti*(1+a))))/(pi*(1-3*(4*a*t[infi])**2))
        g[ix_(infi)] = vals

        g = 1/sqrt( sum( g**2 ) ) * g

    return (g, t, 'rrc')

def btrc(a, N, M):
    t = linspace(-M/2., M/2., N*M, endpoint=False)
    if a == 0:
        g = np.sinc(t)
    else:
        b = np.log(2)/(0.5 * a)
        g = np.sinc(t) * (4*b*t*pi*np.sin(pi*a*t)+2*b**2*cos(pi*a*t)-b**2)/(4*pi**2*t**2+b**2)

    g = 1/sqrt(sum(g**2)) * g
    return (g, t, 'btrc')


def gauss(a, nsamples, nperiods):
    t = linspace(-nperiods/2., nperiods/2., nsamples*nperiods, endpoint=False)

    f = lambda t: exp(-(pi*t/a)**2)
    g = f(t)

    g = g / sqrt(sum(g**2))
    return (g, t, 'gauss')


def rect(nsamples, nperiods):
    t = linspace( -nperiods/2, nperiods/2, nsamples*nperiods, endpoint=False )

    g = zeros(nsamples*nperiods)
    g[:nsamples/2] = 1;
    g[-nsamples/2:] = 1;

    g = g / sqrt(sum(g**2));
    return (fftshift(g), t, 'rect')


def rect_shift(nsamples, nperiods):
    g, t, bla = rect(nsamples, nperiods)
    g = fftshift(g)
    g = np.roll(g, -nsamples/2)
    return (g, t, 'rect')


def triang(nsamples, nperiods):
    (g, t, x) = rect(nsamples, nperiods)
    g[:len(g)/2] = 0
    g = np.roll(g, +nsamples/2)
    # return (g, t, 'triang')

    g = (real(ifft(fft(g)**2)))
    g = g / sqrt(sum(g**2))
    return (g, t, 'triang')


def sinc(N, M):
    t = linspace( -M/2., M/2., N*M, endpoint=False )
    g = np.sinc(t)
    g = g / sqrt(sum(g**2))
    return (g, t, 'sinc')


def filterObject(name, param1=None, param2=None):
    if name in ['rrc', 'rc', 'btrc', 'rect', 'rect_shift', 'triang', 'gauss']:
        return SimpleFilter(name, param1)
    elif name[-3:] in ['_fd', '_td']:
        return IvanRamp(name, param1)
    elif name.startswith('isifree_'):
        n = name[name.find('_')+1:]
        return IsiFreeXia(n, param1, param1, param2)
    elif name == 'xia1st_fd':
        return IsiFreeXia('lin', param1, param1, param2)
    elif name == 'xia4th_fd':
        return IsiFreeXia('paper', param1, param1, param2)
    elif name == 'gen_nyq':
        return ParametricNyquist(param1, param2)
    elif name == 'btrc_spec':
        return ParametricNyquist(1, np.exp)
    elif name == 'rc_fd':
        # return GenericFreqRolloff(lambda t: 0.5 * (1-cos(pi*t)), param1)
        return IvanRamp('rc_fd', param1)
    elif name == 'rc_td':
        return GenericTimeRolloff(lambda t: 0.5 * (1 - cos(pi*t)), param1)
    elif name == 'rrc_fd':
        return GenericFreqRolloff(lambda t: sqrt(0.5 * (1-cos(pi*t))), param1)
    elif name == 'rrc_td':
        return GenericTimeRolloff(lambda t: sqrt(0.5 * (1 - cos(pi*t))), param1)
    elif name in ['sinc_fd', 'dirichlet']:
        return SincFD()
    elif name == 'sinc':
        return SimpleFilter('sinc')
    elif name == 'rc4th_td':
        return GenericTimeRolloff(lambda t: 0.5 * (1 - cos(pi*xia4thPoly(t))), param1)
    elif name == 'rrc4th_td':
        return GenericTimeRolloff(lambda t: sqrt(0.5 * (1 - cos(pi*xia4thPoly(t)))), param1)
    elif name == 'rc4th_fd':
        return GenericFreqRolloff(lambda t: 0.5 * (1 - cos(pi*xia4thPoly(t))), param1)
    elif name == 'rrc4th_fd':
        return GenericFreqRolloff(lambda t: sqrt(0.5 * (1 - cos(pi*xia4thPoly(t)))), param1)
    else:
        raise NotImplementedError('Filter %s not defined' % name)


class Filter(object):
    def getPulse(self, params):
        """Should return the pulse centered around 0."""
        raise NotImplementedError()

    def getFreq(self, params):
        g = fftshift(self.getPulse(params))
        return fftshift(fft(g, axis=0) / len(g))


class SimpleFilter(Filter):
    def __init__(self, name, param1=None):
        self.param1 = param1
        aKM = lambda K, M: (self.param1, K, M)
        KM = lambda K, M: (K, M)
        if name == 'rrc':
            self.__func = rootrc
            self.__args = aKM
        elif name == 'rc':
            self.__func = rc
            self.__args = aKM
        elif name == 'gauss':
            self.__func = gauss
            self.__args = aKM
        elif name == 'btrc':
            self.__func = btrc
            self.__args = aKM
        elif name == 'rect':
            self.__func = rect
            self.__args = KM
        elif name == 'rect_shift':
            self.__func = rect_shift
            self.__args = KM
        elif name == 'sinc':
            self.__func = sinc
            self.__args = KM
        elif name == 'triang':
            self.__func = triang
            self.__args = KM
        else:
            raise NotImplementedError()

    def getPulse(self, K, M):
        return self.__func(*self.__args(K, M))[0]


class SincFD(Filter):
    def getPulse(self, K, M):
        M, N = M, K
        w = M

        G = zeros((N*M,))
        G[:M] = 1
        G = np.roll(G, -(M-2)/2)

        g = fftshift(ifft(G, axis=0))
        g = g / sqrt(sum(abs(g)**2))
        return g


class ParametricNyquist(Filter):
    def __init__(self, n, G):
        self._n = n;
        self._G = G

    def getPulse(self, params):
        def inlClosest(vec, val):
            return abs(vec - val).argmin()
        def inlEstimate_inverse(f, y):
            aim = lambda x: abs(y - f(x))
            res = fmin(aim, (1,), xtol=1e-10, disp=False)
            return res


        upsample = 1
        N, M, K = params.N, params.M, params.K
        B = 0.5
        a = params.a
        n, G = self._n, self._G

        f = linspace(-N/2, N/2, upsample*N*M, endpoint=False)
        af = abs(f)
        res = np.zeros_like(f, dtype=complex)

        d = 1e-3 if a == 0 else 0
        res[af >= B*(1+a+d)] = 0
        res[af <= B*(1-a-d)] = 1

        if a > 0:
            gamma0 = inlEstimate_inverse(G, 0.5)
            gamma_n = gamma0 / ((a*B)**n)

            i1 = (B*(1-a) <= af) & (af <= B); f1 = af[i1]
            i2 = (B < af) & (af <= B*(1+a)); f2 = af[i2]
            res[i1] = G(gamma_n*(f1-B*(1-a))**n)
            res[i2] = 1 - G(gamma_n*(B*(1+a)-f2)**n)
        else:
            if params.M % 2 == 0:
                res[inlClosest(f, -0.5)] = 0.5
                res[inlClosest(f, +0.5)] = 0.5


        g = ifft(ifftshift(res));
        s = sum(abs(imag(g)))
        if s > 1e-5:
            assert(s <= 1e-5)
        g = real(g)
        g = g / sqrt(sum(g**2))
        g = fftshift(g)
        return g


class IvanRamp(Filter):
    def __init__(self, type, rolloff):
        self._a = rolloff
        self._type = type

    def getPulse(self, K, M):
        if self._type.endswith("_fd"):
            r, f = self._getRamp(M, self._type[:-3])
            G = np.zeros((K*M, ), dtype=complex)
            G[:M] = f; G[-M:] = r
            g = fftshift(ifft(G))
        elif self._type.endswith("_td"):
            r, f = self._getRamp(K, self._type[:-3])
            g = zeros((K*M, ), dtype=complex)
            g[:K] = f; g[-K:] = r
            g = fftshift(g)

        return g / np.sqrt(sum(abs(g) ** 2))

    def _getRamp(self, M, type):
        m = np.arange(M)
        eps = np.spacing(1)
        a = self._a

        R = (m-M/2.0-eps)/(a*M)+0.5;
        R[R<0] = 0; R[R>1] = 1; F = 1-R;
        R4th=R**4*(35 - 84*R+70*R**2-20*R**3);F4th=1-R4th;

        if type == 'rootramp':
            rise = R**0.5; fall = (1-rise**2)**0.5
        elif type == 'ramp':
            rise = R; fall = 1-rise
        elif type == 'root4th':
            rise = R4th**0.5; fall = (1-rise**2)**0.5
        elif type == '4th':
            rise = R4th; fall = 1-rise
        elif type == 'rrc':
            rise = (0.5*(cos(F*pi)+1))**0.5; fall = (1-rise**2)**0.5
        elif type == 'rc':
            rise = 0.5 * (cos(F*pi)+1); fall = 1 - rise
        elif type == 'rrc4th':
            rise = (0.5*(cos(F4th*pi)+1))**0.5; fall = (1-rise**2)**0.5
        elif type == 'rc4th':
            rise = 0.5*(cos(F4th*pi)+1); fall = 1-rise
        elif type == 'xia':
            rise = 0.5 * (exp(-1j*F*pi)+1); fall = 1-rise
        elif type == 'xia4th':
            rise = 0.5*(exp(-1j*F4th*pi)+1); fall = 1-rise
        else:
            raise NotImplementedError('Unknown Rampfilter: %s' % self._type)
        return rise, fall


class RolloffFilter(Filter):
    def _rise(self, argrise):
        raise NotImplementedError()

    def _fall(self, argfall):
        raise NotImplementedError()

    def getPulse(self, K, M):
        raise NotImplementedError()

    def _getEdges(self, linArgs, a):
        argrise = (linArgs + (1+a)/2.)/a
        argfall = (linArgs - (1-a)/2.)/a

        rise = self._rise(argrise)
        fall = self._fall(argfall)

        res = np.zeros_like(linArgs.copy(), dtype=complex)
        res[abs(linArgs) >= (1+a)/2.] = 0
        res[abs(linArgs) <= (1-a)/2.] = 1

        iRise = abs(linArgs + 0.5) <= a/2.
        iFall = abs(linArgs - 0.5) <= a/2.

        res[iRise] = rise[iRise]
        res[iFall] = fall[iFall]
        return res


class FreqRolloff(RolloffFilter):
    def __init__(self, rolloff):
        self._rolloff = rolloff

    def getPulse(self, K, M):
        f = linspace(-K/2, K/2, K*M, endpoint=False)

        G = self._getEdges(f, self._rolloff)
        sG = fftshift(G)

        g = ifft(sG)
        g = real(g)
        g = np.roll(g, K*M/2)
        g = g[:K*M]
        g = np.roll(g, -K*M/2)
        g = g / sqrt(sum(g**2))
        g = fftshift(g)
        return g


class TimeRolloff(RolloffFilter):
    def __init__(self, rolloff):
        self._rolloff = rolloff

    def getPulse(self, params):
        N, M, K = params.N, params.M, params.K
        t = linspace(-M/2., M/2., M*N, endpoint=False)

        vals = self._getEdges(t, params.a)
        vals = real(vals)
        vals = vals / sqrt(sum(vals**2))
        return vals


def xia4thPoly(x):
    return x**4*(35-84*x+70*x**2-20*x**3)


class IsiFreeXia(FreqRolloff):
    def __init__(self, ny, rolloff, param1=None, param2=None):
        super(IsiFreeXia, self).__init__(rolloff)
        self._param1 = param1
        self._param2 = param2
        self._nyType = ny
        pass

    def _rise(self, argrise):
        return 0.5 * (1 - np.exp(1j * pi * self._ny(argrise)))

    def _fall(self, argfall):
        return 0.5 * (1 + np.exp(1j * pi * self._ny(argfall)))

    def _ny(self, x):
        m = {'paper' : self._ny_paper,
             'lin'   : self._ny_lin,
             'tanh'  : self._ny_tanh,
             'sin'   : self._ny_sin,
             'pot'   : self._ny_pot,
             'exp'   : self._ny_exp,
             'log'   : self._ny_log
             }

        return m[self._nyType](x)

    def __ny01(self, x):
        res = x.copy()
        res[x >= 1] = 1
        res[x < 0] = 0
        return res

    def _ny_paper(self, x):
        n = self.__ny01(x)
        return xia4thPoly(n)

    def _ny_lin(self, x):
        return self.__ny01(x)

    def _ny_tanh(self, x):
        sc = self._param1
        n = sc * (self.__ny01(x)*2 - 1)

        vals = np.tanh(n)
        vals = vals - min(vals)
        vals = vals / max(vals)
        return vals

    def __map01Branch(self, x, func):
        n = 2*self.__ny01(x) - 1
        sign = np.sign(n)

        vals = sign * func(abs(n))
        return 0.5 * (vals + 1)

    def _ny_sin(self, x):
        pot = self._param1
        return self.__map01Branch(x, lambda t: np.sin(pi/2*t)**pot)

    def _ny_pot(self, x):
        pot = self._param1
        return self.__map01Branch(x, lambda t: t**pot)

    def _ny_exp(self, x):
        pot = self._param1
        return self.__map01Branch(x, lambda t: ((exp(t)-1)/(exp(1)-1))**pot)

    def _ny_log(self, x):
        pot = self._param1
        return self.__map01Branch(x, lambda t: np.log(1+t*(exp(1)-1))**pot)


class BTRC(FreqRolloff):
    def _rise(self, argrise):
        res = 1 - np.exp(2*np.log(0.5)*argrise)
        res2 = np.exp(2*np.log(0.5)*(1-argrise))
        res[argrise >= 0.5] = res2[argrise >= 0.5]
        return res

    def _fall(self, argfall):
        return self._rise(1-argfall)


class GenericTimeRolloff(TimeRolloff):
    def __init__(self, func, rolloff):
        super(GenericTimeRolloff, self).__init__(rolloff)
        self._func = func

    def _rise(self, argrise):
        return self._func(argrise)

    def _fall(self, argfall):
        return self._rise(1-argfall)


class GenericFreqRolloff(FreqRolloff):
    def __init__(self, func, rolloff):
        super(GenericFreqRolloff, self).__init__(rolloff)
        self._func = func

    def _rise(self, argrise):
        return self._func(argrise)

    def _fall(self, argfall):
        return self._rise(1-argfall)
