import unittest
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
import numpy.testing as nt
import numpy as np

import gfdm
from gfdm import gfdmutil
from gfdm import DefaultModulator
from gfdm import DefaultDemodulator

from gfdm.detail.mapping import do_map
from gfdm.detail.Modulator import do_modulate

class ReceiverPulseTests(unittest.TestCase):
    def test_mmse(self):
        p = gfdm.get_defaultGFDM('small')
        p.M = 4
        p.Mon = 4
        A = gfdmutil.get_transmitter_matrix(p)
        g = gfdmutil.get_transmitter_pulse(p)

        snr = -3
        sigma = 10**(-snr/10.)

        Bm = np.linalg.inv(A.T.conj().dot(A)+sigma*np.eye(A.shape[0])).dot(A.T.conj())
        Bm0 = Bm[0, :]
        Bm0 /= sum(g * Bm0)
        nt.assert_array_almost_equal(Bm0, gfdmutil.get_receiver_pulse(p, 'MMSE', snr))

class TransmitterMatrixTests(unittest.TestCase):
    def test_correctSize(self):
        p = gfdm.get_defaultGFDM('small')
        A = gfdmutil.get_transmitter_matrix(p)
        self.assertEqual(A.shape, (p.M*p.K, p.M*p.K))

    def test_ColumnsCorrect(self):
        p = gfdm.get_defaultGFDM('small')
        A = gfdmutil.get_transmitter_matrix(p)

        g00 = gfdmutil.get_transmitter_pulse(p)
        g10 = g00 * np.exp(2j*np.pi/p.K*np.arange(p.K*p.M))
        g01 = np.roll(g00, p.K)

        nt.assert_array_almost_equal(A[:, 0], g00)
        nt.assert_array_almost_equal(A[:, 1], g10)
        nt.assert_array_almost_equal(A[:, p.K], g01)

    def test_truncatedMatrix(self):
        p = gfdm.get_defaultGFDM('small')
        p.K = 4; p.Kon = 3;
        p.M = 3; p.Mon = 2;

        A = gfdmutil.get_transmitter_matrix(p)
        At = gfdmutil.get_truncated_transmitter_matrix(p)

        Aex = A[:, (4,5,6, 8,9,10)]
        nt.assert_array_equal(Aex, At)


class ReceiverMatrixTests(unittest.TestCase):
    def test_matchedFilter(self):
        p = gfdm.get_defaultGFDM('small')
        A = gfdmutil.get_transmitter_matrix(p)

        nt.assert_array_almost_equal(A.T.conj(), gfdmutil.get_receiver_matrix(p, 'MF'))

    def test_ZeroForcing(self):
        p = gfdm.get_defaultGFDM('small')
        A = gfdmutil.get_transmitter_matrix(p)
        B = gfdmutil.get_receiver_matrix(p, 'ZF')

        nt.assert_array_almost_equal(np.dot(B, A), np.eye(A.shape[0]))


class AmbgfunTests(unittest.TestCase):
    def test_ambgfun(self):
        p = gfdm.get_defaultGFDM('small')
        am = gfdmutil.get_ambgfun(p)

        A = gfdmutil.get_transmitter_matrix(p)
        g00 = A[:, 0]
        g01 = A[:, p.K]
        g10 = A[:, 1]

        self.assertEqual(am.shape, (p.K, p.M))
        self.assertAlmostEqual(am[0, 0], 1)
        self.assertAlmostEqual(am[1, 0], sum(g00 * g10))
        self.assertAlmostEqual(am[0, 1], sum(g00 * g01))


class NoiseEnhancementTests(unittest.TestCase):
    def test_NoiseEnhancementDirichletIsOne(self):
        p = gfdm.get_defaultGFDM('small')
        p.pulse = 'dirichlet'
        self.assertAlmostEqual(1, gfdmutil.get_noise_enhancement_factor(p))

    def test_NoiseEnhancementGreaterRolloffGreaterEnhancement(self):
        p = gfdm.get_defaultGFDM('small')

        p.a = 0.1; e1 = gfdmutil.get_noise_enhancement_factor(p)
        p.a = 0.5; e2 = gfdmutil.get_noise_enhancement_factor(p)
        self.assertGreater(e2, e1)


class DefaultModulatorTests(unittest.TestCase):
    def test_small(self):
        p = gfdm.get_defaultGFDM('small')
        p.L = p.K
        D = np.arange(p.K*p.M).reshape(p.K, p.M, order='F')

        x1 = DefaultModulator(p).modulate(D)
        x2 = np.dot(gfdmutil.get_transmitter_matrix(p), D.reshape(p.K*p.M, order='F'))

        nt.assert_array_almost_equal(x1, x2)

    def test_FrequencyDomainAndTimeDomainImplmementationsEqual(self):
        p = gfdm.get_defaultGFDM('BER')
        p.L = p.K
        D = np.arange(p.K*p.M).reshape(p.K, p.M, order='F')

        x1 = DefaultModulator(p)._modulate_freq(D)
        x2 = DefaultModulator(p)._modulate_time(D)

        nt.assert_array_almost_equal(x1, x2)


class DemodulatorTests(unittest.TestCase):
    def test_small_MF(self):
        p = gfdm.get_defaultGFDM('small')
        p.pulse = 'dirichlet'

        D = do_map(p, np.arange(p.K*p.M))
        x = do_modulate(p, D)

        Dh = DefaultDemodulator(p, 'MF').demodulate(x)
        nt.assert_array_almost_equal(D, Dh)

    def test_BER_ZF(self):
        p = gfdm.get_defaultGFDM('small')
        D = do_map(p, np.arange(p.K*p.M))
        x = do_modulate(p, D)

        Dh = DefaultDemodulator(p, 'ZF').demodulate(x)
        nt.assert_array_almost_equal(D, Dh)

    def test_BER_FFFFT(self):
        p = gfdm.get_defaultGFDM('small')
        D = do_map(p, np.arange(p.K*p.M))
        x = do_modulate(p, D)

        Dh = DefaultDemodulator(p, 'ZFFFT').demodulate(x)
        nt.assert_array_almost_equal(D, Dh)

    def test_customReceiverFilter(self):
        p = gfdm.get_defaultGFDM('small')
        p.pulse = 'rc_fd'

        D = do_map(p, np.arange(p.K*p.M))
        x = do_modulate(p, D)

        g = gfdmutil.get_receiver_pulse(p, 'ZF')
        Dm = DefaultDemodulator(p, 'custom')
        Dm.setFilter(pulse=g, L=16)

        Dh = Dm.demodulate(x)
        nt.assert_array_almost_equal(D, Dh)

    def test_withKset(self):
        p = gfdm.get_defaultGFDM('BER')
        p.M = 5
        del p.Mon
        p.K = 6
        p.Kset = (0, 1, 3, 5)
        p.L = p.K

        D = do_map(p, np.arange(p.M*len(p.Kset)))
        x = do_modulate(p, D)

        Dh = DefaultDemodulator(p, 'ZFFFT').demodulate(x)
        nt.assert_array_almost_equal(D, Dh)


class UtilTests(unittest.TestCase):
    def test_shiftedPulses_withoutCP(self):
        p = gfdm.get_defaultGFDM('BER')
        p.Mset = (0, 2, 4)
        g = gfdmutil.get_transmitter_pulse(p)

        pi = gfdmutil.get_shifted_pulses(p, withCP=False)

        self.assertEqual((len(g), 3), pi.shape)
        nt.assert_array_equal(np.roll(g, 2*p.K), pi[:, 1])

    def test_shiftedPulses_withCP(self):
        p = gfdm.get_defaultGFDM('BER')
        p.Mset = (0,)
        CP = 16
        p.NCP = CP

        g = gfdmutil.get_transmitter_pulse(p)
        gi = gfdmutil.get_shifted_pulses(p, withCP=True)

        self.assertEqual((len(g)+CP, 1), gi.shape)
        nt.assert_array_equal(g, gi[CP:, 0])
        nt.assert_array_equal(g[-CP:], gi[:CP, 0])

    def test_do_addcp_oneDim(self):
        p = gfdm.get_defaultGFDM('BER')
        sig = np.arange(p.K*p.M)
        p.NCP = 3
        expRes = np.hstack([sig[-p.NCP:], sig])
        nt.assert_array_equal(gfdmutil.do_addcp(p, sig), expRes)

    def test_do_addcp_twoDim(self):
        p = gfdm.get_defaultGFDM('BER')
        sig = np.zeros((p.K*p.M, 2))
        sig[:, 0] = np.arange(p.K*p.M)
        sig[:, 1] = np.arange(p.K*p.M) + 1
        p.NCP = 3
        expRes = np.vstack([sig[-p.NCP:,:], sig])
        nt.assert_array_equal(gfdmutil.do_addcp(p, sig),
                              expRes)


class BlockWindowingTests(unittest.TestCase):
    def test_get_block_window(self):
        p = gfdm.get_defaultGFDM('BER')
        p.b = p.K
        p.window = 'ramp'

        w = gfdmutil.get_block_window(p)
        nt.assert_array_equal(w[p.K:-p.K], np.ones(p.K*(p.M-2)))
        nt.assert_array_equal(w[:p.K], np.linspace(0, 1, p.K, endpoint=False))
        nt.assert_array_equal(w[-p.K:], np.linspace(1, 0, p.K, endpoint=False))

class CopyAndChangeTests(unittest.TestCase):
    class Params(object):
        pass

    def setUp(self):
        self.original = self.Params()
        self.original.integer = 1
        self.original.string = 'abc'

    def test_copyIsEqual(self):
        c = gfdmutil.copyAndChange(self.original)
        self.assertEqual(c.__dict__, self.original.__dict__)

    def test_changesActually(self):
        self.assertEqual(self.original.integer, 1)
        c = gfdmutil.copyAndChange(self.original, integer=2)
        self.assertEqual(c.integer, 2)
        self.assertEqual(self.original.integer, 1)
        self.assertNotEqual(c.__dict__, self.original.__dict__)



if __name__ == '__main__':
    unittest.main()
