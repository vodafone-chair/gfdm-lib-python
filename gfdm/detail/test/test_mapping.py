import unittest
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
import numpy.testing as nt
import numpy as np

from gfdm import get_defaultGFDM
from gfdm import mapping as mapping



class MappingTests(unittest.TestCase):
    def _check(self, p, data, expe):
        M = mapping.Mapper(p)

        nt.assert_array_equal(M.doMap(data), expe)

    def test_fullMapper(self):
        p = get_defaultGFDM('small')
        p.M = 2; p.K = 3; del p.Mon; del p.Kon

        data = np.array([1, 2, 3, 4, 5, 6])
        expe = np.array([[1, 4],
                         [2, 5],
                         [3, 6]])
        self._check(p, data, expe)

    def test_fewSubcarriers(self):
        p = get_defaultGFDM('small')
        p.M = 2; p.K = 3; p.Kon = 2; del p.Mon

        data = np.array([1, 2, 3, 4])
        expe = np.array([[1, 3],
                         [2, 4],
                         [0, 0]])
        self._check(p, data, expe)

    def test_setSubcarriers(self):
        p = get_defaultGFDM('small')
        p.M = 2; p.K = 3; p.Kset = (0, 2); del p.Mon; del p.Kon

        data = np.array([1, 2, 3, 4])
        expe = np.array([[1, 3],
                         [0, 0],
                         [2, 4]])
        self._check(p, data, expe)

    def test_fewSubsymbols(self):
        p = get_defaultGFDM('small')
        p.M = 3; p.K = 2; p.Mon = 2; del p.Kon

        data = np.array([1, 2, 3, 4])
        expe = np.array([[0, 1, 3],
                         [0, 2, 4]])
        self._check(p, data, expe)

    def test_setSubsymbols(self):
        p = get_defaultGFDM('small')
        p.M = 3; p.K = 2; p.Mset = (0, 2); del p.Kon

        data = np.array([1, 2, 3, 4])
        expe = np.array([[1, 0, 3],
                         [2, 0, 4]])
        self._check(p, data, expe)

    def test_fewSubsymbols_subCarriers_usesSingleGuardValues(self):
        p = get_defaultGFDM('small')
        p.empty_symbol_value = 1j
        p.M = 3; p.K = 2; p.Mon = 2; p.Kon = 1;
        data = np.array([1, 2])
        expe = np.array([[1j, 1, 2],
                         [0, 0, 0]])
        self._check(p, data, expe)

    def test_fewSubsymbols_subCarriers_usesVectorGuardValues(self):
        p = get_defaultGFDM('small')
        p.empty_symbol_value = np.array([1j, 2j])
        p.M = 4; p.K = 3; p.Mset = (1, 3); p.Kon = 2;
        data = np.array([1, 2, 3, 4])
        expe = np.array([[1j, 1, 1j, 3],
                         [2j, 2, 2j, 4],
                         [0,0,0, 0]])
        self._check(p, data, expe)


    def test_fullMapper_ignoresSpecificSymbols(self):
        p = get_defaultGFDM('BER')
        p.M = 3; p.K = 2; p.Mon = p.M; p.Kon = p.K;
        p.ignored_data_locations=[(1,1), (0,0)]
        data = np.array([1,2,3,4])

        expe = np.array([[0, 2, 3],
                         [1, 0, 4]])
        self._check(p, data, expe)

    def test_fewSubcarriers_ignoresSpecificSymbols(self):
        p = get_defaultGFDM('BER')
        p.M = 3; p.K = 2; p.Mon = p.M; p.Kset = (1,)
        p.ignored_data_locations=[(1,2), (0,0)]
        data = np.array([1,2])
        expe = np.array([[0,0,0],
                         [1,2,0]])
        self._check(p, data, expe)

    def test_fullMapper_canKeepIgnoredSymbols(self):
        p = get_defaultGFDM('BER')
        p.M = 3; p.K = 2; p.Mon = p.M; p.Kon = p.K
        p.ignored_data_locations = [(0,0), (1,2)]

        data = np.array([99, 1, 2, 3, 4, 100])
        expe = np.array([[99, 2, 4],
                         [1,3,100]])
        M = mapping.Mapper(p)
        M.keepIgnoredSymbols = True
        nt.assert_array_equal(M.doMap(data), expe)



class DemapperTests(unittest.TestCase):
    def _check(self, p, data, expe):
        M = mapping.Demapper(p)
        nt.assert_array_equal(M.doDemap(data), expe)

    def test_fullMapping(self):
        p = get_defaultGFDM('small')
        p.M = 2; p.K = 3; del p.Mon; del p.Kon

        data = np.array([[1, 4],
                         [2, 5],
                         [3, 6]])
        expe = np.array([1, 2, 3, 4, 5, 6])
        self._check(p, data, expe)

    def test_fewSubcarriers(self):
        p = get_defaultGFDM('small')
        p.M = 2; p.K = 3; p.Kon = 2; del p.Mon

        expe = np.array([1, 2, 3, 4])
        data = np.array([[1, 3],
                         [2, 4],
                         [0, 0]])
        self._check(p, data, expe)

    def test_setSubcarriers(self):
        p = get_defaultGFDM('small')
        p.M = 2; p.K = 3; p.Kset = (0, 2); del p.Mon; del p.Kon

        expe = np.array([1, 2, 3, 4])
        data = np.array([[1, 3],
                         [0, 0],
                         [2, 4]])
        self._check(p, data, expe)

    def test_fewSubsymbols(self):
        p = get_defaultGFDM('small')
        p.M = 3; p.K = 2; p.Mon = 2; del p.Kon

        expe = np.array([1, 2, 3, 4])
        data = np.array([[0, 1, 3],
                         [0, 2, 4]])
        self._check(p, data, expe)

    def test_setSubsymbols(self):
        p = get_defaultGFDM('small')
        p.M = 3; p.K = 2; p.Mset = (0, 2); del p.Kon

        expe = np.array([1, 2, 3, 4])
        data = np.array([[1, 0, 3],
                         [2, 0, 4]])
        self._check(p, data, expe)

    def test_fullMapper_ignoresSpecificSymbols(self):
        p = get_defaultGFDM('small')
        p.M = 3; p.K = 2; p.Mon = p.M; p.Kon = p.K
        p.ignored_data_locations = [(0,0), (1,1)]

        data = np.array([[1,2,3],
                         [4,5,6]])
        expe = np.array([4,2,3,6])

        self._check(p, data, expe)

    def test_fewSubsymbols_ignoredSpecificSymbols(self):
        p = get_defaultGFDM('BER')
        p.M = 3; p.K = 2; p.Mset = (0,2); p.Kon = p.K
        p.ignored_data_locations = [(0,0), (1,1)]
        data = np.array([[1,2,3],
                         [4,5,6]])
        expe = np.array([4,3,6])

        self._check(p, data, expe)

    def test_fullMapper_canKeepIgnoredSymbols(self):
        p = get_defaultGFDM('BER')
        p.M = 3; p.K = 2; p.Mon = p.M; p.Kon = p.K;
        p.ignored_data_locations = [(0,0), (1,1)]
        data = np.array([[1,2,3],
                         [4,5,6]])
        expe = np.array([1,4,2,5,3,6])

        M = mapping.Demapper(p)
        M.keepIgnoredSymbols = True
        nt.assert_array_equal(M.doDemap(data), expe)



if __name__ == '__main__':
    unittest.main()
