import numpy as np
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
from gfdmutil import get_kset, get_mset, get_kmset


class MappingBase(object):
    def __init__(self, pnew):
        self._pnew = pnew
        self.keepIgnoredSymbols = False
        try:
            V = pnew.empty_symbol_value
            k = get_kset(pnew)
            if type(V) is np.ndarray:
                self._empty_symbol_value = V
            else:
                self._empty_symbol_value = V * np.ones(len(k))

            assert len(self._empty_symbol_value) == len(k), "Correct number of guard-symbol-values required"
            self._empty_symbol_value = self._empty_symbol_value.reshape((len(k), 1))

        except AttributeError:
            self._empty_symbol_value = 0


    def _hasIgnoredLocations(self):
        return hasattr(self._pnew, 'ignored_data_locations') and len(self._pnew.ignored_data_locations) > 0

    def _getIgnoredLocations(self):
        fullData = np.zeros((self._pnew.K, self._pnew.M))
        for ip, p in enumerate(self._pnew.ignored_data_locations):
            fullData[p[0], p[1]] = 1

        kset, mset = get_kmset(self._pnew)
        r1 = fullData[kset, :]
        r2 = r1[:, mset]
        return np.flatnonzero(r2.flatten(order='F'))


class Mapper(MappingBase):
    def doMap(self, data):
        res = np.zeros((self._pnew.K, self._pnew.M), dtype=complex)
        if not self.keepIgnoredSymbols and self._hasIgnoredLocations():
            ignored = self._getIgnoredLocations()
            data = self._insertZeroForIgnored(data, ignored)
        k = tuple(get_kset(self._pnew))
        m = tuple(get_mset(self._pnew))
        Dm = data.reshape(len(k), len(m), order='F')
        if len(k) == self._pnew.K and len(m) == self._pnew.M:
            return Dm

        # First set the subcarriers
        res1 = np.ones((len(k), self._pnew.M), dtype=complex) * self._empty_symbol_value
        res1[:, m] = Dm
        # Now copy the elements to the correct subsymbol slots
        res[k, :] = res1

        return res

    def _insertZeroForIgnored(self, data, ignored):
        data = data.copy()
        for p in ignored:
            data = np.insert(data, p, 0)
        return data


class Demapper(MappingBase):
    def doDemap(self, data):
        k = tuple(get_kset(self._pnew))
        m = tuple(get_mset(self._pnew))

        r1 = data[k, :]
        r2 = r1[:, m]

        data = r2.flatten(1)
        if not self.keepIgnoredSymbols and self._hasIgnoredLocations():
            res = np.delete(data, self._getIgnoredLocations())
        else:
            res = data
        return res


def do_map(p, d):
    return Mapper(p).doMap(d)


def do_unmap(p, D):
    return Demapper(p).doDemap(D)
