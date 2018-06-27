class GFDMParams(object):
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
    pass


def get_defaultGFDM(gfdmType):
    r = GFDMParams()
    r.NCP = 0
    r.mu = 2
    r.L = 4
    r.pulse = 'rrc'
    r.a = 0.5
    r.b = 0

    if gfdmType == 'BER':
        r.M = 5
        r.K = 128
        r.Kon = 128
    elif gfdmType == 'small':
        r.M = 3
        r.K = 4
        r.Kon = 4
    elif gfdmType == 'OFDM':
        r.M = 1
        r.K = 128
        r.Kon = 128
        r.pulse = 'rect'
        r.L = r.K
    elif gfdmType == 'TCOM':
        r.M = 7
        r.K = 64
        r.Kon = r.K/2
        r.NCP = 16
        r.pulse = 'rc_fd'
        r.a = 0.1
        r.mu = 4
    else:
        raise NotImplementedError("'%s' gfdm parameter set unknown" % gfdmType)

    r.Mon = r.M
    r.N = r.M * r.K
    return r
