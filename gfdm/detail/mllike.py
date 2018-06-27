from defaultGFDM import get_defaultGFDM
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#

from gfdmutil import get_transmitter_matrix, get_transmitter_pulse
from gfdmutil import get_receiver_matrix, get_receiver_pulse


from mapping import do_map, do_unmap


from wlib.qammodulation import qammod as do_qammodulate

from wlib.qammodulation import qamdemod as do_qamdemodulate

from Modulator import do_modulate

from Demodulator import do_demodulate

from gfdmutil import get_kmset, get_noise_enhancement_factor


def get_random_symbols(p):
    import wlib.transmitter as t
    ms, ks = get_kmset(p)
    return t.rndData(p.mu, len(ms)*len(ks))
