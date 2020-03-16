import numpy as np
from scipy.special import legendre
from scipy.integrate import trapz
from cosmosis.datablock import option_section
from names import bao_par_section, syst_par_section, bao_data_section


def setup(options):
    max_multipole = options.get_int(option_section, "max_multipole", 2)
    multipoles = np.arange(0, max_multipole+2, 2)
    return multipoles


def execute(block, config):
    multipoles = config

    k = block[bao_data_section, "k"]
    mu = block[bao_data_section, "mu"]
    pkmu_bao = block[bao_data_section, "power_spectrum_damped_rescaled"]

    legendre_kernels = np.array([legendre(multipole)(mu) for multipole in multipoles])
    integrands = np.einsum("ik,jk->ijk", legendre_kernels, pkmu_bao)
    pk_multipoles = (2*multipoles.reshape(-1, 1) + 1) * trapz(integrands, x=mu, axis=-1)

    for l, pk_multipole in enumerate(pk_multipoles):
        a0 = block[syst_par_section, "a0_l{}".format(2*l)]
        a1 = block[syst_par_section, "a1_l{}".format(2*l)]
        a2 = block[syst_par_section, "a2_l{}".format(2*l)]
        syst_correction = a0 + a1*k + a2*k**2
        pk_multipole += syst_correction.flatten()
        pk_multipoles[l] = pk_multipole

    block[bao_data_section, "power_spectrum_multipoles"] = pk_multipoles
    block[bao_data_section, "multipoles"] = multipoles

    return 0


