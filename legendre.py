import numpy as np
from scipy.special import legendre
from scipy.integrate import trapz
from cosmosis.datablock import option_section
from names import bao_par_section, bao_data_section


def setup(options):
    max_multipole = options.get_int(option_section, "max_multipole", 4)
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

    block[bao_data_section, "power_spectrum_multipoles"] = pk_multipoles
    block[bao_data_section, "multipoles"] = multipoles

    return 0


