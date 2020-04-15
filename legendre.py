import numpy as np
from scipy.special import legendre
from scipy.integrate import trapz
from cosmosis.datablock import option_section
from names import bao_par_section, syst_par_section, bao_data_section


def setup(options):
    max_multipole = options.get_int(option_section, "max_multipole", 2)
    min_k_power = options.get_int(option_section, "min_k_power", 0)
    max_k_power = options.get_int(option_section, "max_k_power", 2)
    multipoles = np.arange(0, max_multipole+2, 2)

    data = {
            "multipoles": multipoles,
            "min_k_power": min_k_power,
            "max_k_power": max_k_power,
           }
    return data


def execute(block, config):
    multipoles = config["multipoles"]
    min_k_power = config["min_k_power"]
    max_k_power = config["max_k_power"]

    k = block[bao_data_section, "k"]
    mu = block[bao_data_section, "mu"]
    pkmu_bao = block[bao_data_section, "power_spectrum_damped_rescaled"]

    legendre_kernels = np.array([legendre(multipole)(mu) for multipole in multipoles])
    integrands = np.einsum("ik,jk->ijk", legendre_kernels, pkmu_bao)
    pk_multipoles = (2*multipoles.reshape(-1, 1) + 1) * trapz(integrands, x=mu, axis=-1)

    for l, pk_multipole in enumerate(pk_multipoles):
        powers = np.arange(min_k_power, max_k_power+1)
        a_s = [block[syst_par_section, "a{}_l{}".format(power, 2*l)] for power in powers]
        syst_correction = np.zeros_like(k, dtype=float)
        for power, a in zip(powers, a_s):
            syst_correction +=  a * k**power

        pk_multipole += syst_correction.flatten()
        pk_multipoles[l] = pk_multipole

    block[bao_data_section, "power_spectrum_multipoles"] = pk_multipoles
    block[bao_data_section, "multipoles"] = multipoles

    return 0


