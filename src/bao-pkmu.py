import sys
import numpy as np
from scipy.interpolate import interp1d
from cosmosis.datablock import option_section
from names import bao_par_section, bao_data_section, template_data_section


def setup(options):
    kmin = options.get_double(option_section, "kmin", 1e-5)
    kmax = options.get_double(option_section, "kmax", 1e0)
    nk = options.get_int(option_section, "nk", 1000)
    nmu = options.get_int(option_section, "nmu", 1000)

    # k vary along columns, and mu along lines
    k = np.linspace(kmin, kmax, nk).reshape(-1, 1)
    mu = np.linspace(0., 1., nmu)

    loaded_data = dict()
    loaded_data["k"] = k
    loaded_data["mu"] = mu

    return loaded_data


def execute(block, config):
    k = config["k"]
    mu = config["mu"]

    k_nw = block[template_data_section, "k_nw"]
    pk_nw = block[template_data_section, "pk_nw"]
    k_bao = block[template_data_section, "k_bao"]
    pk_bao = block[template_data_section, "pk_bao"]

    pk_no_wiggle_interp = interp1d(k_nw, pk_nw)
    pk_bao_interp = interp1d(k_bao, pk_bao)

    if (k_nw.min() > k.min()) or (k_nw.max() < k.max()):
        print("WARNING: The range of the no wiggle template don't cover the k range "
              "you specified.", file=sys.stderr)
        print("k no wiggle range = {}, {}".format(k_nw.min(), k_nw.max()),
              file=sys.stderr)
    if (k_bao.min() > k.min()) or (k_bao.max() < k.max()):
        print("WARNING: The range of the linear template don't cover the k range "
              "you specified.", file=sys.stderr)
        print("k linear range = {}, {}".format(k_bao.min(), k_bao.max()),
              file=sys.stderr)

    sigma_fog = block[bao_par_section, "sigma_fog"]
    sigma_rec = block[bao_par_section, "sigma_rec"]
    sigma_par = block[bao_par_section, "sigma_par"]
    sigma_perp = block[bao_par_section, "sigma_perp"]
    bias = block[bao_par_section, "bias"]
    alpha_par = block[bao_par_section, "alpha_par"]
    alpha_perp = block[bao_par_section, "alpha_perp"]
    f = block[bao_par_section, "growth_factor"]

    kp, mup = scale_k_and_mu(k, mu, alpha_par, alpha_perp)

    beta = f / bias

    K_rsd = kaiser_rsd_prefactor(k, mu, beta, sigma_rec)
    F = fog_correction_prefactor(k, mu, sigma_fog)
    NL = bao_broadening(k, mu, sigma_par, sigma_perp)

    pk_bao = pk_bao_interp(kp)
    pk_nw = pk_no_wiggle_interp(kp)
    pk_bao_rescaled = bias**2 * K_rsd * F * (pk_nw + (pk_bao - pk_nw) * NL)

    block[bao_data_section, "power_spectrum_damped_rescaled"] = pk_bao_rescaled
    block[bao_data_section, "power_spectrum_template"] = pk_bao
    block[bao_data_section, "power_spectrum_template_no_wiggle"] = pk_nw
    block[bao_data_section, "k"] = k
    block[bao_data_section, "mu"] = mu
    block[bao_data_section, "k_rescaled"] = kp.flatten()
    block[bao_data_section, "mu_rescaled"] = mup

    return 0


def scale_k_and_mu(k, mu, alpha_par, alpha_perp):
    """
    Computes Alcock-Paczynski scaling of k and mu (cf. Beutler 2014 eqs. 58,59).
    """
    F = alpha_par / alpha_perp
    kp = k / alpha_perp * np.sqrt(1 + mu**2 * (1/F**2 - 1))
    mup = mu / F / np.sqrt(1 + mu**2 * (1/F**2 - 1))
    return kp, mup


def kaiser_rsd_prefactor(k, mu, beta, sigma_rec):
    """
    Kaiser RSD prefactor with optional reconstruction correction from Seo 16'.
    """
    if sigma_rec == 0:
        Sk = np.zeros_like(k)
    else:
        Sk = np.exp(-1./2 * k**2 * sigma_rec**2)
    K_rsd = (1 + beta * mu**2 * (1 - Sk))**2
    return K_rsd


def fog_correction_prefactor(k, mu, sigma_fog):
    """
    FOG correction prefactor (cf Xu et. al. 16' eq. 13).
    """
    F = 1. / (1 + k**2 * mu**2 * sigma_fog**2)**2
    return F


def bao_broadening(k, mu, sigma_par, sigma_perp):
    """
    BAO broadening correction
    """
    NL = np.exp(-1./2 * (k**2 * mu**2 * sigma_par**2 + k**2 * (1 - mu**2) * sigma_perp**2))
    return NL


