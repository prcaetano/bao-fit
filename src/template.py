import sys
import numpy as np
from cosmosis.datablock import option_section, names
from names import template_data_section


def setup(options):
    no_wiggle_template_fname = options.get_string(option_section, "no_wiggle_template_fname", "")
    bao_template_fname = options.get_string(option_section, "bao_template_fname", "")
    output_no_wiggle_template_fname = options.get_string(option_section,
                                                         "output_no_wiggle_template_fname", "")
    output_bao_template_fname = options.get_string(option_section,
                                                   "output_bao_template_fname", "")


    loaded_bao_template = bao_template_fname != ""
    if loaded_bao_template:
        k_bao, pk_bao = np.loadtxt(bao_template_fname, unpack=True)
    else:
        print("No bao_template_fname passed. I'm assuming some module before me computed "
              "the linear power spectrum and placed it at matter_power_lin section.",
              file=sys.stderr)
        z = options.get_double(option_section, "z", 0.)


    loaded_no_wiggle_template = no_wiggle_template_fname != ""
    if loaded_no_wiggle_template:
        k_no_wiggle, pk_no_wiggle = np.loadtxt(no_wiggle_template_fname, unpack=True)
    else:
        print("No no_wiggle_template_fname passed. I'll compute an Eisenstein&Hu "
              "template at the given cosmology, with an polynomial broadband "
              "term fitted together.", file=sys.stderr)
        use_eh_sound_horizon = options.get_bool(option_section, "use_eh_sound_horizon", True)
        use_eh_k_eq = options.get_bool(option_section, "use_eh_k_eq", True)
        pmin = options.get_int(option_section, "min_power_broadband_polynomial", 0)
        pmax = options.get_int(option_section, "max_power_broadband_polynomial", 0)

    # Save variables to be used during execute
    vars_to_save = ["loaded_bao_template", "loaded_no_wiggle_template", "k_bao", "pk_bao", "z",
                    "output_bao_template_fname", "k_nw", "pk_nw",
                    "output_no_wiggle_template_fname", "pmin", "pmax",
                    "use_eh_sound_horizon", "use_eh_k_eq"]
    loaded_data = dict()
    for key in vars_to_save:
        if key in locals():
            loaded_data[key] = locals()[key]

    return loaded_data


def execute(block, config):

    # Loading variables to local scope
    if "loaded_bao_template" in config:
        loaded_bao_template = config["loaded_bao_template"]
    if "loaded_no_wiggle_template" in config:
        loaded_no_wiggle_template = config["loaded_no_wiggle_template"]
    if "k_bao" in config:
        k_bao = config["k_bao"]
    if "pk_bao" in config:
        pk_bao = config["pk_bao"]
    if "z" in config:
        z = config["z"]
    if "output_bao_template_fname" in config:
        output_bao_template_fname = config["output_bao_template_fname"]
    if "k_nw" in config:
        k_nw = config["k_nw"]
    if "pk_nw" in config:
        pk_nw = config["pk_nw"]
    if "output_no_wiggle_template_fname" in config:
        output_no_wiggle_template_fname = config["output_no_wiggle_template_fname"]
    if "pmin" in config:
        pmin = config["pmin"]
    if "pmax" in config:
        pmax = config["pmax"]
    if "use_eh_sound_horizon" in config:
        use_eh_sound_horizon = config["use_eh_sound_horizon"]
    if "use_eh_k_eq" in config:
        use_eh_k_eq = config["use_eh_k_eq"]


    if not loaded_bao_template:
        k_bao = block[names.matter_power_lin, "k_h"]
        pk_bao = block[names.matter_power_lin, "p_k"]
        z_pk = block[names.matter_power_lin, "z"]
        mask = np.isclose(z_pk, z)
        if mask.sum() != 1:
            print("WARNING: The linear power spectrum was not computed for the value of "
                  "z you set. Proceeding instead with z = {}".format(z_pk[0]),
                  file=sys.stderr)
            z = z_pk[0]
            mask = np.isclose(z_pk, z)
        pk_bao = pk_bao[mask].flatten()

    block[template_data_section, "k_bao"] = k_bao
    block[template_data_section, "pk_bao"] = pk_bao
    if output_bao_template_fname != "":
        np.savetxt(output_bao_template_fname, np.c_[k_bao, pk_bao])


    if not loaded_no_wiggle_template:
        k_nw = k_bao
        if use_eh_sound_horizon:
            s = None
        else:
            print("Using sound horizon computed by boltzmann solver.", file=sys.stderr)
            s = block[names.distances, "RS_ZDRAG"]
        if use_eh_k_eq:
            k_eq = None
        else:
            print("Using k_eq computed by boltzmann solver.", file=sys.stderr)
            k_eq = block[names.distances, "K_EQUALITY"]
        pk_nw = eisenstein_hu(k_nw, h=block[names.cosmological_parameters, "h0"],
                              Om=block[names.cosmological_parameters, "omega_m"],
                              Ob=block[names.cosmological_parameters, "omega_b"],
                              ns=block[names.cosmological_parameters, "n_s"],
                              s=s, k_eq=k_eq)
        # Normalizing pk_nw scale to pk_bao minimizing the residue (bao)
        #delta_k = np.ones_like(k_nw)
        #delta_k[:-1] = k_nw[1:] - k_nw[:-1]
        #delta_k[-1] = delta_k[-2]
        #pk_nw = pk_nw * (pk_nw * pk_bao * delta_k).sum() / (pk_nw * pk_nw * delta_k).sum()
        # Normalizing pk_nw scale to pk_bao at large scales
        pk_nw = pk_nw * pk_bao[0] / pk_nw[0]

    block[template_data_section, "k_nw"] = k_nw
    block[template_data_section, "pk_nw"] = pk_nw
    if output_no_wiggle_template_fname != "":
        np.savetxt(output_no_wiggle_template_fname, np.c_[k_nw, pk_nw])

    return 0


def eisenstein_hu(k, h, Om, Ob, ns, s=None, k_eq=None, Tcmb0=2.7255):
    """
    Computes Eisenstein and Hu no-wiggle power spectrum.

    Parameters:
        k (np.ndarray): wavenumbers, in Mpc/h
        h (float): Hubble constant (in km/s/Mpc / (100 km/s))
        Om (float): Omega_matter today
        Ob (float): Omega_baryons today
        ns (float): primordial power spectrum spectral index
        s (float or None): sound horizon at z_drag, in Mpc (if None uses E&H formula)
        k_eq (float or None): k of matter-radiation equality, in Mpc
                              (if None uses E&H formula)
        Tcmb0 (float): CMB temperature today (defaults to 2.7255)

    Returns:
        P (np.ndarray): Eisenstein and Hu power spectrum
        (normalized to 1 at the lowest k asked)
    """
    ## Code based on nbodykit 0.3 implementation of the formulas on the classic E&H 1998 paper
    Obh2      = Ob * h ** 2
    Omh2      = Om * h ** 2
    f_baryon  = Ob / Om
    theta_cmb = Tcmb0 / 2.7

    k = np.sort(k)

    # wavenumber of equality
    if k_eq is None:
        k_eq = 0.0746 * Omh2 * theta_cmb ** (-2) # units of 1/Mpc

    if s is None:
        sound_horizon = h * 44.5 * np.log(9.83/Omh2) / np.sqrt(1 + 10 * Obh2** 0.75) # in Mpc/h
    else:
        sound_horizon = s * h
    alpha_gamma = 1 - 0.328 * np.log(431*Omh2) * f_baryon + 0.38 * np.log(22.3*Omh2) * f_baryon ** 2

    k = k * h
    ks = k * sound_horizon / h
    q = k / (13.41*k_eq)

    gamma_eff = Omh2 * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
    q_eff = q * Omh2 / gamma_eff
    L0 = np.log(2*np.e + 1.8 * q_eff)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)

    T = L0 / (L0 + C0 * q_eff**2)

    k = k/h # k -> h/Mpc
    P = k**ns * T**2
    P = P/P[0]
    return P


