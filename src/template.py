import sys
import numpy as np
from cosmosis.datablock import option_section
from names import template_data_section

matter_power_lin = "matter_power_lin"


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
        pmin = options.get_int(option_section, "min_power_broadband_polynomial", 0)
        pmax = options.get_int(option_section, "max_power_broadband_polynomial", 0)
        raise NotImplemented

    if loaded_no_wiggle_template and output_no_wiggle_template_fname != "":
        np.savetxt(output_no_wiggle_template_fname, np.c_[k_no_wiggle, pk_no_wiggle])
    if loaded_bao_template and output_bao_template_fname != "":
        np.savetxt(output_bao_template_fname, np.c_[k_bao, pk_bao])

    loaded_data = dict()
    loaded_data["loaded_bao_template"] = loaded_bao_template
    loaded_data["loaded_no_wiggle_template"] = loaded_no_wiggle_template
    if loaded_bao_template:
        loaded_data["k_bao"] = k_bao
        loaded_data["pk_bao"] = pk_bao
    else:
        loaded_data["output_bao_template_fname"] = output_bao_template_fname
        loaded_data["z"] = z
    if loaded_no_wiggle_template:
        loaded_data["k_nw"] = k_no_wiggle
        loaded_data["pk_nw"] = pk_no_wiggle
    else:
        loaded_data["pmin"] = pmin
        loaded_data["pmax"] = pmax
    return loaded_data


def execute(block, config):
    if config["loaded_bao_template"]:
        block[template_data_section, "k_bao"] = config["k_bao"]
        block[template_data_section, "pk_bao"] = config["pk_bao"]
    else:
        k_h = block[matter_power_lin, "k_h"]
        pk_lin = block[matter_power_lin, "p_k"]
        z_pk = block[matter_power_lin, "z"]
        mask = np.isclose(z_pk, config["z"])
        if mask.sum() != 1:
            print("WARNING: The linear power spectrum was not computed for the value of "
                  "z you set. Proceeding instead with z = {}".format(z_pk[0]),
                  file=sys.stderr)
            z = z_pk[0]
            mask = np.isclose(z_pk, z)
        pk_lin = pk_lin[mask].flatten()
        block[template_data_section, "k_bao"] = k_h
        block[template_data_section, "pk_bao"] = pk_lin

        if config["output_bao_template_fname"] != "":
            np.savetxt(config["output_bao_template_fname"], np.c_[k_h, pk_lin])

    block[template_data_section, "k_nw"] = config["k_nw"]
    block[template_data_section, "pk_nw"] = config["pk_nw"]
    return 0

