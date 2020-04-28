import sys
import numpy as np
from cosmosis.datablock import option_section
from names import template_data_section


def setup(options):
    no_wiggle_template_fname = options.get_string(option_section, "no_wiggle_template_fname", "")
    bao_template_fname = options.get_string(option_section, "bao_template_fname", "")
    output_no_wiggle_template_fname = options.get_string(option_section,
                                                         "output_no_wiggle_template_fname", "")
    output_bao_template_fname = options.get_string(option_section,
                                                   "output_bao_template_fname", "")

    if bao_template_fname != "":
        k_bao, pk_bao = np.loadtxt(bao_template_fname, unpack=True)
    else:
        print("No bao_template_fname passed. I'm assuming some module before me computed "
              "the linear power spectrum and placed it at matter_power_lin section.",
              file=sys.stderr)
        raise NotImplemented

    if no_wiggle_template_fname != "":
        k_no_wiggle, pk_no_wiggle = np.loadtxt(no_wiggle_template_fname, unpack=True)
    else:
        print("No no_wiggle_template_fname passed. I'll compute an Eisenstein&Hu "
              "template at the given cosmology, with an polynomial broadband "
              "term fitted together.", file=sys.stderr)
        pmin = options.get_int(option_section, "min_power_broadband_polynomial", 0)
        pmax = options.get_int(option_section, "max_power_broadband_polynomial", 0)
        raise NotImplemented

    if output_no_wiggle_template_fname != "":
        np.savetxt(output_no_wiggle_template_fname, np.c_[k_no_wiggle, pk_no_wiggle])
    if output_bao_template_fname != "":
        np.savetxt(output_bao_template_fname, np.c_[k_bao, pk_bao])

    loaded_data = dict()
    loaded_data["k_bao"] = k_bao
    loaded_data["pk_bao"] = pk_bao
    loaded_data["k_nw"] = k_no_wiggle
    loaded_data["pk_nw"] = pk_no_wiggle
    return loaded_data


def execute(block, config):
    block[template_data_section, "k_nw"] = config["k_nw"]
    block[template_data_section, "pk_nw"] = config["pk_nw"]
    block[template_data_section, "k_bao"] = config["k_bao"]
    block[template_data_section, "pk_bao"] = config["pk_bao"]

    return 0

