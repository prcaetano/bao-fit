import sys
import numpy as np
from scipy.interpolate import interp1d
from cosmosis.gaussian_likelihood import GaussianLikelihood
from cosmosis.datablock import option_section
from names import bao_par_section, bao_data_section


class BAOLikelihood(GaussianLikelihood):
    x_section = bao_data_section
    x_name    = "k"
    y_section = bao_data_section
    y_name    = "power_spectrum_multipoles"
    like_name = "bao"


    def extract_theory_points(self, block):
        k = block[bao_data_section, self.x_name].flatten()
        pk_multipoles = block[bao_data_section, self.y_name]
        if pk_multipoles.shape[0] < self.n_data_multipoles:
            print("WARNING: The number of computed multipoles is smaller than the the ones present in "
                  "the data file. I'm restricting the fit to the multipoles you told me to calculate.",
                  file=sys.stderr)
            max_idx = self.n_k * pk_multipoles.shape[0]
            self.data_y = self.data_y[:max_idx]
            self.data_x = self.data_x[:max_idx]
            self.cov = self.cov[:max_idx,:max_idx]
            self.inv_cov = self.build_inverse_covariance()
            if not self.likelihood_only:
                self.chol = np.linalg.cholesky(self.cov)
        else:
            pk_multipoles = pk_multipoles[:self.n_data_multipoles]

        try:
            pk_multipoles_interp = np.array([interp1d(k, pk_multipole)(self.k)
                                             for pk_multipole in pk_multipoles])
        except ValueError as e:
            if "A value in x_new is " in str(e):
                raise RuntimeError("The range of k used is too restrict to cover the observed k range.")
            else:
                raise e
        return pk_multipoles_interp.flatten()


    def build_covariance(self):
        covdata_fname = self.options.get_string("cov_fname")
        self.cov = np.loadtxt(covdata_fname)
        mask = np.repeat(np.atleast_2d(self.mask_k), self.n_data_multipoles, axis=0).flatten()
        self.cov = self.cov[mask][:,mask]
        return self.cov


    def build_inverse_covariance(self):
        nsamples_cov = self.options.get_int("nsamples_covariance", 0)
        if nsamples_cov == 0:
            hartlap_correction = 1.
        else:
            hartlap_correction = (nsamples_cov - self.cov.shape[0] - 2) / (nsamples_cov - 1)
            if hartlap_correction < 0:
                print("ERROR: Too small number of samples used for covariance calculation "
                      " The hartlap factor is undefined. Aborting.", file=sys.stderr)
                raise RuntimeError
        return hartlap_correction * np.linalg.inv(self.cov)


    def build_data(self):
        pkdata_fname = self.options.get_string("pk_fname")
        kmin = self.options.get_double("kmin", 0.)
        kmax = self.options.get_double("kmax", np.inf)

        pkdata = np.loadtxt(pkdata_fname)
        k, pkdata = pkdata[:,0], pkdata[:,1:]
        self.mask_k = (k < kmax) * (k > kmin)
        k, pkdata = k[self.mask_k], pkdata[self.mask_k]

        self.n_k, self.n_data_multipoles = pkdata.shape
        self.k = k
        pkdata = pkdata.T.flatten()
        return np.arange(len(pkdata)), pkdata


setup, execute, cleanup = BAOLikelihood.build_module()

