# CosmoSIS module for anisotropic BAO fitting

This module implements the Beutler et. al. 17' model for inference of the BAO scale. It includes template generation (using CAMB and E&H together with spline fitting), P(k, mu) computation, multipole projection and the gaussian likelihood computation, which can be sampled using any sampler available within CosmoSIS. Broadband marginalization is included during the multipole calculation, independently for each multipole, and the user can choose the minimum and maximum power of the polynomial to use.

## Requirements
You need to have installed CosmoSIS 1.6 (instructions can be found at https://bitbucket.org/joezuntz/cosmosis/wiki/Home), and also numpy 1.18 and scipy 1.2.

## Use
The directory config under this repository contains examples of configuration files.
