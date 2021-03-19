# coding: utf-8
"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Function definitions for variogram models. In each function, m is a list of
defining parameters and d is an array of the distance values at which to
calculate the variogram model.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.

Copyright (c) 2015-2020, PyKrige Developers
"""
import numpy as np
import torch
from torch import nn

class LinearVariogramModel(nn.Module):
    def __init__(self):
        super(LinearVariogramModel, self).__init__()
        self.slope = nn.Parameter(torch.tensor(0.1))
        self.nugget = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        self.slope.data.clamp_(1e-7)
        self.nugget.data.clamp_(1e-7)

        return self.slope * x + self.nugget


def linear_variogram_model(m, d):
    """Linear model, m is [slope, nugget]"""
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


class PowerVariogramModel(nn.Module):
    def __init__(self):
        super(PowerVariogramModel, self).__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.exponent = nn.Parameter(torch.tensor(0.1))
        self.nugget = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        self.scale.data.clamp_(1e-7)
        self.exponent.data.clamp_(0.001, 1.999)
        self.nugget.data.clamp_(1e-7)

        return self.scale * x ** self.exponent + self.nugget


def power_variogram_model(m, d):
    """Power model, m is [scale, exponent, nugget]"""
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d ** exponent + nugget


class GaussianVariogramModel(nn.Module):
    def __init__(self):
        super(GaussianVariogramModel, self).__init__()
        self.psill = nn.Parameter(torch.tensor(0.1))
        self.range_ = nn.Parameter(torch.tensor(0.1))
        self.nugget = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        self.psill.data.clamp_(1e-7)
        self.range_.data.clamp_(1e-7)
        self.nugget.data.clamp_(1e-7)

        return self.psill * \
               (1.0 - torch.exp(-(x ** 2.0) / (self.range_ * 4.0 / 7.0) ** 2.0)) + \
               self.nugget


def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1.0 - np.exp(-(d ** 2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget


class ExponentialVariogramModel(nn.Module):
    def __init__(self):
        super(ExponentialVariogramModel, self).__init__()
        self.psill.data = nn.Parameter(torch.tensor(0.1))
        self.range_.data = nn.Parameter(torch.tensor(0.1))
        self.nugget.data = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        self.psill.clamp_(1e-7)
        self.range_.clamp_(1e-7)
        self.nugget.clamp_(1e-7)

        return self.psill * \
               (1.0 - torch.exp(-x / (self.range_ / 3.0))) + \
               self.nugget


def exponential_variogram_model(m, d):
    """Exponential model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1.0 - np.exp(-d / (range_ / 3.0))) + nugget


class SphericalVariogramModel(nn.Module):
    def __init__(self):
        super(SphericalVariogramModel, self).__init__()
        self.psill = nn.Parameter(torch.tensor(0.1))
        self.range_ = nn.Parameter(torch.tensor(0.1))
        self.nugget = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        self.psill.data.clamp_(1e-7)
        self.range_.data.clamp_(1e-7)
        self.nugget.data.clamp_(1e-7)

        result = self.psill * \
                 ((3.0 * x) / (2.0 * self.range_) -
                  (x ** 3.0) / (2.0 * self.range_ ** 3.0)) + \
                 self.nugget

        result[x == 0] = 0
        result[x > self.range_] = self.psill + self.nugget

        return result


def spherical_variogram_model(m, d):
    """Spherical model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return np.piecewise(
        d,
        [d <= range_, d > range_],
        [
            lambda x: psill
            * ((3.0 * x) / (2.0 * range_) - (x ** 3.0) / (2.0 * range_ ** 3.0))
            + nugget,
            psill + nugget,
        ],
    )


class HoleEffectVariogramModel(nn.Module):
    def __init__(self):
        super(HoleEffectVariogramModel, self).__init__()
        self.psill = nn.Parameter(torch.tensor(0.1))
        self.range_ = nn.Parameter(torch.tensor(0.1))
        self.nugget = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        self.psill.data.clamp_(1e-7)
        self.range_.data.clamp_(1e-7)
        self.nugget.data.clamp_(1e-7)

        return (
                self.psill *
                (1.0 - (1.0 - x / (self.range_ / 3.0)) *
                 torch.exp(-x / (self.range_ / 3.0)))
                + self.nugget
        )


def hole_effect_variogram_model(m, d):
    """Hole Effect model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return (
        psill * (1.0 - (1.0 - d / (range_ / 3.0)) * np.exp(-d / (range_ / 3.0)))
        + nugget
    )


if __name__ == '__main__':
    lm = LinearVariogramModel()