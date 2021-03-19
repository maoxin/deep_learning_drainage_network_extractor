# coding: utf-8
"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Contains class OrdinaryKriging, which provides easy access to
2D Ordinary Kriging.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.
.. [2] N. Cressie, Statistics for spatial data, 
   (Wiley Series in Probability and Statistics, 1993) 137 p.

Copyright (c) 2015-2020, PyKrige Developers
"""

from torch import nn
import torch

from models.kriging import variogram_models
from models.kriging.core import (
    _initialize_variogram_model,
)


class OrdinaryKriging(nn.Module):
    r"""Convenience class for easy access to 2D Ordinary Kriging.

    Parameters
    ----------
    variogram_model : str
        Specifies which variogram model to use; may be one of the following:
        linear, power, gaussian, spherical, exponential, hole-effect.
        Default is linear variogram model.
    nlags : int, optional
        Number of averaging bins for the semivariogram. Default is 6.
    weight : bool, optional
        Flag that specifies if semivariance at smaller lags should be weighted
        more heavily when automatically calculating variogram model.
        The routine is currently hard-coded such that the weights are
        calculated from a logistic function, so weights at small lags are ~1
        and weights at the longest lags are ~0; the center of the logistic
        weighting is hard-coded to be at 70% of the distance from the shortest
        lag to the largest lag. Setting this parameter to True indicates that
        weights will be applied. Default is False. (Kitanidis suggests that the
        values at smaller lags are more important in fitting a variogram model,
        so the option is provided to enable such weighting.)
    verbose : bool, optional
        Enables program text output to monitor kriging process.
        Default is False (off).
    coordinates_type : str, optional
        One of 'euclidean' or 'geographic'. Determines if the x and y
        coordinates are interpreted as on a plane ('euclidean') or as
        coordinates on a sphere ('geographic'). In case of geographic
        coordinates, x is interpreted as longitude and y as latitude
        coordinates, both given in degree. Longitudes are expected in
        [0, 360] and latitudes in [-90, 90]. Default is 'euclidean'.
    itp_H : int, the height of the image to interpolate
    itp_W : int, the width of the image to interpolate

    References
    ----------
    .. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
       Hydrogeology, (Cambridge University Press, 1997) 272 p.
    .. [2] N. Cressie, Statistics for spatial data, 
       (Wiley Series in Probability and Statistics, 1993) 137 p.
    """

    eps = 1.0e-10  # Cutoff for comparison to zero
    variogram_dict = {
        "linear": variogram_models.LinearVariogramModel,
        "power": variogram_models.PowerVariogramModel,
        "gaussian": variogram_models.GaussianVariogramModel,
        "spherical": variogram_models.SphericalVariogramModel,
        "exponential": variogram_models.ExponentialVariogramModel,
        "hole-effect": variogram_models.HoleEffectVariogramModel,
    }

    def __init__(
        self,
        variogram_model="linear",
        weight=False,
        verbose=False,
        coordinates_type="euclidean",
        nlags=6,
        itp_H = 64,
        itp_W = 64,
    ):
        super(OrdinaryKriging, self).__init__()

        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        self.model = None

        if (self.variogram_model not in self.variogram_dict.keys()):
            raise ValueError(
                "Specified variogram model '%s' is not supported." % variogram_model
            )
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]()

        self.verbose = verbose
        self.coordinates_type = coordinates_type
        self.weight = weight
        self.nlags = nlags
        self.x_points = torch.arange(itp_H, dtype=torch.float32)
        self.y_points = torch.arange(itp_W, dtype=torch.float32)

    def forward(self, x, y, z):
        """

        :param x: i indices, (B, H, W)
        :param y: j indices, (B, H, W)
        :param z: value, (B, H, W)
        :return:
        """
        # Code assumes 1D input arrays of floats. Ensures that any extraneous
        # dimensions don't get in the way. Copies are created to avoid any
        # problems with referencing the original passed arguments.
        # Also, values are forced to be float... in the future, might be worth
        # developing complex-number kriging (useful for vector field kriging)
        X_ORIG = x.flatten(start_dim=1)
        Y_ORIG = y.flatten(start_dim=1)
        Z = z.flatten(start_dim=1)
        # (B, L)

        # self.verbose = verbose
        # self.enable_plotting = enable_plotting
        # if self.enable_plotting and self.verbose:
        #     print("Plotting Enabled\n")

        # adjust for anisotropy... only implemented for euclidean (rectangular)
        # coordinates, as anisotropy is ambiguous for geographic coordinates...
        if self.coordinates_type == "euclidean":
            X_ADJUSTED = X_ORIG
            Y_ADJUSTED = Y_ORIG
        elif self.coordinates_type == "geographic":
            raise NotImplementedError
        else:
            raise ValueError(
                "Only 'euclidean' and 'geographic' are valid "
                "values for coordinates-keyword."
            )

        if self.verbose:
            print("Initializing variogram model...")

        (
            self.lags,
            self.semivariance,
            loss,
        ) = _initialize_variogram_model(
            # np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
            X_ADJUSTED,
            Y_ADJUSTED,
            Z,
            self.variogram_model,
            self.variogram_function,
            self.nlags,
            self.weight,
            self.coordinates_type,
        )

        zvalues, _ = self.execute(self.x_points, self.y_points, X_ADJUSTED, Y_ADJUSTED, Z)
        # (B, H, W)

        return loss, zvalues


    def _get_kriging_matrix(self, X_ADJUSTED, Y_ADJUSTED):
        """
        Assembles the kriging matrix.

        :param X_ADJUSTED: (B, L)
        :param Y_ADJUSTED: (B, L)
        :return: a (B, L+1, L+1)
        """

        B, L = X_ADJUSTED.shape

        if self.coordinates_type == "euclidean":
            xy = torch.stack([X_ADJUSTED, Y_ADJUSTED], 2)
            # (B, L, 2)
            d = torch.sqrt(((xy.unsqueeze(2) - xy.unsqueeze(1))**2).sum(dim=3))
            # (B, L, L)
        elif self.coordinates_type == "geographic":
            raise NotImplementedError

        a = torch.zeros(B, L + 1, L + 1)
        a[:, :L, :L] = -self.variogram_function(d)

        torch.diagonal(a, dim1=1, dim2=2).fill_(0)
        a[:, L, :] = 1.0
        a[:, :, L] = 1.0
        a[:, L, L] = 0.0

        return a

    def _exec_vector(self, a, bd, Z):
        """
        Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets.

        :param a: (B, L+1, L+1)
        :param bd: (B, NM, L)
        :param Z: (B, L)
        :return:
        """

        B, NM, L = bd.shape
        zero_index = None
        zero_value = False

        if (torch.abs(bd) < self.eps).any():
            zero_value = True

            zero_index = torch.abs(bd) <= self.eps

        b = torch.zeros((B, NM, L + 1))
        # (B, NM, L+1)
        b[:, :, :L] = -self.variogram_function(bd)
        if zero_value:
            b[:, :, :L][zero_index] = 0

        b[:, :, L] = 1.0

        # x = torch.stack([torch.lstsq(b[i].permute(1, 0), a[i])[0].permute(1, 0) for i in range(len(b))], dim=0)
        x = torch.solve(b.permute(0, 2, 1), a)[0].permute(0, 2, 1)
        # (B, NM, L)

        zvalues = (x[:, :, :L] * Z.unsqueeze(1)).sum(2)
        # (B, NM)
        sigmasq = (x * -b).sum(2)
        # (B, NM)

        return zvalues, sigmasq

    def execute(
        self,
        xpoints,
        ypoints,
        X_ADJUSTED,
        Y_ADJUSTED,
        Z,
    ):
        """Calculates a kriged grid and the associated variance with grid format

        Parameters
        ----------
        xpoints : array_like, shape (N,)
            x-coordinates of MxN grid.
        ypoints : array_like, shape (M,)
            y-coordinates of MxN grid
            Note that in this case, xpoints and ypoints must have
            the same dimensions (i.e., M = N).
        X_ADJUSTED, Y_ADJUSTED : (B, L)

        Returns
        -------
        zvalues : ndarray, shape (B, N, M)
            Z-values of specified grid or at the specified set of points.
        sigmasq : ndarray, shape (B, N, M)
            Variance at specified grid points or at the specified
            set of points.
        """

        if self.verbose:
            print("Executing Ordinary Kriging...\n")

        xpts = xpoints
        ypts = ypoints
        assert len(xpts.shape) == 1
        assert len(ypts.shape) == 1
        nx = xpts.size(0) # N
        ny = ypts.size(0) # M
        a = self._get_kriging_matrix(X_ADJUSTED, Y_ADJUSTED)

        grid_x, grid_y = torch.meshgrid(xpts, ypts)
        # (N, M)
        xpts = grid_x.flatten()
        ypts = grid_y.flatten()
        # (NM,)

        if self.coordinates_type == "euclidean":
            # # Prepare for cdist:
            xy_data = torch.stack([X_ADJUSTED, Y_ADJUSTED], 2)
            # (B, L, 2)

            xy_points = torch.stack([xpts, ypts], 1).unsqueeze(0)
            # (1, NM, 2)

            bd = torch.sqrt(((xy_points.unsqueeze(2) - xy_data.unsqueeze(1)) ** 2).sum(3))

        elif self.coordinates_type == "geographic":
            # In spherical coordinates, we do not correct for anisotropy.
            # Also, we don't use scipy.spatial.cdist, so we do not have to
            # format the input data accordingly.
            raise NotImplementedError

        zvalues, sigmasq = self._exec_vector(a, bd, Z)
        # (B, NM)

        B = zvalues.shape[0]
        zvalues = zvalues.view((B, nx, ny))
        sigmasq = sigmasq.view((B, nx, ny))
        # (B, N, M)

        return zvalues, sigmasq


if __name__ == '__main__':
    import random
    models = [
        'linear',
        'power',
        'gaussian',
        'spherical',
        'exponential',
        'hole-effect',
    ]

    # test all models ok
    delta_i = [0, 1]
    delta_j = [0, 1]

    z = torch.randn(6, 256, 256)
    x = torch.zeros(6, 256, 256)
    y = torch.zeros(6, 256, 256)

    for b in range(6)[:1]:
        for i in range(256):
            for j in range(256):
                i_ = i * 2 + random.choice(delta_i)
                j_ = j * 2 + random.choice(delta_j)

                x[b, i, j] = i_
                y[b, i, j] = j_
    #
    # for m in models:
    #     ok = OrdinaryKriging(m, False, True, 'euclidean', 6, 64, 64)
    #     loss, zvalues = ok(x, y, z)

    # test function ok
    from pykrige.ok import OrdinaryKriging as OK2
    import numpy as np
    x = x[0]
    y = y[0]
    z = np.load("/media/data1/mx_dataset/drainage_network_v2/samples/HU4_0101_i10243_j14851_elev_cm_crop512_pad3.npy")
    z = torch.tensor(z[:256, :256], dtype=torch.float32)
    z = (z - z.min()) / z.max() * 10

    ok = OrdinaryKriging('linear', False, False, 'euclidean', 6, 512, 512)
    # ok2 = OK2(x.flatten().numpy(), y.flatten().numpy(), z.flatten().numpy(), 'linear',
    #           verbose=False, enable_plotting=False)

    # z_gt, ss = ok2.execute(style='grid',
    #                       xpoints=ok.x_points.flatten().numpy(),
    #                       ypoints=ok.y_points.flatten().numpy())

    from torch.optim import Adam
    from tqdm import tqdm
    optimizer = Adam(ok.parameters(), lr=0.1)
    for i in tqdm(range(100)):
        optimizer.zero_grad()
        loss, z_pred = ok(x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0))
        tqdm.write(str(loss.item()))
        loss.backward()
        optimizer.step()

    # print(ok2.variogram_model_parameters)
    print(ok.variogram_function.slope.detach().item(),
          ok.variogram_function.nugget.detach().item())
    print(ok.lags)
    print(ok.semivariance)




    # gridx = torch.tensor(np.arange(0.0, 5.5, 0.5))
    # gridy = torch.tensor(np.arange(0.0, 5.5, 0.5))
    pass