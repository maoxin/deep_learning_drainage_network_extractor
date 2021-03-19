# coding: utf-8
"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Methods used by multiple classes.

References
----------
[1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.

[2] T. Vincenty, Direct and Inverse Solutions of Geodesics on the Ellipsoid
    with Application of Nested Equations, Survey Review 23 (176),
    (Directorate of Overseas Survey, Kingston Road, Tolworth, Surrey 1975)

Copyright (c) 2015-2020, PyKrige Developers
"""

import torch

eps = 1.0e-10  # Cutoff for comparison to zero


def _initialize_variogram_model(
    X,
    Y,
    Z,
    variogram_model,
    variogram_function,
    nlags,
    weight,
    coordinates_type,
):
    """Initializes the variogram model for kriging. If user does not specify
    parameters, calls automatic variogram estimation routine.
    Returns lags, semivariance, and variogram model parameters.

    Parameters
    ----------
    X: ndarray -> (B, L)
        float array [L, 2], the input array of coordinates
    Y: ndarray -> (B, L)
    Z: ndarray -> (B, L)
        float array [L], the input array of values to be kriged
    variogram_model: str
        user-specified variogram model to use
    variogram_function: callable
        function that will be called to evaluate variogram model
        (only used if user does not specify variogram model parameters)
    nlags: int
        integer scalar, number of bins into which to group inter-point distances
    weight: bool
        boolean flag that indicates whether the semivariances at smaller lags
        should be weighted more heavily in the automatic variogram estimation
    coordinates_type: str
        type of coordinates in X array, can be 'euclidean' for standard
        rectangular coordinates or 'geographic' if the coordinates are lat/lon

    Returns
    -------
    lags: ndarray -> (B, nlags)
        float array [nlags], distance values for bins into which the
        semivariances were grouped
    semivariance: ndarray -> (B, nlags)
        float array [nlags], averaged semivariance for each bin
    """

    # distance calculation for rectangular coords now leverages
    # scipy.spatial.distance's pdist function, which gives pairwise distances
    # in a condensed distance vector (distance matrix flattened to a vector)
    # to calculate semivariances...
    if coordinates_type == "euclidean":
        xy_data = torch.stack([X, Y], 2)
        # (B, L, 2)
        d = torch.sqrt(((xy_data.unsqueeze(2) - xy_data.unsqueeze(1))**2).sum(3))
        # (B, L, L)

        g = 0.5 * (Z.unsqueeze(2) - Z.unsqueeze(1))**2
        # (B, L, L)

    # geographic coordinates only accepted if the problem is 2D
    # assume X[:, 0] ('x') => lon, X[:, 1] ('y') => lat
    # old method of distance calculation is retained here...
    # could be improved in the future
    elif coordinates_type == "geographic":
        raise NotImplementedError
    else:
        raise ValueError(
            "Specified coordinate type '%s' is not supported." % coordinates_type
        )

    # Equal-sized bins are now implemented. The upper limit on the bins
    # is appended to the list (instead of calculated as part of the
    # list comprehension) to avoid any numerical oddities
    # (specifically, say, ending up as 0.99999999999999 instead of 1.0).
    # Appending dmax + 0.001 ensures that the largest distance value
    # is included in the semivariogram calculation.
    dmax = d.max()
    dmin = d.min()
    dd = (dmax - dmin) / nlags
    bins = [dmin + n * dd for n in range(nlags)]
    dmax += 0.001
    bins.append(dmax)

    B = X.shape[0]
    lags = torch.zeros(nlags)
    semivariance = torch.zeros(nlags)

    for n in range(nlags):
        # This 'if... else...' statement ensures that there are data
        # in the bin so that numpy can actually find the mean. If we
        # don't test this first, then Python kicks out an annoying warning
        # message when there is an empty bin and we try to calculate the mean.
        lags[n] = (d[(d >= bins[n]) & (d < bins[n + 1])]).mean()
        semivariance[n] = (g[(d >= bins[n]) & (d < bins[n+1])]).mean()

    lags = lags[~torch.isnan(semivariance)]
    semivariance = semivariance[~torch.isnan(semivariance)]
    # (N, ), including B

    if weight:
        drange = lags.max() - lags.min()
        k = 2.1972 / (0.1 * drange)
        x0 = 0.7 * drange + lags.min()
        weights = 1.0 / (1.0 + torch.exp(-k * (x0 - lags)))
        weights = weights / weights.sum()

        resid = (((variogram_function(lags) - semivariance) * weights)**2).sum()

    else:
        resid = ((variogram_function(lags) - semivariance)**2).sum()


    return lags, semivariance, resid
