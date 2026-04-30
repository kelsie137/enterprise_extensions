#code adopted from bencebecsy/QuickCW/blob/main/QuickCW/PulsarDistPriors.py
#implements two new parameter type: pulsar directly measured distances and parallax measured distances

import numpy as np
from scipy.special import ndtr, ndtri
from enterprise.signals import parameter

DM_DIST_QMIN = 1.0e-3
DM_DIST_QMAX = 0.999


PX_DIST_QMIN = 1.0e-3
PX_DIST_QMAX = 0.999

def _check_dist_err(dist, err):
    if dist <= 0.0:
        raise ValueError("dist must be positive.")
    if err <= 0.0:
        raise ValueError("err must be positive.")


def _dm_prior_constants(dist, err):
    lo = 0.8 * dist
    hi = 1.2 * dist
    sig = 0.25 * err

    width = hi - lo
    gauss_area = sig * np.sqrt(2.0 * np.pi)
    tail_area = 0.5 * gauss_area
    total_area = width + gauss_area

    return lo, hi, sig, width, gauss_area, tail_area, total_area


def _DMDistLnPriorRaw(value, dist, err):
    """
    Normalized log-prior before quantile clipping.
    """

    _check_dist_err(dist, err)

    lo, hi, sig, width, gauss_area, tail_area, total_area = _dm_prior_constants(
        dist, err
    )

    x = np.asarray(value)

    logp = np.where(
        x < lo,
        -0.5 * ((x - lo) / sig)**2,
        np.where(
            x <= hi,
            0.0,
            -0.5 * ((x - hi) / sig)**2,
        ),
    )

    logp = logp - np.log(total_area)

    if np.ndim(value) == 0:
        return float(logp)
    return logp


def _DMDistPPFRaw(q, dist, err):
    """
    Inverse CDF before quantile clipping.
    """

    _check_dist_err(dist, err)

    lo, hi, sig, width, gauss_area, tail_area, total_area = _dm_prior_constants(
        dist, err
    )

    q_arr = np.asarray(q, dtype=float)
    scalar = q_arr.ndim == 0
    u = np.atleast_1d(q_arr)

    if np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("q must lie in [0, 1].")

    p_left = tail_area / total_area
    p_mid = width / total_area

    out = np.empty_like(u, dtype=float)

    left = u < p_left
    mid = (u >= p_left) & (u <= p_left + p_mid)
    right = ~(left | mid)

    if np.any(left):
        arg = u[left] * total_area / gauss_area
        out[left] = lo + sig * ndtri(np.clip(arg, 0.0, 1.0))

    if np.any(mid):
        out[mid] = lo + total_area * (u[mid] - p_left)

    if np.any(right):
        arg = (u[right] * total_area - width) / gauss_area
        out[right] = hi + sig * ndtri(np.clip(arg, 0.0, 1.0))

    if scalar:
        return float(out[0])
    return out.reshape(q_arr.shape)


def DMDistLnPrior(value, dist, err):
    """
    Clipped DM-distance log-prior.

    dist = d_DM
    err  = quoted DM-distance uncertainty
    sig  = sigma_DM = 0.25 * err

    The unclipped prior is flat on [0.8 d_DM, 1.2 d_DM],
    with half-Gaussian tails. The support is then clipped to
    the [DM_DIST_QMIN, DM_DIST_QMAX] quantiles.
    """

    _check_dist_err(dist, err)

    xmin = _DMDistPPFRaw(DM_DIST_QMIN, dist, err)
    xmax = _DMDistPPFRaw(DM_DIST_QMAX, dist, err)

    x = np.asarray(value)
    inside = (x >= xmin) & (x <= xmax)

    logp = _DMDistLnPriorRaw(x, dist, err)
    logp = np.where(
        inside,
        logp - np.log(DM_DIST_QMAX - DM_DIST_QMIN),
        -np.inf,
    )

    if np.ndim(value) == 0:
        return float(logp)
    return logp


def DMDistPPF(q, dist, err):
    """
    Inverse CDF for the clipped DM-distance prior.
    """

    _check_dist_err(dist, err)

    q_arr = np.asarray(q, dtype=float)

    if np.any((q_arr < 0.0) | (q_arr > 1.0)):
        raise ValueError("q must lie in [0, 1].")

    q_raw = DM_DIST_QMIN + (DM_DIST_QMAX - DM_DIST_QMIN) * q_arr
    return _DMDistPPFRaw(q_raw, dist, err)


def DMDistSampler(dist, err, size=None):
    """
    Draw from the clipped DM-distance prior.
    """

    q = np.random.uniform(size=size)
    return DMDistPPF(q, dist=dist, err=err)


def DMDistParameter(dist=0, err=1, size=None):
    """
    enterprise parameter class for the clipped DM-distance prior.
    """

    class DMDist(parameter.Parameter):
        _size = size

        _logprior = parameter.Function(DMDistLnPrior, dist=dist, err=err)
        _ppf = parameter.Function(DMDistPPF, dist=dist, err=err)

        # Keep enterprise-compatible sampler format.
        _sampler = staticmethod(DMDistSampler)

        _typename = parameter._argrepr("DMDist", dist=dist, err=err)

        def ppf(self, q, **kwargs):
            return self._ppf(q, **kwargs)

    return DMDist





def _check_dist_err(dist, err):
    if dist <= 0.0:
        raise ValueError("dist must be positive.")
    if err <= 0.0:
        raise ValueError("err must be positive.")


def _px_prior_constants(dist, err):
    # Gaussian prior on parallax pi = 1 / d.
    pi0 = 1.0 / dist
    pi_sig = err / dist**2

    # Positive-distance prior means conditioning on pi > 0.
    p_pos = 1.0 - ndtr((0.0 - pi0) / pi_sig)

    return pi0, pi_sig, p_pos


def _PXDistLnPriorRaw(value, dist, err):
    """
    Normalized log-prior before quantile clipping.

    This is the distance prior induced by a Gaussian parallax prior,
    conditioned on positive parallax / positive distance.
    """

    _check_dist_err(dist, err)

    pi0, pi_sig, p_pos = _px_prior_constants(dist, err)

    d = np.asarray(value)
    inside = d > 0.0

    pi = 1.0 / d
    z = (pi - pi0) / pi_sig

    logp = (
        -0.5 * z**2
        - np.log(np.sqrt(2.0 * np.pi) * pi_sig)
        - 2.0 * np.log(d)
        - np.log(p_pos)
    )

    logp = np.where(inside, logp, -np.inf)

    if np.ndim(value) == 0:
        return float(logp)

    return logp


def _PXDistPPFRaw(q, dist, err):
    """
    Inverse CDF before quantile clipping.

    q is a distance quantile, with positive parallax conditioning.
    """

    _check_dist_err(dist, err)

    pi0, pi_sig, p_pos = _px_prior_constants(dist, err)

    q_arr = np.asarray(q, dtype=float)
    scalar = q_arr.ndim == 0
    u = np.atleast_1d(q_arr)

    if np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("q must lie in [0, 1].")

    # For d = 1/pi, larger distances correspond to smaller positive parallaxes.
    # F_D(d | pi > 0) = P(pi >= 1/d | pi > 0).
    cdf_pi_at_cut = 1.0 - u * p_pos
    pi = pi0 + pi_sig * ndtri(cdf_pi_at_cut)

    out = 1.0 / pi

    if scalar:
        return float(out[0])

    return out.reshape(q_arr.shape)


def PXDistLnPrior(value, dist, err):
    """
    Clipped PX-distance log-prior.

    The unclipped prior is induced by a Gaussian parallax prior,
    conditioned on positive distance. The support is clipped to
    [PX_DIST_QMIN, PX_DIST_QMAX] quantiles.
    """

    _check_dist_err(dist, err)

    xmin = _PXDistPPFRaw(PX_DIST_QMIN, dist, err)
    xmax = _PXDistPPFRaw(PX_DIST_QMAX, dist, err)

    d = np.asarray(value)
    inside = (d >= xmin) & (d <= xmax)

    logp = _PXDistLnPriorRaw(d, dist, err)
    logp = np.where(
        inside,
        logp - np.log(PX_DIST_QMAX - PX_DIST_QMIN),
        -np.inf,
    )

    if np.ndim(value) == 0:
        return float(logp)

    return logp


def PXDistPPF(q, dist, err):
    """
    Inverse CDF for the clipped PX-distance prior.
    """

    _check_dist_err(dist, err)

    q_arr = np.asarray(q, dtype=float)

    if np.any((q_arr < 0.0) | (q_arr > 1.0)):
        raise ValueError("q must lie in [0, 1].")

    q_raw = PX_DIST_QMIN + (PX_DIST_QMAX - PX_DIST_QMIN) * q_arr
    return _PXDistPPFRaw(q_raw, dist, err)


def PXDistSampler(dist, err, size=None):
    """
    Draw from the clipped PX-distance prior.
    """

    q = np.random.uniform(size=size)
    return PXDistPPF(q, dist=dist, err=err)


def PXDistParameter(dist=0, err=1, size=None):
    """
    enterprise parameter class for the clipped PX-distance prior.
    """

    class PXDist(parameter.Parameter):
        _size = size

        _logprior = parameter.Function(PXDistLnPrior, dist=dist, err=err)
        _ppf = parameter.Function(PXDistPPF, dist=dist, err=err)

        _sampler = staticmethod(PXDistSampler)

        _typename = parameter._argrepr("PXDist", dist=dist, err=err)

        def ppf(self, q, **kwargs):
            return self._ppf(q, **kwargs)

    return PXDist