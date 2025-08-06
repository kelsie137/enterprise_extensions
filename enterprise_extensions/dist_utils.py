#code adopted from bencebecsy/QuickCW/blob/main/QuickCW/PulsarDistPriors.py
#implements two new parameter type: pulsar directly measured distances and parallax measured distances

import numpy as np
from scipy.stats import norm

from enterprise.signals import parameter

def DMDistLnPrior(value, dist, err):
    """
    Log-prior for DMDist parameters.
    Flat between (dist - err, dist + err), with Gaussian tails on both sides.

    Parameters
    ----------
    value : float or np.ndarray
        Evaluation point(s).
    dist : float
        Central distance.
    err : float
        Symmetric uncertainty (half-width of flat region).

    Returns
    -------
    ln_prior : float or np.ndarray
        Log prior evaluated at the input value(s).
    """
    # Define edges and parameters
    box_low = dist - err
    box_high = dist + err
    sigma = 0.25 * err

    # Heights
    box_height = 1 / (box_high - box_low)
    gauss_height = 1 / (np.sqrt(2 * np.pi) * sigma)
    log_scaling = np.log(box_height / gauss_height)

    # Evaluate log prior
    ln_prior = np.where(
        value <= box_low,
        norm.logpdf(value, loc=box_low, scale=sigma) + log_scaling,
        np.where(
            value < box_high,
            np.log(box_height),
            norm.logpdf(value, loc=box_high, scale=sigma) + log_scaling
        )
    )

    # Normalization: log(area)
    norm_area = 1 + box_height / gauss_height
    return ln_prior - np.log(norm_area)


def DMDistSampler(dist, err, size=None):
    """Sampling function for DMDist parameters.

    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter

    :return:        random draw from prior (float or ndarray with lenght size)
    """

    boxheight = 1/((dist+err)-(dist-err))
    gaussheight = 1/(np.sqrt(2*np.pi*(0.25*err)**2))
    area = 1+1*boxheight/gaussheight

    #probability of being in the uniform part
    boxprob = 1/area

    #decide if we are in the middle or not
    alpha = np.random.uniform()
    if alpha<boxprob:
        return np.random.uniform(dist-err, dist+err)
    else:
        x = np.random.normal(0, 0.25*err, size=size)
        if x>0.0:
            return x+dist+err
        else:
            return x+dist-err

def DMDistParameter(dist=0, err=1, size=None):
    """Class factory for DM distance parameters with a pdf that is
    flat for dist+-err and a half Gaussian beyond that

    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter

    :return:        ``DMDist`` parameter class
    """

    class DMDist(parameter.Parameter):
        _size = size
        _logprior = parameter.Function(DMDistLnPrior, dist=dist, err=err)
        _sampler = staticmethod(DMDistSampler)
        _typename = parameter._argrepr("DMDist", dist=dist, err=err)

    return DMDist



def PXDistLnPrior(value, dist, err):
    """
    Log-prior for PXDist parameters.
    Prior over distance based on Gaussian distribution for parallax (1/value).

    Parameters
    ----------
    value : float or np.ndarray
        Evaluation point(s), representing distance.
    dist : float
        Mean distance.
    err : float
        Distance uncertainty.

    Returns
    -------
    ln_prior : float or np.ndarray
        Log prior evaluated at the input value(s).
    """
    # Gaussian in parallax space
    pi = 1.0 / dist
    pi_err = err / dist**2

    # Convert to log-prior using transformation rule for PDF under inverse
    # p(x) = N(1/x; pi, pi_err) * (1/x^2)
    # ln p(x) = ln N(1/x) - 2 ln x

    inv_value = 1.0 / value
    ln_pdf = -0.5 * ((inv_value - pi) / pi_err)**2 - np.log(np.sqrt(2 * np.pi) * pi_err)
    ln_jacobian = -2 * np.log(value)

    return ln_pdf + ln_jacobian



def PXDistSampler(dist, err, size=None):
    """Sampling function for PXDist parameters.

    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter

    :return:        random draw from prior (float or ndarray with lenght size)
    """

    pi = 1/dist
    pi_err = err/dist**2

    #just draw parallax from Gaussian with proper mean and std and return its inverse
    return 1/np.random.normal(pi, pi_err)

def PXDistParameter(dist=0, err=1, size=None):
    """Class factory for PX distance parameters with a pdf of inverse Gaussian (since parallax is Gaussian)
    
    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter
    
    :return:        ``PXDist`` parameter class
    """

    class PXDist(parameter.Parameter):
        _size = size
        _logprior = parameter.Function(PXDistLnPrior, dist=dist, err=err)
        _sampler = staticmethod(PXDistSampler)
        _typename = parameter._argrepr("PXDist", dist=dist, err=err)

    return PXDist
