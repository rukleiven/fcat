import numpy as np


def aicc(num_data: int, num_features: int, rmse: float) -> float:
    """
    Calculates the corrected Afaikes information criterion

    :param num_data: Number of data points
    :param rmse: Root mean square error
    :param num_features: Number of fitting parameters
    """
    return aic(num_data, rmse) + (2*num_features**2 + 2.0*num_features)/(num_data - num_features - 1)


def aic(num_features: int, rmse: float) -> float:
    """
    Calculates the Afaikes information criterion

    :param rmse: Root mean square error
    :param num_features: Number of fitting parameters
    """
    return 2.0*num_features - 2.0*np.log(rmse)
