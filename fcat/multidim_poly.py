import numpy as np
import logging
from typing import Sequence
from itertools import product
from fcat.utilities import aicc
from copy import deepcopy

__all__ = ('MultiDimPolynomial', 'optimize_poly_order')


def polynomial_term_iterator(order: Sequence[int], variables: Sequence[float]):
    """
    This function iterates through all terms in a polynomial up to the passed order

    :param order: Maximum order for all parameters
    :parm variables: Explanatory variables

    Example:
    If order is [1, 2] and variables is [x, y]

    this returns the terms
    const, x, y, y^2, x*y, x*y^2
    """
    for powers in product(*[range(o+1) for o in order]):
        yield np.prod([v**p for v, p in zip(variables, powers)])


class MultiDimPolynomial:
    """
    Class for fitting a multidimensional polynomial do a set data

    :param data: Dataset. It is assumed that the target variable is given
        in the last column.
        Exampe:

        x1, x2, y
        0.1, 0.2, 4.0
        0.2, 0.1, 4.0

    :param order: Order of the different parameters. The length of the order list
        must match the the number of features. For the example above where there
        is two explanatory variables. We could fit a polynomial up to first
        order in the first parameter and third order in the second argument
        by passing [1, 3]. This, is equivalent of fitting

        y = c0 + c1*x1 + c2*x2 + c2*x2^2 + c3*x2^3 + c4*x1*x2 + c5*x1*x2^2 + c6*x1*x2^3
    """
    def __init__(self, data: np.ndarray, order: Sequence[int]):
        self.data = data
        self.order = order
        self.coeff = self.fit()

    def fit(self) -> np.ndarray:
        """
        Performs the fit of the selected data
        """
        y = self.data[:, -1]
        orig_design_mat = self.data[:, :-1]
        num_data, num_feat = orig_design_mat.shape

        if len(self.order) != num_feat:
            raise ValueError("order must be the same length as the number of original features")

        # Build the fitting matrix
        design_matrix = []

        for i in range(num_data):
            design_matrix.append(
                [term for term in polynomial_term_iterator(self.order, orig_design_mat[i, :])]
            )

        design_matrix = np.array(design_matrix)
        coeff = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
        return coeff

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluates the polynomial at position x
        """
        return sum(self.coeff[i]*v for i, v in enumerate(polynomial_term_iterator(self.order, x)))

    @property
    def num_data(self) -> int:
        return self.data.shape[0]

    @property
    def rmse(self) -> float:
        """
        Returns the root mean square error of the fit
        """
        predictions = [self.evaluate(self.data[i, :-1]) for i in range(self.num_data)]
        diff = predictions - self.data[:, -1]
        return np.sqrt(np.mean(diff**2))

    @property
    def num_features(self) -> int:
        return len(self.coeff)

    @property
    def aicc(self) -> float:
        return aicc(self.num_data, self.num_features, self.rmse)

    def show(self):
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        predictions = [self.evaluate(self.data[i, :-1]) for i in range(self.data.shape[0])]
        minval, maxval = np.min(self.data[:, -1]), np.max(self.data[:, -1])
        rng = maxval - minval

        ax.plot([minval-0.05*rng, maxval+0.05*rng], [minval-0.05*rng, maxval+0.05*rng],
                color='black')

        ax.plot(predictions, self.data[:, -1], 'o', mfc='none', color='#222222')
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Exact value")
        return fig

    def print_report(self):
        """
        Print a report of the fit
        """
        print("================================================")
        print("         Multidimensional poly summary          ")
        print("================================================")
        print(f"Order:            {self.order}")
        print(f"RMSE:             {self.rmse:.2e}")
        print(f"AICC:             {self.aicc:.2e}")
        print(f"Num. coeff:       {self.num_features}")
        print(f"Num. data points: {self.num_data}")
        print("================================================")


def optimize_poly_order(data: np.ndarray, max_order: Sequence[int]) -> MultiDimPolynomial:
    """
    Optimizes the polynomial order by trying all combinations up to

    :param data: Dataset to be fitted. See `class:fcat.MultiDimPolynomial` for further details
    :param order: List of maximum order of each exlanatory variable
    """
    best_aicc = np.inf
    best_model = None
    best_order = None
    for order in product(*[range(o+1) for o in max_order]):
        poly = MultiDimPolynomial(data, order)
        if poly.num_features >= poly.num_data - 1 and best_model is not None:
            continue

        new_aicc = aicc(data.shape[0], poly.num_features, poly.rmse)
        if new_aicc < best_aicc:
            best_model = deepcopy(poly)
            best_order = deepcopy(order)
            best_aicc = new_aicc
        logging.info(f"Order {order}: AICC: {new_aicc}")

    logging.info(f"Best AICC value found for order {best_order}: {best_aicc}")
    return best_model
