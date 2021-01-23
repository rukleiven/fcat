import numpy as np
from typing import Sequence
from itertools import product


__all__ = ('MultiDimPolynomial',)


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
