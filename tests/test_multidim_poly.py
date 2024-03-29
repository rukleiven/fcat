import pytest
import numpy as np
from fcat.multidim_poly import polynomial_term_iterator
from fcat import MultiDimPolynomial, optimize_poly_order


@pytest.mark.parametrize('order, variables, want', [
    #        (  x,  y)   (const, y,   x, x*y)
    ([1, 1], (2.0, 3.0), (1.0, 3.0, 2.0, 6.0)),

    #        (  x,   y)   (const, x, x*x)
    ([2, 0], (2.0, 3.0), (1.0, 2.0, 4.0)),

    #        (  x,   y)   (const, y, y*y)
    ([0, 2], (2.0, 3.0), (1.0, 3.0, 9.0)),

    #           (  x,   y,   z)  (const, y, y*y)
    ([0, 2, 0], (2.0, 3.0, 4.0), (1.0, 3.0, 9.0)),

    #           (  x,   y,   z)  (const, z,   y,  y*z, y*y, y*y*z)
    ([0, 2, 1], (2.0, 3.0, 4.0), (1.0, 4.0, 3.0, 12.0, 9.0,  36.0)),

    #           (  x,   y,   z)  (const, z,  z*z,   y,  y*z, y*z*z,y*y,y*y*z, y*y*z*z, x, x*z
    ([2, 2, 2], (2.0, 3.0, 4.0), (1.0, 4.0, 16.0, 3.0, 12.0, 48.0, 9.0, 36.0, 144.0, 2.0, 8.0,
                                  #                             x*z*z, x*y,x*y*z,x*y*z*z, x*y*y, x*y*y*z, x*y*y*z*z,x*x, x*x*z, x*x*z*z, x*x*y*z
                                  32.0, 6.0, 24.0, 96.0, 18.0, 72.0, 288.0, 4.0, 16.0, 64.0, 12.0, 48.0,
                                  #                             x*x*y*z*z, x*x*y*y, x*x*y*y*z, x*x*y*y*z*z
                                  192.0, 36.0, 144.0, 576.0
                                  ))
])
def test_poly_iterator(order, variables, want):
    items = list(polynomial_term_iterator(order, variables))
    assert np.allclose(items, want)

# This test runs a bunch of fits where the error should be zero


@pytest.mark.parametrize('order, data, want', [
    # [x1, x2]: y = 0.2*x1 - 0.2
    ([1, 0], np.array([[1.0, 2.0, 0.0], [2.0, -1.0, 0.2]]), (-0.2, 0.2)),
    # [x1, x2]: y = 0.1*x1
    ([1, 0], np.array([[1.0, 2.0, 0.1], [2.0, -1.0, 0.2], [3.0, -1.0, 0.3],  [4.0, -1.0, 0.4]]), (0.0, 0.1)),
    # [x1, x2]: y = 0.1 - 0.2*x1 + 0.2*x2                                                       const, x2, c1, x1*x2
    ([1, 1], np.array([[0.0, 0.0, 0.1], [1.0, 1.0, 0.1], [
     1.0, -1.0, -0.3], [0.0, 1.0, 0.3]]), [0.1, 0.2, -0.2, 0.0])
]
)
def test_fit_given_order_consistency(order, data, want):
    fitter = MultiDimPolynomial(data, order)

    X = data[:, :-1]
    y = data[:, -1]
    pred = [fitter.evaluate(X[i, :]) for i in range(X.shape[0])]
    diff = pred - y
    assert np.sqrt(np.mean(diff**2)) == pytest.approx(0.0)
    assert fitter.rmse == pytest.approx(0.0)
    assert np.allclose(fitter.coeff, want)


def test_optimize_model():
    np.random.seed(0)
    data = np.random.rand(10, 3)
    model = optimize_poly_order(data, [4, 4])

    # The number of coefficients must be smaller than the number of data points
    assert model.num_features < data.shape[0]


def test_optimize_model_known_results():
    # y = 0.1 - 0.2*x1 + 0.2*x2
    data = np.array([[0.0, 0.0, 0.1],
                     [1.0, 1.0, 0.1],
                     [1.0, -1.0, -0.3],
                     [0.0, 1.0, 0.3],
                     [0.0, 2.0, 0.5],
                     [1.0, 2.0, 0.3],
                     [2.0, 2.0, 0.1],
                     [-2.0, 2.0, 0.9]])

    want = [0.1, 0.2, -0.2, 0.0]

    model = optimize_poly_order(data, [4, 4])
    assert all(x == y for x, y in zip(model.order, [1, 1]))
    assert np.allclose(want, model.coeff)


def test_fit_model():
    np.random.seed(0)
    data = np.random.rand(10, 3)
    model = MultiDimPolynomial(data, [2, 2])
    fig = model.show()
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert len(ax.lines[0].get_xdata()) == 2
    assert len(ax.lines[0].get_ydata()) == 2
    assert len(ax.lines[1].get_xdata()) == data.shape[0]
    assert len(ax.lines[1].get_ydata()) == data.shape[0]
