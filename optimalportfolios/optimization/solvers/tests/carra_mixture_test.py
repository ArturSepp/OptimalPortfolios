import numpy as np
from enum import Enum

from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.solvers.carra_mixure import opt_maximize_cara, opt_maximize_cara_mixture


class LocalTests(Enum):
    CARA = 1
    CARA_MIX = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.CARA:
        means = np.array([0.3, 0.1])
        covar = np.array([[0.2 ** 2, 0.01],
                          [0.01, 0.1 ** 2]])
        w_rb = opt_maximize_cara(means=means, covar=covar, carra=10, is_exp=False, disp=True)
        w_rb = opt_maximize_cara(means=means, covar=covar, carra=10, is_exp=True, disp=True)

    elif local_test == LocalTests.CARA_MIX:
        means = [np.array([0.05, -0.1]), np.array([0.05, 2.0])]
        covars = [np.array([[0.2 ** 2, 0.01],
                          [0.01, 0.2 ** 2]]),
                 np.array([[0.2 ** 2, 0.01],
                           [0.01, 0.2 ** 2]])
                 ]
        probs = np.array([0.95, 0.05])
        optimal_weights = opt_maximize_cara_mixture(means=means, covars=covars, probs=probs,
                                                    constraints=Constraints(),
                                                    carra=20.0, verbose=True)
        print(optimal_weights)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CARA_MIX)