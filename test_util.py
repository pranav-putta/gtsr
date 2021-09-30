import unittest
from util import shift_polynomial, load_config, generate_model_fit, test_model_on_validation
from matplotlib import pyplot as plt
import numpy as np


class TestUtil(unittest.TestCase):
    def test_polynomial_shift(self):
        poly = [1, 2, 3, 4, 5, 6]  # x^5 + 2x^4 + 3x^3 + 4x^2 + 6x + 6
        shift = 3
        correct = [1, -13, 69, -185, 251, -135]

        transformed = shift_polynomial(poly, shift)
        self.assertListEqual(correct, transformed)

    def test_process(self):
        config = load_config()
        model = generate_model_fit(config)
        test_model_on_validation(model, config)


if __name__ == '__main__':
    unittest.main()
