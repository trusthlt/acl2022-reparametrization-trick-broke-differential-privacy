import unittest
from experiment import Experiment, NoPrivacyMechanism, CompletelyRandomMechanism, LaplaceMechanism, \
    BeigiEtAlLaplaceMechanism, LaplaceMechanismWrongSensitivity

import numpy as np


class Test(unittest.TestCase):

    def test_loss_from_matrix1(self):
        frequency_matrix = np.array(
            [
                [10, 0],
                [0, 10]
            ])

        # this is infinite privacy loss: both data points would have perfect reconstruction precision
        loss = Experiment.compute_loss_from_frequency_matrix(frequency_matrix)
        self.assertEqual(loss, np.infty)

    def test_loss_from_matrix2(self):
        frequency_matrix = np.array(
            [
                [10, 10],
                [10, 10]
            ])

        # this is zero privacy loss -- all is random
        loss = Experiment.compute_loss_from_frequency_matrix(frequency_matrix)
        self.assertAlmostEqual(loss, 0, delta=0.00001)

    def test_no_privacy_mechanism(self):
        # regardless of dimensionality or epsilon, this must end up with infinite privacy loss, as
        # we simply copy input to output
        loss = Experiment.estimate_empirical_loss(100, 0.001, NoPrivacyMechanism())
        self.assertEqual(loss, np.infty)

    def test_absolute_privacy_mechanism(self):
        # regardless of dimensionality or epsilon, this should end up with almost zero privacy loss
        loss = Experiment.estimate_empirical_loss(100, 10, CompletelyRandomMechanism())
        self.assertLessEqual(loss, 0.01)

    def test_reconstruct_original_vectors1(self):
        a = np.array(
            [
                [1, 0, 0],
                [0, 1, 1]
            ]
        )
        out = Experiment.reconstruct_original_vector(a)
        # we expect the first to be (0, 0, 0, and the second (1, 1, 1)
        self.assertTrue(np.array_equiv(out, np.array([[0, 0, 0], [1, 1, 1]])))

    def test_reconstruct_original_vectors2(self):
        a = np.array(
            [
                [1],
                [0]
            ]
        )
        out = Experiment.reconstruct_original_vector(a)
        # we get the input back
        self.assertTrue(np.array_equiv(out, a))

    def test_laplace_inv_cdf(self):
        samples = LaplaceMechanism.laplace_inv_cdf_correct(10_000_000)
        # must be zero-mean
        self.assertAlmostEqual(0, samples.mean(), places=2)

        # must be variance 2b^2 = 2
        self.assertAlmostEqual(2.0, np.mean(np.power(samples, 2)).item(), places=2)

    def test_laplace_inv_cdf_beigi(self):
        samples = BeigiEtAlLaplaceMechanism.laplace_inv_cdf_dptext(10_000_000)
        # must be at least the same size as input!
        self.assertEqual(10_000_000, samples.shape[0])

        # should be zero-mean, but it's not
        self.assertNotAlmostEqual(0, samples.mean(), places=2)

        # should be be variance 2b^2 = 2, but it's not
        self.assertNotAlmostEqual(2.0, np.mean(np.power(samples, 2)).item(), places=2)

    def test_estimate_empirical_loss1(self):
        loss = Experiment.estimate_empirical_loss(1, 1, LaplaceMechanism(), number_of_repeats=1_000_000)
        self.assertLess(loss, 1.0)

    def test_estimate_empirical_loss2(self):
        # now with broken Laplace
        loss = Experiment.estimate_empirical_loss(1, 1, LaplaceMechanismWrongSensitivity(), number_of_repeats=10_000_000)
        # which ends up with loss greater than epsilon
        self.assertGreater(loss, 1.0)


if __name__ == '__main__':
    unittest.main()
