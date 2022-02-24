import json

import numpy as np


class RealVectorDPMechanism:

    def execute(self, query_output: np.ndarray, epsilon: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def infer_sensitivity_from_data(input_data_points: np.ndarray) -> float:
        """
        We will compute sensitivity from the dimensionality of the data
        It is simply the second dimension - min value is 0, max is 1, so sensitivity = (max-min) * n-dims

        :param input_data_points: 2-dim array
        :return: sensitivity
        """
        sensitivity = input_data_points.shape[1]
        print("Sensitivity\n", sensitivity)
        return sensitivity

    @property
    def name(self) -> str:
        return self.__class__.__name__


class NoPrivacyMechanism(RealVectorDPMechanism):
    """
    Simply copy input to output without any privatization; this must lead to infinite privacy loss
    """

    def execute(self, query_output: np.ndarray, epsilon: float) -> np.ndarray:
        return query_output


class CompletelyRandomMechanism(RealVectorDPMechanism):
    """
    Randomly returns every single value with 0.5 probability regardless of the input
    """

    def execute(self, query_output: np.ndarray, epsilon: float) -> np.ndarray:
        return np.random.rand(*query_output.shape).round().astype(int)


class LaplaceMechanism(RealVectorDPMechanism):

    @staticmethod
    def laplace_inv_cdf_correct(no_samples: int) -> np.ndarray:
        # print("no_samples:", no_samples)
        _x = np.random.uniform(0, 1, no_samples)
        return - np.sign(_x - 0.5) * np.log(1 - 2 * np.abs(_x - 0.5))

    def execute(self, input_data_points: np.ndarray, epsilon: float) -> np.ndarray:
        # assert 2-dim array: list of n-dimensional data points
        assert input_data_points.ndim == 2

        sensitivity = self.infer_sensitivity_from_data(input_data_points)

        # we will sample a 1-d vector of list_size x dimensionality, will be "reshaped" back
        total_number_of_samples_required = input_data_points.size
        # print(total_number_of_samples_required)

        # draw zero-mean samples using inverse CDF and reshape to match input data
        zero_mean_samples = self.laplace_inv_cdf_correct(total_number_of_samples_required).reshape(
            input_data_points.shape
        )
        # print(zero_mean_samples)

        # scale b = sensitivity over epsilon
        b = sensitivity / epsilon

        # rescale and add; we'll utilize broadcasting
        return input_data_points + b * zero_mean_samples


class LaplaceMechanismWrongSensitivity(LaplaceMechanism):

    def execute(self, input_data_points: np.ndarray, epsilon: float) -> np.ndarray:
        sensitivity = super().execute(input_data_points, epsilon)
        # make it 10-times smaller
        return 0.5 * sensitivity


class LaplaceMechanismADePT(LaplaceMechanism):

    def execute(self, input_data_points: np.ndarray, epsilon: float) -> np.ndarray:
        # assert 2-dim array: list of n-dimensional data points
        assert input_data_points.ndim == 2

        # ADePT's sensitivity was 2C -- from -1, to +1; so it's 1
        sensitivity = 1

        # we will sample a 1-d vector of list_size x dimensionality, will be "reshaped" back
        total_number_of_samples_required = input_data_points.size
        # print(total_number_of_samples_required)

        # draw zero-mean samples using inverse CDF and reshape to match input data
        zero_mean_samples = self.laplace_inv_cdf_correct(total_number_of_samples_required).reshape(
            input_data_points.shape
        )
        # print(zero_mean_samples)

        # scale b = sensitivity over epsilon
        b = sensitivity / epsilon

        # rescale and add; we'll utilize broadcasting
        return input_data_points + b * zero_mean_samples


class BeigiEtAlLaplaceMechanism(RealVectorDPMechanism):

    @staticmethod
    def laplace_inv_cdf_dptext(no_samples: int) -> np.ndarray:
        _x = np.random.uniform(0, 1, no_samples)
        result = - np.sign(_x) * np.log(1 - 2 * np.abs(_x))

        # print("with nan\n", result)

        # but we need to fix NaNs -- replace with zero
        np.nan_to_num(result, False, nan=0.0)
        # print("after nan_to_0\n", result)

        return result

    def execute(self, input_data_points: np.ndarray, epsilon: float) -> np.ndarray:
        # we will sample a 1-d vector of list_size x dimensionality, will be "reshaped" back
        total_number_of_samples_required = input_data_points.size

        # draw zero-mean samples using inverse CDF and reshape to match input data
        zero_mean_samples = self.laplace_inv_cdf_dptext(total_number_of_samples_required).reshape(
            input_data_points.shape
        )

        # We will to compute sensitivity from the dimensionality of the data
        # It is simply the second dimension - min value is 0, max is 1, so sensitivity = (max-min) * n-dims
        sensitivity = self.infer_sensitivity_from_data(input_data_points)

        # scale b = sensitivity over epsilon
        b = sensitivity / epsilon

        # rescale and add
        return input_data_points + b * zero_mean_samples


class Experiment:

    @staticmethod
    def privatize_data_points(input_data_points: np.ndarray, epsilon: float,
                              mechanism: RealVectorDPMechanism) -> np.ndarray:
        # we have two dimensions: 0-axis are individual data points; 1-axis each data point secret values vector
        assert input_data_points.ndim == 2

        # print(input_data_points.shape)
        print("Input data points\n", input_data_points)

        privatized = mechanism.execute(input_data_points, epsilon)
        print("Privatized\n", privatized)
        # double check output from DP mechanism
        assert input_data_points.shape == privatized.shape

        # round and truncate
        truncated = np.where(privatized < 0.5, 0, 1)
        print("Truncated\n", truncated)
        # assert all values are either ones or zeros
        assert np.min(truncated) >= 0
        assert np.max(truncated) <= 1

        return truncated

    @staticmethod
    def compute_loss_from_frequency_matrix(frequency_matrix: np.ndarray) -> float:
        print(frequency_matrix)

        # for all Y, estimate Pr(D | Y) / Pr(D' | Y)
        marginal_sum_y0 = np.sum(frequency_matrix[:, 0])
        marginal_sum_y1 = np.sum(frequency_matrix[:, 1])

        cond_d_given_y0 = frequency_matrix[0, 0] / marginal_sum_y0  # this is also precision for class D
        cond_d_prime_given_y0 = frequency_matrix[1, 0] / marginal_sum_y0  # this is also precision for class D'

        print("Some cond probs", cond_d_given_y0, cond_d_prime_given_y0)

        if cond_d_prime_given_y0 == 0:
            # 100% precision of reconstruction = infinity privacy loss
            privacy_loss_y0 = np.infty
        else:
            privacy_loss_y0 = np.max([np.log(cond_d_given_y0) - np.log(cond_d_prime_given_y0),
                                      (np.log(cond_d_prime_given_y0) - np.log(cond_d_given_y0))])
        print("privacy loss Y0", privacy_loss_y0)

        cond_d_given_y1 = frequency_matrix[0, 1] / marginal_sum_y1
        cond_d_prime_given_y1 = frequency_matrix[1, 1] / marginal_sum_y1

        if cond_d_given_y1 == 0:
            # 100% precision of reconstruction = infinity privacy loss
            privacy_loss_y1 = np.infty
        else:
            privacy_loss_y1 = np.max([np.log(cond_d_given_y1) - np.log(cond_d_prime_given_y1),
                                      (np.log(cond_d_prime_given_y1) - np.log(cond_d_given_y1))])
        print("privacy loss Y1", privacy_loss_y1)

        empirical_loss = np.max([privacy_loss_y0, privacy_loss_y1])
        print("Estimated empirical loss", empirical_loss)

        return empirical_loss

    @staticmethod
    def estimate_empirical_loss(dimensionality, epsilon, mechanism: RealVectorDPMechanism,
                                number_of_repeats=10_000_000) -> float:
        """

        :param dimensionality:
        :param epsilon:
        :param mechanism:
        :param number_of_repeats: instead of repeating N times with the same instance, we create
        them at once to utilize numpy vectorization

        :return:
        """
        frequency_matrix = np.zeros(shape=(2, 2), dtype=int)

        actual_number_of_data_points = number_of_repeats
        repetitions = 1

        # by default, 10M samples is great, but for bigger dimensions, it will fails on memory
        if dimensionality > 16:
            repetitions = 10
            actual_number_of_data_points = 1_000_000

        for _ in range(repetitions):
            # create D and D' (where D = 0; and D' = 1)
            d_data_points = np.zeros(shape=(actual_number_of_data_points, dimensionality))
            dp_d_output = Experiment.privatize_data_points(d_data_points, epsilon, mechanism)

            # turn DP output vector (mixed 0 and 1) into either all zeros or all ones -- reconstruct the original vector
            reconstructed_d = Experiment.reconstruct_original_vector(dp_d_output)

            # count how many correct zeros and ones were reconstructed
            frequency_matrix[0, 0] = np.sum(np.where(reconstructed_d == 0, 1, 0))
            frequency_matrix[0, 1] = np.sum(np.where(reconstructed_d == 1, 1, 0))

            d_prime_data_points = np.ones(shape=(actual_number_of_data_points, dimensionality))
            dp_d_prime_output = Experiment.privatize_data_points(d_prime_data_points, epsilon, mechanism)
            reconstructed_d_prime = Experiment.reconstruct_original_vector(dp_d_prime_output)

            frequency_matrix[1, 0] = np.sum(np.where(reconstructed_d_prime == 0, 1, 0))
            frequency_matrix[1, 1] = np.sum(np.where(reconstructed_d_prime == 1, 1, 0))

        return Experiment.compute_loss_from_frequency_matrix(frequency_matrix)

    @staticmethod
    def save_results(results_dict: dict) -> None:
        with open('results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
            f.flush()
            f.close()

    @staticmethod
    def main():
        np.random.seed(1234)

        # collecting results into JSON
        results_dict = {}

        for mechanism in (LaplaceMechanismADePT(), LaplaceMechanismWrongSensitivity(),
                          LaplaceMechanism(), NoPrivacyMechanism(),
                          CompletelyRandomMechanism(),
                          BeigiEtAlLaplaceMechanism(),):
            for epsilon in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
                for dimensionality in (1, 2, 8, 32, 64, 128):
                    empirical_loss = Experiment.estimate_empirical_loss(dimensionality, epsilon, mechanism)

                    # update dict
                    if mechanism.name not in results_dict:
                        results_dict[mechanism.name] = dict()
                    if epsilon not in results_dict[mechanism.name]:
                        results_dict[mechanism.name][epsilon] = dict()
                    results_dict[mechanism.name][epsilon][dimensionality] = empirical_loss

                    # save results
                    Experiment.save_results(results_dict)

    @staticmethod
    def main_dptext_test():
        np.random.seed(1234)

        # collecting results into JSON
        results_dict = {}

        for mechanism in (BeigiEtAlLaplaceMechanism(),):
            for epsilon in (1.0,):
                for dimensionality in (1,):
                    empirical_loss = Experiment.estimate_empirical_loss(dimensionality, epsilon, mechanism,
                                                                        number_of_repeats=20)

                    # update dict
                    if mechanism.name not in results_dict:
                        results_dict[mechanism.name] = dict()
                    if epsilon not in results_dict[mechanism.name]:
                        results_dict[mechanism.name][epsilon] = dict()
                    results_dict[mechanism.name][epsilon][dimensionality] = empirical_loss

                    # save results
                    Experiment.save_results(results_dict)

    @staticmethod
    def reconstruct_original_vector(privatized_vectors: np.ndarray) -> np.ndarray:

        # Now for each data point we use simple "majority voting" to turn a vector of (0, 0, 1, 0, 1) into
        # (0, 0, 0, 0, 0) and vice versa; we do it by computing mean and rounding

        # compute mean along the 2-nd axis (which will be somewhere between 0 and 1)
        mean_value_for_each_data_point = np.mean(privatized_vectors, axis=1)
        print("Mean value\n", mean_value_for_each_data_point)

        # round the mean value
        reconstructed_value_for_each_data_point = np.round(mean_value_for_each_data_point).astype(int)

        # this is now a 1-D vector - size of all data points; we need to extend it back to 2-D by repeating the
        # reconstructed value for all columns, e.g.
        # [0, 1, 1] ->
        # [0, 0, 0, 0, 0]
        # [1, 1, 1, 1, 1]
        # [1, 1, 1, 1, 1]
        # (if the input dimensionality were 5)
        assert reconstructed_value_for_each_data_point.ndim == 1

        # make it a column array
        expanded = np.expand_dims(reconstructed_value_for_each_data_point, axis=1)
        print("expanded", expanded)
        # and copy each value to all columns
        result = np.tile(expanded, (1, privatized_vectors.shape[1]))
        print(result)
        assert result.shape == privatized_vectors.shape

        print("result\n", result)
        return result

    @staticmethod
    def estimate_errors():
        # collecting results
        empirical_losses = []

        for i in range(100):
            for mechanism in (LaplaceMechanismADePT(),):
                for epsilon in (0.1,):
                    for dimensionality in (2,):
                        np.random.seed(i)
                        empirical_loss = Experiment.estimate_empirical_loss(dimensionality, epsilon, mechanism)
                        empirical_losses.append(empirical_loss)
                        print("Empirical losses")
                        # print(empirical_losses)
                        print(np.mean(empirical_losses))
                        print(np.std(empirical_losses))
                        print(len(empirical_losses))


if __name__ == '__main__':
    Experiment.main()
    Experiment.estimate_errors()
