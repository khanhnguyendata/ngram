import numpy as np
from analysis.ngram import calculate_avg_ln
from typing import List, Union


def optimize_gd(prob_matrix: np.array, learning_rate: float, n_iter: int, init_weights: List = None) -> np.array:
    """
    Optimize model weights using gradient descent
    :param init_weights: initial weights. If None, all models will have equal initial weights.
    :param prob_matrix: probability matrix of n_words x n_models.
    :param learning_rate: fraction of gradient to update model weights in each iteration
    :param n_iter: number of iterations to run gradient descent
    :return: model weights after running gradient descent
    """
    # 1. Initialize model weights
    if init_weights is not None:
        weights = np.array(init_weights)
    else:
        n_models = prob_matrix.shape[1]
        weights = np.ones(n_models) / n_models

    ngram_probs = prob_matrix[:, 1:]
    uniform_prob = prob_matrix[:, [0]]
    for iteration in range(n_iter):
        # 2. Calculate gradients for each n-gram model
        interpolated_probs = np.sum(prob_matrix * weights, axis=1, keepdims=True)
        ngram_gradients = np.mean((ngram_probs - uniform_prob) / interpolated_probs, axis=0)

        # 3. Update interpolation weights for all models
        weights[1:] += learning_rate * ngram_gradients
        weights[0] = 1 - weights[1:].sum()

    return weights


class GradientDescent:
    def __init__(self) -> None:
        """
        Implement gradient descent optimization of model weights,
        while tracking intermediate values for plotting
        """
        pass

    def fit(self, prob_matrix: np.array, learning_rate: float, n_iter: int, init_weights: List = None) -> None:
        """
        Update model weights via gradient descent and store intermediate values for plotting
        :param init_weights: initial weights. If None, all models will have equal initial weights.
        :param prob_matrix: probability matrix of n_words x n_models.
        :param learning_rate: fraction of gradient to update model weights in each iteration
        :param n_iter: number of iterations to run gradient descent
        """
        self.tracked_info = {}

        # 1. Initialize model weights
        if init_weights is not None:
            self.weights = np.array(init_weights)
        else:
            n_models = prob_matrix.shape[1]
            self.weights = np.ones(n_models) / n_models

        uniform_prob = prob_matrix[:, [0]]
        ngram_probs = prob_matrix[:, 1:]
        for iteration in range(n_iter):
            # 2. Calculate gradients for each n-gram model
            interpolated_probs = np.sum(prob_matrix * self.weights, axis=1, keepdims=True)
            gradients = np.mean((ngram_probs - uniform_prob) / interpolated_probs, axis=0)
            self.tracked_info[iteration] = {'weights': self.weights.copy(),
                                            'gradients': gradients.copy(),
                                            'avg_ll': calculate_avg_ln(prob_matrix, self.weights)}

            # 3. Update interpolation weights for all models
            self.weights[1:] += learning_rate * gradients
            self.weights[0] = 1 - self.weights[1:].sum()


def optimize_em(prob_matrix: np.array, n_iter: int, init_weights: List = None) -> np.array:
    """
    Optimize model weights using EM algorithm
    :param init_weights: initial weights. If None, all models will have equal initial weights.
    :param prob_matrix: probability matrix of n_words x n_models.
    :param n_iter: number of iterations to run EM
    :return: model weights after running EM
    """
    # 1. Initialize model weights
    if init_weights is not None:
        weights = np.array(init_weights)
    else:
        n_models = prob_matrix.shape[1]
        weights = np.ones(n_models) / n_models

    for iteration in range(n_iter):
        # 2. E-step: calculate posterior probabilities from current model weights
        weighted_probs = prob_matrix * weights
        total_probs = weighted_probs.sum(axis=1, keepdims=True)
        posterior_probs = weighted_probs / total_probs

        # 3. M-step: update model weights using posterior probabilities from E-step
        weights = posterior_probs.mean(axis=0)

    return weights


class EM:
    def __init__(self) -> None:
        """
        Implement gradient descent optimization of model weights,
        while tracking intermediate values for plotting
        """
        pass

    def fit(self, prob_matrix: np.array, n_iter: int, init_weights: List = None) -> None:
        """
        Update model weights via gradient descent and store intermediate values for plotting
        :param init_weights: initial weights. If None, all models will have equal initial weights.
        :param prob_matrix: probability matrix of n_words x n_models.
        :param n_iter: number of iterations to run gradient descent
        """
        self.tracked_info = {}

        # 1. Initialize model weights
        if init_weights is not None:
            self.weights = np.array(init_weights)
        else:
            n_models = prob_matrix.shape[1]
            self.weights = np.ones(n_models) / n_models

        uniform_prob = prob_matrix[:, [0]]
        ngram_probs = prob_matrix[:, 1:]
        for iteration in range(n_iter):
            # 2. E-step: calculate posterior probabilities from current model weights
            weighted_probs = prob_matrix * self.weights
            total_probs = weighted_probs.sum(axis=1, keepdims=True)
            posterior_probs = weighted_probs / total_probs

            self.tracked_info[iteration] = {'weights': self.weights,
                                            'avg_ll': calculate_avg_ln(prob_matrix, self.weights)}

            # 3. M-step: update model weights using posterior probabilities from E-step
            self.weights = posterior_probs.mean(axis=0)