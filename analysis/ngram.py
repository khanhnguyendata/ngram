import numpy as np
from typing import List, Tuple, Callable, Union
from nltk.util import ngrams
from analysis.preprocess import get_tokenized_sentences


class NgramCounter:
    def __init__(self, file_name: str) -> None:
        """
        Initialize n-gram counter from tokenized text and count number of n-grams in text
        :param file_name: path of tokenized text. Each line is a sentence with tokens separated by comma.
        """
        self.sentence_generator = get_tokenized_sentences(file_name)
        self.count()

    def count(self) -> None:
        """
        Count number of n-grams in text (both overall and starting n-grams), one sentence at a time
        """
        self.sentence_count = 0
        self.token_count = 0
        self.counts = {}
        self.start_counts = {}

        for sentence in self.sentence_generator:
            self.sentence_count += 1
            self.token_count += len(sentence)

            for ngram_length in range(1, 6):
                for ngram_position, ngram in enumerate(ngrams(sentence, ngram_length)):
                    if ngram_position == 0:
                        self.start_counts[ngram] = self.start_counts.get(ngram, 0) + 1
                    self.counts[ngram] = self.counts.get(ngram, 0) + 1


class NgramModel:
    def __init__(self, train_counter: NgramCounter) -> None:
        """
        Initialize unigram model from unigram counter, count the number of unique unigrams (vocab)
        :param train_counter: counted n-gram counter
        """
        self.counter = train_counter
        self.counts = train_counter.counts
        self.start_counts = train_counter.start_counts
        self.vocab_size = len(list(ngram for ngram in self.counts.keys() if len(ngram) == 1))
        self.uniform_prob = 1 / (self.vocab_size + 1)

    def calculate_unigram_prob(self, unigram: Tuple[str]) -> None:
        """
        Calculate conditional probability for a unigram
        :param unigram: length-1 tuple containing the unigram
        """
        if unigram in self.start_counts:
            prob_nom = self.start_counts[unigram]
            prob_denom = self.counter.sentence_count
            self.start_probs[unigram] = prob_nom / prob_denom

        prob_nom = self.counts[unigram]
        prob_denom = self.counter.token_count
        self.probs[unigram] = prob_nom / prob_denom

    def calculate_multigram_prob(self, ngram: Tuple[str]) -> None:
        """
        Calculate conditional probability for higher n-gram (multigram)
        :param ngram: tuple containing words of the n-gram
        """
        prevgram = ngram[:-1]
        if ngram in self.start_counts:
            prob_nom = self.start_counts[ngram]
            prob_denom = self.start_counts[prevgram]
            self.start_probs[ngram] = prob_nom / prob_denom

        prob_nom = self.counts[ngram]
        prob_denom = self.counts[prevgram]
        self.probs[ngram] = prob_nom / prob_denom

    def train(self) -> None:
        """
        For each n-gram, calculate its conditional probability in the training text
        """
        self.probs = {}
        self.start_probs = {}
        for ngram in self.counts:
            if len(ngram) == 1:
                self.calculate_unigram_prob(ngram)
            else:
                self.calculate_multigram_prob(ngram)

    def evaluate(self, eval_file: str) -> np.ndarray:
        """
        Evaluate trained n-gram model on evaluation text
        :param eval_file: file path of tokenized evaluation text
        :return: probability matrix of evaluation text (number of words * number of models)
        """
        eval_token_count = sum(len(sentence) for sentence in get_tokenized_sentences(eval_file))
        prob_matrix = np.zeros(shape=(eval_token_count, 6))

        # Fill in uniform probability to first column of matrix
        prob_matrix[:, 0] = self.uniform_prob

        # Fill in n-gram probabilities row-by-row to the matrix
        row = 0
        for sentence in get_tokenized_sentences(eval_file):
            for token_position, token in enumerate(sentence):
                for ngram_length in range(1, 6):
                    ngram_start = token_position + 1 - ngram_length
                    ngram_end = token_position + 1
                    # For n-gram at start of sentence (negative start position)
                    if ngram_start < 0:
                        ngram = tuple(sentence[0:ngram_end])
                        prob_matrix[row, ngram_length] = self.start_probs.get(ngram, 0)
                    # For regular n-gram (non-negative start position)
                    else:
                        ngram = tuple(sentence[ngram_start:ngram_end])
                        prob_matrix[row, ngram_length] = self.probs.get(ngram, 0)
                row += 1

        return prob_matrix


def calculate_avg_ll(prob_matrix: np.ndarray, weights: List[float] = None, log_function: Callable = np.log2) -> float:
    """
    Calculate average log likelihood from weighted combination of columns in probability matrix of evaluation text
    :param prob_matrix: probability matrix of evaluation text
    :param weights: corresponding weight of each column
    :param log_function: log function to use (often np.log2 for base 2 log, or np.log for natural log)
    :return: average log likelihood from weighted combination of columns
    """
    n_models = prob_matrix.shape[1]
    if weights is None:
        weights = np.ones(n_models) / n_models
    interpolated_probs = np.sum(prob_matrix * weights, axis=1)
    average_log_likelihood = log_function(interpolated_probs).mean()
    return average_log_likelihood


def calculate_avg_ln(prob_matrix: np.array, weights: Union[List[float], np.array] = None) -> float:
    """
    Calculate average natural log likelihood of evaluation text with given interpolation weights
    :param prob_matrix: probability matrix of n_words x n_models
    :param weights: given weights for each model
    :return: average natural log of evaluation text with given weights
    """
    return calculate_avg_ll(prob_matrix, weights, log_function=np.log)