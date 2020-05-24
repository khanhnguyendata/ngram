import numpy as np
from typing import Dict, Tuple
from nltk.util import ngrams
from preprocess import get_tokenized_sentences


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

        for sentence in self.sentence_generator:
            self.sentence_count += 1
            self.token_count += len(sentence)

            for ngram_length in range(1, 6):
                for ngram_position, ngram in enumerate(ngrams(sentence, ngram_length)):
                    ngram_count = self.counts.setdefault(ngram, {'start': 0, 'all': 0})
                    if ngram_position == 0:
                        ngram_count['start'] += 1
                    ngram_count['all'] += 1


class NgramModel:
    def __init__(self, train_counter: NgramCounter) -> None:
        """
        Initialize unigram model from unigram counter, count the number of unique unigrams (vocab)
        :param train_counter: counted n-gram counter
        """
        self.counter = train_counter
        self.counts = train_counter.counts
        self.vocab_size = len(list(ngram for ngram in self.counts.keys() if len(ngram) == 1))
        self.uniform_prob = 1 / (self.vocab_size + 1)

    def train(self) -> None:
        """
        For each n-gram, calculate its conditional probability in the training text
        """
        def calculate_unigram_prob(unigram: Tuple[str], unigram_count: Dict[str, int]) -> None:
            """
            Calculate conditional probability for a unigram
            :param unigram: length-1 tuple containing the unigram
            :param unigram_count: count of unigram in the training text (both overall and starting count)
            """
            if unigram_count['start']:
                prob_nom = unigram_count['start']
                prob_denom = self.counter.sentence_count
                self.start_probs[unigram] = prob_nom / prob_denom

            prob_nom = unigram_count['all']
            prob_denom = self.counter.token_count
            self.probs[unigram] = prob_nom / prob_denom

        def calculate_multigram_prob(ngram: Tuple[str], ngram_count: Dict[str, int]) -> None:
            """
            Calculate conditional probability for higher n-gram (multigram)
            :param ngram: tuple containing words of the n-gram
            :param ngram_count: count of n-gram in the training text (both overall and starting count)
            """
            prevgram = ngram[:-1]
            if ngram_count['start']:
                prob_nom = ngram_count['start']
                prob_denom = self.counts[prevgram]['start']
                self.start_probs[ngram] = prob_nom / prob_denom

            prob_nom = ngram_count['all']
            prob_denom = self.counts[prevgram]['all']
            self.probs[ngram] = prob_nom / prob_denom

        self.probs = {}
        self.start_probs = {}
        for ngram, ngram_count in self.counts.items():
            if len(ngram) == 1:
                calculate_unigram_prob(ngram, ngram_count)
            else:
                calculate_multigram_prob(ngram, ngram_count)

    def evaluate(self, eval_file: str) -> np.ndarray:
        """
        Evaluate trained n-gram model on evaluation text
        :param eval_file: file path of tokenized evaluation text
        :return: probability matrix of evaluation text (number of words * number of models)
        """
        eval_token_count = sum(len(sentence) for sentence in get_tokenized_sentences(eval_file))
        eval_prob_matrix = np.zeros(shape=(eval_token_count, 6))
        eval_prob_matrix[:, 0] = self.uniform_prob

        row = 0
        for sentence in get_tokenized_sentences(eval_file):
            for token_position, token in enumerate(sentence):
                for ngram_length in range(1, 6):
                    ngram_start = token_position + 1 - ngram_length
                    ngram_end = token_position + 1
                    # For n-gram at start of sentence (negative start position)
                    if ngram_start < 0:
                        ngram = tuple(sentence[0:ngram_end])
                        eval_prob_matrix[row, ngram_length] = self.start_probs.get(ngram, 0)
                    # For regular n-gram (non-negative start position)
                    else:
                        ngram = tuple(sentence[ngram_start:ngram_end])
                        eval_prob_matrix[row, ngram_length] = self.probs.get(ngram, 0)
                row += 1

        return eval_prob_matrix
