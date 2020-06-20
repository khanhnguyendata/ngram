from analysis.preprocess import get_tokenized_sentences
from math import log2


class UnigramCounter:
    def __init__(self, file_name: str) -> None:
        """
        Initialize unigram counter from tokenized text and count number of unigrams in text
        :param file_name: path of tokenized text. Each line is a sentence with tokens separated by comma.
        """
        self.sentence_generator = get_tokenized_sentences(file_name)
        self.count()

    def count(self) -> None:
        """
        Count number of unigrams in text, one sentence at a time
        """
        self.sentence_count = 0
        self.token_count = 0
        self.counts = {}

        for sentence in self.sentence_generator:
            self.sentence_count += 1
            self.token_count += len(sentence)
            for unigram in sentence:
                self.counts[unigram] = self.counts.get(unigram, 0) + 1


class UnigramModel:
    def __init__(self, train_counter: UnigramCounter) -> None:
        """
        Initialize unigram model from unigram counter, count the number of unique unigrams (vocab)
        :param train_counter: counted unigram counter
        """
        self.counter = train_counter
        self.counts = train_counter.counts.copy()
        self.counts['[UNK]'] = 0
        self.vocab = set(self.counts.keys())
        self.vocab_size = len(self.vocab)

    def train(self, k: int = 1) -> None:
        """
        For each unigram in the vocab, calculate its probability in the text
        :param k: smoothing pseudo-count for each unigram
        """
        self.probs = {}
        for unigram, unigram_count in self.counts.items():
            prob_nom = unigram_count + k
            prob_denom = self.counter.token_count + k * self.vocab_size
            self.probs[unigram] = prob_nom / prob_denom

    def evaluate(self, evaluation_counter: UnigramCounter) -> float:
        """
        Calculate the average log likelihood of the model on the evaluation text
        :param evaluation_counter: unigram counter for the text on which the model is evaluated on
        :return: average log likelihood that the unigram model assigns to the evaluation text
        """
        test_log_likelihood = 0
        test_counts = evaluation_counter.counts

        for unigram, test_count in test_counts.items():
            if unigram not in self.vocab:
                unigram = '[UNK]'
            train_prob = self.probs[unigram]
            log_likelihood = test_count * log2(train_prob)
            test_log_likelihood += log_likelihood

        avg_test_log_likelihood = test_log_likelihood / evaluation_counter.token_count
        return avg_test_log_likelihood
