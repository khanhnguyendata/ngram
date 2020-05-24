from nltk import ngrams
from math import log2, isclose


def get_tokenized_sentences(tokenized_file_name, padding=0):
    with open(tokenized_file_name) as file_handle:
        sentences = file_handle.read().splitlines()
        for sentence in sentences:
            if sentence:
                tokenized_sentence = sentence.split(',')
                if tokenized_sentence:
                    tokenized_sentence = ['<S>']*padding + tokenized_sentence
                yield tokenized_sentence


class WordCounter:
    def __init__(self, sentence_generator):
        self.sentence_generator = sentence_generator
        self.sentence_count = 0
        self.token_count = 0
        self.all_ngram_counts = {}
        for ngram_length in range(1, 6):
            self.all_ngram_counts[ngram_length] = {}

        for sentence in self.sentence_generator:
            if sentence:
                self.sentence_count += 1
            for token in sentence:
                if token != '<S>':
                    self.token_count += 1
            for ngram_length in range(1, 6):
                ngram_counts = self.all_ngram_counts[ngram_length]
                for i, sentence_ngram in enumerate(ngrams(sentence, ngram_length)):
                    ngram_count = ngram_counts.setdefault(sentence_ngram, {'start': 0, 'all': 0})
                    if i == 0:
                        ngram_count['start'] += 1
                    ngram_count['all'] += 1


class UnigramModel():
    def __init__(self, train, k=1):
        self.k = k
        self.train_unigram_counts = train.all_ngram_counts[1].copy()
        self.train_unigrams = set(self.train_unigram_counts.keys())
        self.train_unigram_vocab = len(self.train_unigrams)
        self.train_unigram_counts[('<UNK>',)] = {'all': 0, 'start': 0}

        self.train_prob_denom = train.token_count + len(self.train_unigram_counts) * self.k
        self.train_prob_noms = {}
        self.train_probs = {}

        for unigram, unigram_count in self.train_unigram_counts.items():
            prob_nom = self.train_unigram_counts[unigram]['all'] + self.k
            self.train_prob_noms[unigram] = prob_nom
            prob = prob_nom / self.train_prob_denom
            self.train_probs[unigram] = prob
        self.train_probs[('<UNK>',)] = self.k / self.train_prob_denom
        assert isclose(sum(prob for prob in self.train_probs.values()), 1, rel_tol=1e-5)

    def calculate_avg_ll(self, test):
        self.test_ll = 0
        self.test_unigram_counts = test.all_ngram_counts[1].copy()
        self.test_modified_unigram_counts = {}
        self.test_unigram_infos = {}

        for unigram, unigram_count in self.test_unigram_counts.items():
            if unigram not in self.train_unigrams:
                unigram = (('<UNK>',))
            unigram_train_prob = self.train_probs[unigram]
            unigram_test_count = unigram_count['all']
            unigram_ll = unigram_test_count * log2(unigram_train_prob)
            self.test_ll += unigram_ll

            # Tracking relevant information when going through test set
            self.test_modified_unigram_counts[unigram] = self.test_modified_unigram_counts.get(unigram, 0) + unigram_count['all']
            self.test_unigram_infos[unigram] = self.test_unigram_infos.get(unigram, {})
            if 'log' not in self.test_unigram_infos[unigram]:
                self.test_unigram_infos[unigram]['log'] = log2(unigram_train_prob)
            self.test_unigram_infos[unigram]['count'] = self.test_unigram_infos[unigram].get('count', 0) + unigram_test_count


        assert sum(self.test_modified_unigram_counts.values()) == test.token_count
        assert isclose(self.test_ll, sum(info['count'] * info['log'] for unigram, info in self.test_unigram_infos.items()), rel_tol=1)

        self.avg_test_ll = self.test_ll / test.token_count
        return self.avg_test_ll