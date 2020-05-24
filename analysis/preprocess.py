from nltk.tokenize import RegexpTokenizer, sent_tokenize
from typing import Iterator


def replace_characters(text: str) -> str:
    """
    Replace tricky punctuations that can mess up sentence tokenizers
    :param text: text with non-standard punctuations
    :return: text with standardized punctuations
    """
    replacement_rules = {'“': '"', '”': '"', '’': "'", '--': ','}
    for symbol, replacement in replacement_rules.items():
        text = text.replace(symbol, replacement)
    return text


def generate_tokenized_sentences(paragraph: str) -> Iterator[str]:
    """
    Tokenize each sentence in paragraph.
    For each sentence, tokenize each words and return the tokenized sentence one at a time.
    :param paragraph: text of paragraph
    """
    word_tokenizer = RegexpTokenizer(r'[-\'\w]+')

    for sentence in sent_tokenize(paragraph):
        tokenized_sentence = word_tokenizer.tokenize(sentence)
        if tokenized_sentence:
            tokenized_sentence.append('[END]')
            yield tokenized_sentence


def tokenize_raw_text(raw_text_path: str, token_text_path: str) -> None:
    """
    Read a input text file and write its content to an output text file in the form of tokenized sentences
    :param raw_text_path: path of raw input text file
    :param token_text_path: path of tokenized output text file
    """
    with open(raw_text_path) as read_handle, open(token_text_path, 'w') as write_handle:
        for paragraph in read_handle:
            paragraph = paragraph.lower()
            paragraph = replace_characters(paragraph)

            for tokenized_sentence in generate_tokenized_sentences(paragraph):
                write_handle.write(','.join(tokenized_sentence))
                write_handle.write('\n')


def get_tokenized_sentences(file_name: str) -> Iterator[str]:
    """
    Return tokenized sentence one at a time from a tokenized text
    :param file_name: path of tokenized text
    """
    with open(file_name) as file_handle:
        for sentence in file_handle.read().splitlines():
            tokenized_sentence = sentence.split(',')
            yield tokenized_sentence