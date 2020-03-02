def get_tokenized_sentences(tokenized_file_name):
    with open(tokenized_file_name) as file_handle:
        sentences = file_handle.read().splitlines()
        for sentence in sentences:
            if sentence:
                tokenized_sentences = sentence.split(',')
                yield tokenized_sentences