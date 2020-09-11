"""
Some useful functions to parse and preprocess texts
"""

from nltk import tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
import math

SYMBOLS = [',', '.', '!', '@', '#', '$', '%', '^', '&', '*', '-', '_', '+', '=']
STOP_WORDS = []


def read_from_file(text: str) -> str:
    result = ''
    s = open(text, encoding='utf-8', errors='coerce')
    for line in s.readlines():
        result += line
    return result


def split_text_by_template(text: str, template: str) -> list:
    return text.split(template)


def find_number_of_occurrences_of_template(text: str, template: str) -> int:
    return text.count(template)


def tokenize_text(text: str) -> list:
    return tokenize.word_tokenize(text.lower())


def count_tokens(tokens: list) -> int:
    return len(tokens)


def count_unique_types(tokens: list) -> list:
    unique = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return unique


def lemmatize_text(text: str) -> list:
    lemmatizer = Mystem()
    return tokenize_text(''.join(lemmatizer.lemmatize(text)))


def remove_tokenize_symbols_and_other_trash(tokens: str or list) -> str or list:
    if isinstance(tokens, str):
        tokens = tokenize_text(tokens)
    result = []
    for token in tokens:
        if token.isalpha():
            result.append(token)
    return result


def count_sentences_in_text(text: str) -> int:
    return len(tokenize.sent_tokenize(text.lower()))


def count_frequencies(tokens: list) -> dict:
    freq_dict = {}
    for token in tokens:
        if token not in freq_dict:
            freq_dict[token] = 1
        else:
            freq_dict[token] += 1
    return freq_dict


def get_tf_idf_vectors(corpus: str or list):
    tf_idf = TfidfVectorizer()
    return tf_idf.fit_transform(corpus)


def get_tf_idf_counts(texts: str) -> dict:
    text_list = []
    for text in texts:
        text = lemmatize_text(text)
        text = remove_tokenize_symbols_and_other_trash(text)
        text_list.append(text)
    tf_list = []
    for text in text_list:
        tf_list.append([])
        for token in text:
            if token not in tf_list[-1]:
                tf_list[-1].append((token, text.count(token)/len(text)))

    list_unique_tokens = count_unique_types(
        remove_tokenize_symbols_and_other_trash(
            lemmatize_text(str(texts))))

    collection_size = len(text_list)

    idf_dict = {}

    for token in list_unique_tokens:
        token_count = 0
        for text in text_list:
            if token in text:
                token_count += 1
        idf_dict[token] = math.log(collection_size/token_count)

    for text in tf_list:
        for token in text:
            try:
                print(f'Word is: {token[0]}; Tf-Idf is: {idf_dict[token[0]]*token[1]}')
            except KeyError:
                print(f'Word: {token[0]} is not found')
