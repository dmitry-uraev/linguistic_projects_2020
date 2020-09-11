"""
Here all implemented functions are used
"""

from preprocessing.preprocessing_functions import read_from_file
from preprocessing.preprocessing_functions import split_text_by_template
from preprocessing.preprocessing_functions import find_number_of_occurrences_of_template
from preprocessing.preprocessing_functions import tokenize_text
from preprocessing.preprocessing_functions import count_tokens
from preprocessing.preprocessing_functions import count_unique_types
from preprocessing.preprocessing_functions import lemmatize_text
from preprocessing.preprocessing_functions import remove_tokenize_symbols_and_other_trash
from preprocessing.preprocessing_functions import count_sentences_in_text
from preprocessing.preprocessing_functions import count_frequencies
from preprocessing.preprocessing_functions import get_tf_idf_vectors
from preprocessing.preprocessing_functions import get_tf_idf_counts


text = read_from_file('lifenews2.txt')
texts = split_text_by_template(text, '<-=->')

number_of_occurrences_3th_article = find_number_of_occurrences_of_template(texts[2], "о")
tokens_3th_article = tokenize_text(texts[2])
number_of_token_3th_article = count_tokens(tokens_3th_article)
unique_types_3th_article = count_unique_types(tokens_3th_article)
words_in_3th_article = lemmatize_text(texts[2])
words_in_3th_article = remove_tokenize_symbols_and_other_trash(words_in_3th_article)
words_number_3th_article = count_tokens(words_in_3th_article)
sentences_in_3th_article = count_sentences_in_text(texts[2])
bag_of_words_dict_3th_article = count_frequencies(tokens_3th_article)
tf_idf_vectors = get_tf_idf_vectors(texts)

get_tf_idf_counts(texts[:3])
