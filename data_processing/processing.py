from data_processing.preprocessing import labels2dictOfdicts
from data_processing.preprocessing import get_article_chunks
from data_processing.preprocessing import get_labels
from data_processing.preprocessing import produce_tokens_artIDs_offsets
from data_processing.preprocessing import produce_tokens_and_labels
import os


def dict_mean(dict_list: list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = round(sum(float(d[key]) for d in dict_list) / len(dict_list), 4)
    return mean_dict


def output_artIDs_tokens_offsets(texts_path, vocab_path, sentence_length, nested=False, only_double_enter=False,
                                 normalize_encode=True, pack_sentences=True):
    '''
    Tokenize the input and pair tokens with the assigned/predicted labels. If nested = True, list of lists is returned,
    else, only one list,
    '''

    documents = os.listdir(texts_path)
    documents = [os.path.join(texts_path, i) for i in documents]

    sentences = []
    for doc in documents:
        chunked_paragraph = get_article_chunks(text_path=doc, path_to_vocab=vocab_path, max_chunk_len=sentence_length,
                                               only_double_enter=only_double_enter, pack_sentences=pack_sentences)
        for chunk in chunked_paragraph:
            chunk_sentence = produce_tokens_artIDs_offsets(chunk,
                                                           path_to_vocab=vocab_path, nested=nested,
                                                           normalize_encode=normalize_encode)
            chunk_sentence = chunk_sentence[:sentence_length - 2]
            if nested:
                sentences.append(chunk_sentence)
            else:
                sentences += chunk_sentence
    return sentences


def output_tokens_and_tags(texts_path, labels_path, vocab_path, sentence_length, nested=False, only_double_enter=False,
                           normalize_encode=True, pack_sentences=True):
    '''
     Outputs two lists: list of all the tokens from the input data, and list of their matching labels
     '''
    article_subpaths = os.listdir(texts_path)
    article_paths = [os.path.join(texts_path, i) for i in article_subpaths]
    ultimate_labels_dict = labels2dictOfdicts(labels_path)
    token_sentences = []
    label_sentences = []
    for article_path in article_paths:
        chunked_paragraph = get_article_chunks(text_path=article_path, path_to_vocab=vocab_path,
                                               max_chunk_len=sentence_length,
                                               only_double_enter=only_double_enter, pack_sentences=pack_sentences)
        labels_dict = get_labels(article_path=article_path, all_labels_dict=ultimate_labels_dict)
        for chunk in chunked_paragraph:
            chunk_sentence, chunk_labels, chunk_offsets = produce_tokens_and_labels(chunk,
                                                                                    labels_dict=labels_dict,
                                                                                    path_to_vocab=vocab_path,
                                                                                    nested=nested,
                                                                                    normalize_encode=normalize_encode)
            token_sentences.append(chunk_sentence)
            label_sentences.append(chunk_labels)

    return token_sentences, label_sentences
