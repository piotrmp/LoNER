from data_processing.into_bert import PTC_categories_dict, PTC_dict_with_None, BIO_dict, IOBES_dict
import tensorflow as tf
import numpy as np
from itertools import groupby

def vectorLabel2strings(vector_label, label_dict=PTC_categories_dict):
    '''Przerabiamy wektorowe etykiety na słowne dla tokenu'''
    labels_list = list(label_dict.keys())
    if 1 in vector_label:
        label_ids = [i for i, value in enumerate(vector_label) if value == 1]
        token_string_labels = [labels_list[i] for i in label_ids]
    else:
        token_string_labels = ['Outside']
    return token_string_labels

def propagandaID2string(category_id, label_dict = PTC_dict_with_None):
    categories = list(label_dict.keys())
    label = categories[category_id]
    return [label]

def propagandaBIO_ID2string(category_id, label_dict = BIO_dict):
    categories = list(label_dict.keys())
    label = categories[category_id]
    if label != 'None':
        label = label[2:]
    return [label]

def propagandaIOBES_ID2string(category_id, label_dict = IOBES_dict):
    categories = list(label_dict.keys())
    label = categories[category_id]
    if label != 'None':
        label = label[2:]
    return [label]


def stick_preds2tokens(sentence_tokens, sentence_string_labels):
    '''Dopasowujemy liste etykiet do listy tokenów'''
    assert len(sentence_tokens) == len(sentence_string_labels)
    #string_labels = [vectorLabel2strings(label) for label in sentence_labels]
    labeled_list = [[tok,tags] for tok,tags in zip(sentence_tokens, sentence_string_labels)]
    
    return labeled_list
    
def array_SLpreds2lists(list_of_array_preds, label_mask):
    '''Bierzemy liste z predykcjami dla batchy (w arrayach), sklejamy ją w jedną całość,
       , przerabiamy softmaxy na etkiety [0001101...] i wybieramy tylko tokeny, które nie są
       [PAD], [SEP] czy [CLS]. Zwracamy pojedynczą listę z tagami dla kolejnych słów bez podziału na zdania'''
    concatenated_preds = np.concatenate(list_of_array_preds)
    argmaxed_preds = np.argmax(concatenated_preds, axis=-1)
    token_preds = tf.boolean_mask(argmaxed_preds, label_mask).numpy().tolist()
    string_preds = [propagandaID2string(pred) for pred in token_preds]
    return string_preds

def array_BIOpreds2lists(list_of_array_preds, label_mask):
    '''Bierzemy liste z predykcjami dla batchy (w arrayach), sklejamy ją w jedną całość,
       , przerabiamy softmaxy na etkiety [0001101...] i wybieramy tylko tokeny, które nie są
       [PAD], [SEP] czy [CLS]. Zwracamy pojedynczą listę z tagami dla kolejnych słów bez podziału na zdania'''
    concatenated_preds = np.concatenate(list_of_array_preds)
    argmaxed_preds = np.argmax(concatenated_preds, axis=-1)
    token_preds = tf.boolean_mask(argmaxed_preds, label_mask).numpy().tolist()
    string_preds = [propagandaBIO_ID2string(pred) for pred in token_preds]
    return string_preds

def array_IOBESpreds2lists(list_of_array_preds, label_mask):
    '''Bierzemy liste z predykcjami dla batchy (w arrayach), sklejamy ją w jedną całość,
       , przerabiamy softmaxy na etkiety [0001101...] i wybieramy tylko tokeny, które nie są
       [PAD], [SEP] czy [CLS]. Zwracamy pojedynczą listę z tagami dla kolejnych słów bez podziału na zdania'''
    concatenated_preds = np.concatenate(list_of_array_preds)
    argmaxed_preds = np.argmax(concatenated_preds, axis=-1)
    token_preds = tf.boolean_mask(argmaxed_preds, label_mask).numpy().tolist()
    string_preds = [propagandaIOBES_ID2string(pred) for pred in token_preds]
    return string_preds

def array_preds2lists(list_of_array_preds, label_mask):
    '''Bierzemy liste z predykcjami dla batchy (w arrayach), sklejamy ją w jedną całość,
       , przerabiamy softmaxy na etkiety [0001101...] i wybieramy tylko tokeny, które nie są
       [PAD], [SEP] czy [CLS]. Zwracamy pojedynczą listę z tagami dla kolejnych słów bez podziału na zdania'''
    concatenated_preds = np.concatenate(list_of_array_preds)
    argmaxed_preds = np.argmax(concatenated_preds, axis=-1)
    token_preds = tf.boolean_mask(argmaxed_preds, label_mask).numpy().tolist()
    string_preds = [vectorLabel2strings(pred) for pred in token_preds]
    return string_preds


def array_CRFpreds2lists(list_of_array_CRF_preds, label_mask):

    concatenated_preds = np.concatenate(list_of_array_CRF_preds)
    token_preds = tf.boolean_mask(concatenated_preds, label_mask).numpy().tolist()
    string_preds = [propagandaID2string(pred) for pred in token_preds]
    return string_preds

def array_CRF_BIOpreds2lists(list_of_array_CRF_BIO_preds, label_mask):

    concatenated_preds = np.concatenate(list_of_array_CRF_BIO_preds)
    token_preds = tf.boolean_mask(concatenated_preds, label_mask).numpy().tolist()
    string_preds = [propagandaBIO_ID2string(pred) for pred in token_preds]
    return string_preds

def array_CRF_IOBESpreds2lists(list_of_array_CRF_IOBES_preds, label_mask):

    concatenated_preds = np.concatenate(list_of_array_CRF_IOBES_preds)
    token_preds = tf.boolean_mask(concatenated_preds, label_mask).numpy().tolist()
    string_preds = [propagandaIOBES_ID2string(pred) for pred in token_preds]
    return string_preds

def merge_subtokens(span_tuple_list):
    span_list = [list(i) for i in span_tuple_list]
    span_list.sort(key=lambda interval: interval[0])
    merged = [span_list[0]]
    for current in span_list:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged

def merge_neighbour_spans(span_tuple_list):
    span_list = [list(i) for i in span_tuple_list]
    span_list.sort(key=lambda interval: interval[0])
    merged = [span_list[0]]
    for current in span_list:
        previous = merged[-1]
        if current[0] <= previous[1] or current[0]  <= previous[1] + 1:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged

def group_preds(tokens_with_preds):
    token_groups = []
    tokens_with_preds = sorted(tokens_with_preds, key= lambda x: x[0][3])
    for key, group in groupby(tokens_with_preds, lambda x: x[0][3]):
        token_groups.append(sorted(list(group), key= lambda x: x[0][1]))
    return token_groups

def list2dict(sorted_preds_list, merge_spans = False):
    preds_dict = {}
    preds_dict['artID'] = sorted_preds_list[0][0][3]
    for i in sorted_preds_list:
        for j in i[1]:
            if j not in preds_dict.keys() and j != 'Outside' and j != 'None':
                preds_dict[j] = [(i[0][1], i[0][2])]
            elif j in preds_dict.keys() and j != 'Outside' and j != 'None':
                preds_dict[j].append((i[0][1], i[0][2]))
    for prop_cat in list(preds_dict.keys()):
        if prop_cat != 'artID':
            if merge_spans:
                preds_dict[prop_cat] = merge_neighbour_spans(preds_dict[prop_cat])
            else:
                preds_dict[prop_cat] = merge_subtokens(preds_dict[prop_cat])
    
    return preds_dict

def save_dicts2txt(article_dicts_list, save_path):
    with open(save_path, 'w') as f:
        for article_dict in article_dicts_list:
            for category in list(article_dict.keys()):
                if category != 'artID':
                    for span in article_dict[category]:
                        f.write(str(article_dict['artID']))
                        f.write('\t')
                        f.write(str(category))
                        f.write('\t')
                        f.write(str(span[0]))
                        f.write('\t')
                        f.write(str(span[1]))
                        f.write('\n')
    return None

def preds2semeval(predictions, output_path, merge_spans=True):
    '''Drukujemy predykcje do plku txt w formacie semevalovym'''
    grouped_articles = group_preds(predictions)
    article_dicts = [list2dict(article, merge_spans=merge_spans)
                     for article in grouped_articles]
    text_preds = save_dicts2txt(article_dicts, output_path)
    
    return text_preds # czyli None


def rawPreds2semeval(arrayed_logits_list, label_masks_array, token_metas, output_path, merge_spans = True):
    '''Od predykcji w liście logitów do formatu semevalowego w txt'''
    concatenated_preds = array_preds2lists(arrayed_logits_list, label_masks_array) # na tym etapie tokeny dostają swoje spany (start, end)
    preds_with_spans = stick_preds2tokens(token_metas, concatenated_preds)
    semeval_preds = preds2semeval(preds_with_spans, output_path, merge_spans=merge_spans)
    return semeval_preds # None

def CRFpreds2semeval(decoded_sequences_list, label_masks_array, token_metas, output_path, merge_spans = True):

    concatenated_preds = array_CRFpreds2lists(decoded_sequences_list, label_masks_array)
    preds_with_spans = stick_preds2tokens(token_metas, concatenated_preds)
    semeval_preds = preds2semeval(preds_with_spans, output_path, merge_spans=merge_spans)
    return semeval_preds

def SingleLabelPreds2semeval(arrayed_logits_list, label_masks_array, token_metas, output_path, merge_spans = True):

    concatenated_preds = array_SLpreds2lists(arrayed_logits_list, label_masks_array)
    preds_with_spans = stick_preds2tokens(token_metas, concatenated_preds)
    semeval_preds = preds2semeval(preds_with_spans, output_path, merge_spans=merge_spans)
    return semeval_preds

def BIOPreds2semeval(arrayed_logits_list, label_masks_array, token_metas, output_path, merge_spans = True):

    concatenated_preds = array_BIOpreds2lists(arrayed_logits_list, label_masks_array)
    preds_with_spans = stick_preds2tokens(token_metas, concatenated_preds)
    semeval_preds = preds2semeval(preds_with_spans, output_path, merge_spans=merge_spans)
    return semeval_preds


def IOBESPreds2semeval(arrayed_logits_list, label_masks_array, token_metas, output_path, merge_spans = True):

    concatenated_preds = array_IOBESpreds2lists(arrayed_logits_list, label_masks_array)
    preds_with_spans = stick_preds2tokens(token_metas, concatenated_preds)
    semeval_preds = preds2semeval(preds_with_spans, output_path, merge_spans=merge_spans)
    return semeval_preds

def CRF_BIO_preds2semeval(decoded_sequences_list, label_masks_array, token_metas, output_path, merge_spans = True):

    concatenated_preds = array_CRF_BIOpreds2lists(decoded_sequences_list, label_masks_array)
    preds_with_spans = stick_preds2tokens(token_metas, concatenated_preds)
    semeval_preds = preds2semeval(preds_with_spans, output_path, merge_spans=merge_spans)
    return semeval_preds

def CRF_IOBES_preds2semeval(decoded_sequences_list, label_masks_array, token_metas, output_path, merge_spans = True):

    concatenated_preds = array_CRF_IOBESpreds2lists(decoded_sequences_list, label_masks_array)
    preds_with_spans = stick_preds2tokens(token_metas, concatenated_preds)
    semeval_preds = preds2semeval(preds_with_spans, output_path, merge_spans=merge_spans)
    return semeval_preds


