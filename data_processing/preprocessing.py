import os, re, unicodedata
from itertools import groupby
from tensorflow_text.python.ops import BertTokenizer



def get_article_chunks(text_path, path_to_vocab, max_chunk_len, only_double_enter=True, pack_sentences=True):
    '''Takes the text of a given article and divides it into chunks of predefined length.
    Returns a list of dictionaries containing text chunks and their offsets within the original text'''
    base_unc_tokenizer = BertTokenizer(vocab_lookup_table=path_to_vocab,
                                       lower_case=True)
    paths = os.path.split(text_path)
    artID = int(re.findall('[0-9]+', paths[-1])[0])
    with open(text_path, encoding='utf-8') as f:
        text = f.read()
    if only_double_enter:
        sentences = re.split('\n\n', text)
    else:
        sentences = re.split('\n', text)
    if pack_sentences:
        sentences_and_lengths = [(chunk_text, len(base_unc_tokenizer.split(chunk_text).flat_values)) for chunk_text in
                                 sentences]
        
        chunk_len_count = 0
        chunks_text_list = []
        chunk_groups = []
        for sent_idn in range(len(sentences_and_lengths)):
            if int(chunk_len_count + sentences_and_lengths[sent_idn][1] + 1) < max_chunk_len - 4:
                chunks_text_list.append(sentences_and_lengths[sent_idn][0])
                chunk_len_count += int(sentences_and_lengths[sent_idn][1] + 1)
                if sent_idn == len(sentences_and_lengths) - 1:
                    chunk_groups.append(chunks_text_list)
            else:
                chunk_groups.append(chunks_text_list)
                chunk_len_count = 0
                chunks_text_list = []
                chunks_text_list.append(sentences_and_lengths[sent_idn][0])
                chunk_len_count += (sentences_and_lengths[sent_idn][1] + 1)
        
        chunks = ['\n'.join(mini_chunks) for mini_chunks in chunk_groups]
    else:
        chunks = sentences
    
    chunks_meta = []
    a = 0
    for chunk in chunks:
        chunk_metadata = {'text': chunk,
                          'start_ind': text.index(chunk, a),
                          'end_ind': text.index(chunk, a) + len(chunk),
                          'artID': artID,
                          'subtoken_num': len(base_unc_tokenizer.split(chunk).flat_values)}
        chunks_meta.append(chunk_metadata)
        a += len(chunk)
    return chunks_meta



def triplets2dict(nested_list):
    """Converts label-tuples into dictionaries """
    labels_dict = {}
    artID = int(nested_list[0][0])
    for dict_triple in nested_list:
        if dict_triple[1] not in list(labels_dict.keys()):
            labels_dict[dict_triple[1]] = [(dict_triple[2],
                                            dict_triple[3])]
        else:
            labels_dict[dict_triple[1]].append((dict_triple[2],
                                                dict_triple[3]))
    return artID, labels_dict


def labels2dictOfdicts(labels_path):
    """Takes labels path as an input and produces a dictionary containing labels
    and their metadata that will bal alter appended to the tokens they belong to"""

    with open(labels_path, 'r', encoding='utf-8') as f:
        everything = f.readlines()
    everything = [i.rstrip('\n').split('\t') for i in everything]
    everything = sorted([[int(i[0]), i[1], int(i[2]), int(i[3])] for i in everything], key=lambda x: x[0])
    grouped_everything = []
    for k, g in groupby(everything, key=lambda x: x[0]):
        grouped_everything.append(list(g))
    everyting_dicts = [triplets2dict(i) for i in grouped_everything]
    
    final_dict = {}
    
    for i, j in everyting_dicts:
        final_dict[i] = j
    
    return final_dict


def get_labels(article_path, all_labels_dict):
    """Extracts article number from the article path"""
    artID = int(re.findall('[0-9]+', os.path.basename(article_path))[0])
    try:
        article_labels_dict = all_labels_dict[artID]
    except:
        article_labels_dict = {}
    return article_labels_dict



def reorganize_tuple(tuple_with_lists):
    '''Convert the output when the tokenizer is fed unknow words'''
    if len(tuple_with_lists[0]) == 1:
        output = (tuple_with_lists[0][0], tuple_with_lists[1][0], tuple_with_lists[2][0])
    elif len(tuple_with_lists[0]) >= 1:
        output = []
        for i in range(len(tuple_with_lists[0])):
            single_tuple = (tuple_with_lists[0][i], tuple_with_lists[1][i], tuple_with_lists[2][i])
            output.append(single_tuple)
    return output


def label_token(token_tuple, labels_dict):
    '''Appends label to a token'''
    resulting_tuple = []
    resulting_tuple.append(token_tuple[0])
    for prop_cat in list(labels_dict.keys()):
        for offset in labels_dict[prop_cat]:
            if offset[0] <= token_tuple[1] and offset[1] >= token_tuple[2]:
                resulting_tuple.append(prop_cat)
    if len(resulting_tuple) == 1:
        resulting_tuple.append('None')
    return tuple(resulting_tuple)


def label_for_token(token_tuple, labels_dict):
    """Appends label to a token based on the token offsets within the text-chunk"""
    resulting_tuple = []
    resulting_tuple.append(token_tuple[0])  # indeks słowa ze słownika
    for prop_cat in list(labels_dict.keys()):
        for offset in labels_dict[prop_cat]:
            if offset[0] <= token_tuple[1] and offset[1] >= token_tuple[2]:
                resulting_tuple.append(prop_cat)
    if len(resulting_tuple) == 1:
        resulting_tuple.append('None')
    return resulting_tuple[1:]


def update_offsets(reorganized_tuples, chunk_bo):
    """Updates token offsets taking into account chunk position within the original article"""
    reorganized_tuples_2 = []
    for i in reorganized_tuples:
        if type(i) == tuple:
            reorganized_tuples_2.append((i[0], i[1] + chunk_bo, i[2] + chunk_bo))
        elif type(i) == list:
            new_tuples = [(j[0], j[1] + chunk_bo, j[2] + chunk_bo) for j in i]
            reorganized_tuples_2.append(new_tuples)
    return reorganized_tuples_2


def produce_tokens_and_labels(chunked_text_dict, labels_dict, path_to_vocab, nested=False, normalize_encode=True):
    """Takes texts' and labels' dicts and produces lists containing tokens, their labels and their offsets"""
    
    base_unc_tokenizer = BertTokenizer(vocab_lookup_table=path_to_vocab,
                                       lower_case=True)
    if normalize_encode == True:
        wordpieces, start_positions, end_positions = base_unc_tokenizer.tokenize_with_offsets(
            unicodedata.normalize('NFKD',
                                  chunked_text_dict['text'].replace("’", "'")).encode('ascii',
                                                                                      'replace'))
    else:
        wordpieces, start_positions, end_positions = base_unc_tokenizer.tokenize_with_offsets(chunked_text_dict['text'])
    token_list = [(token, start, end) for
                  token, start, end in
                  zip(wordpieces.to_list()[0], start_positions.to_list()[0], end_positions.to_list()[0])]
    token_list = [reorganize_tuple(i) for i in token_list]
    chunk_offset = int(chunked_text_dict['start_ind'])
    token_list = update_offsets(token_list, chunk_offset)
    tokens = []
    labels = []
    offsets = []
    for token in token_list:
        if type(token) == tuple:
            token_with_label = label_token(token, labels_dict=labels_dict)
            tokens.append(token_with_label[0])
            offsets.append((token[1] - chunk_offset, token[2] - chunk_offset))
            if len(token_with_label) == 2:
                labels.append(token_with_label[1])
            elif len(token_with_label) > 2:
                labels.append(tuple(token_with_label[1:]))
        elif type(token) == list:
            token_details = [label_token(subtoken_tuple, labels_dict=labels_dict) for subtoken_tuple in token]
            sub_tokens = [i[0] for i in token_details]
            sub_labels = [i[1] if len(i) == 2 else tuple(i[1:]) for i in token_details]
            sub_offsets = [(subtoken_tuple[1] - chunk_offset, subtoken_tuple[2] - chunk_offset) for subtoken_tuple in
                           token]
            if nested == True:
                tokens.append(sub_tokens)
                labels.append(sub_labels)
                offsets.append(sub_offsets)
            else:
                tokens += sub_tokens
                labels += sub_labels
                offsets += sub_offsets
    
    return tokens, labels, offsets


def produce_tokens_artIDs_offsets(chunked_text_dict, path_to_vocab, nested=False, normalize_encode=True):
    '''Takes a dictionary with chunk-text and article metadata and
    produces a nested list that will later be used as a placeholder for nested list of predictions.'''

    base_unc_tokenizer = BertTokenizer(vocab_lookup_table=path_to_vocab,
                                       lower_case=True)
    if normalize_encode == True:
        wordpieces, start_positions, end_positions = base_unc_tokenizer.tokenize_with_offsets(
            unicodedata.normalize('NFKD',
                                  chunked_text_dict['text'].replace("’", "'")).encode('ascii',
                                                                                      'replace'))
    else:
        wordpieces, start_positions, end_positions = base_unc_tokenizer.tokenize_with_offsets(chunked_text_dict['text'])
    token_list = [(token, start, end) for
                  token, start, end in
                  zip(wordpieces.to_list()[0], start_positions.to_list()[0], end_positions.to_list()[0])]
    token_list = [reorganize_tuple(i) for i in token_list]
    token_list = update_offsets(token_list, int(chunked_text_dict['start_ind']))
    tokens_with_details = []
    for token in token_list:
        if type(token) == tuple:
            detailed_token = (token[0], token[1], token[2], chunked_text_dict['artID'])
            tokens_with_details.append(detailed_token)
        elif type(token) == list:
            detailed_subtokens = [(subtoken[0], subtoken[1], subtoken[2], chunked_text_dict['artID']) for subtoken in
                                  token]
            if nested == True:
                tokens_with_details.append(detailed_subtokens)
            else:
                tokens_with_details += detailed_subtokens

    return tokens_with_details

