import numpy as np
import os
# Empty-lengths categories dict for the PTC dataset
PTC_categories_dict = {
    'Loaded_Language': 0,
    'Name_Calling,Labeling': 0,
    'Repetition': 0,
    'Exaggeration,Minimisation': 0,
    'Doubt': 0,
    'Appeal_to_fear-prejudice': 0,
    'Flag-Waving': 0,
    'Causal_Oversimplification': 0,
    'Slogans': 0,
    'Appeal_to_Authority': 0,
    'Black-and-White_Fallacy': 0,
    'Thought-terminating_Cliches': 0,
    'Whataboutism,Straw_Men,Red_Herring': 0,
    'Bandwagon,Reductio_ad_hitlerum': 0
}
# Empty-lengths categories dict for the PICO dataset
PIO_categories_dict = {
    'p_Age': 0,
    'p_Sex': 0,
    'p_Sample size': 0,
    'p_Condition': 0,
    'i_Surgical': 0,
    'i_Physical': 0,
    'i_Pharmacological': 0,
    'i_Educational': 0,
    'i_Psychological': 0,
    'i_Other': 0,
    'i_Control': 0,
    'o_Physical': 0,
    'o_Pain': 0,
    'o_Mortality': 0,
    'o_Adverse effects': 0,
    'o_Mental': 0,
    'o_Other': 0
                    }

# Add 'None' category o the original dictionary
PTC_dict_with_None = PTC_categories_dict.copy()
PTC_dict_with_None['None'] = 0

PIO_dict_with_None = PIO_categories_dict.copy()
PIO_dict_with_None['None'] = 0

# BIO scheme dict with 'None' as 'O'
BIO_dict = {}
for category in list(PTC_categories_dict.keys()):
    BIO_dict[f'B-{category}'] = 0
for category in list(PTC_categories_dict.keys()):
    BIO_dict[f'I-{category}'] = 0
BIO_dict['None'] = 0

# IOBES scheme dict with 'None' as 'O'
IOBES_dict = {}
for category in list(PTC_categories_dict.keys()):
    IOBES_dict[f'S-{category}'] = 0
for category in list(PTC_categories_dict.keys()):
    IOBES_dict[f'B-{category}'] = 0
for category in list(PTC_categories_dict.keys()):
    IOBES_dict[f'I-{category}'] = 0
for category in list(PTC_categories_dict.keys()):
    IOBES_dict[f'E-{category}'] = 0
IOBES_dict['None'] = 0

def find_file(name, path):
    '''Finds vocab.txt file inside hub-model path'''
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def singleLabel2array(labels_sentence, categories_dict):
    '''Converts string labels into numerical vectors of zeros and ones,
     with the posision of "1" indicating the propaganda category
    - propaganda category number id is based on the order of keys in the categories_dict argument'''
    sentence = []
    
    for i in labels_sentence:
        
        categories = categories_dict.copy()
        if type(i) == str:  # if there is one label for a word
            if i in list(categories.keys()):
                categories[i] = 1
        elif type(i) == tuple:
            category = i[0]
            if category in list(categories.keys()):
                categories[category] = 1
        sentence.append(list(categories.values()))
    return sentence


def singleLabelBIO2array(labels_sentence, reference_dict, output_dict):
    '''Converts string labels into numerical vectors of zeros and ones,
     with the posision of "1" indicating the propaganda category (according to BIO - labeling scheme)
        - propaganda category number id is based on the order of keys in the categories_dict argument'''
    string_single_labels = [i if type(i) == str else i[0] for i in labels_sentence]
    BIO_string_single_labels = []
    BIO_array_labels = []
    
    for i in range(len(string_single_labels)):
        if i == 0:
            if string_single_labels[i] in list(reference_dict.keys()):
                BIO_string_single_labels.append(f'B-{string_single_labels[i]}')
            else:
                BIO_string_single_labels.append('None')
        else:
            if string_single_labels[i] in list(reference_dict.keys()):
                if string_single_labels[i] == string_single_labels[i - 1]:
                    BIO_string_single_labels.append(f'I-{string_single_labels[i]}')
                else:
                    BIO_string_single_labels.append(f'B-{string_single_labels[i]}')
            else:
                BIO_string_single_labels.append('None')
    for label in BIO_string_single_labels:
        support_dict = output_dict.copy()
        support_dict[label] = 1
        to_add = list(support_dict.values())
        if len(to_add) != 29:
            print('Somethings not wright!')
        BIO_array_labels.append(to_add)
    
    return BIO_array_labels


def singleLabelIOBESarray(labels_sentence, reference_dict, output_dict):
    '''Converts string labels into numerical vectors of zeros and ones,
     with the posision of "1" indicating the propaganda category (according to IOBES - labeling scheme)
        - propaganda category number id is based on the order of keys in the categories_dict argument'''

    string_single_labels = [i if type(i) == str else i[0] for i in labels_sentence]
    IOBES_string_single_labels = []
    IOBES_array_labels = []
    
    if len(string_single_labels) == 1:
        if string_single_labels[0] in list(reference_dict.keys()):
            IOBES_string_single_labels.append(f'B-{string_single_labels[0]}')
        else:
            IOBES_string_single_labels.append('None')
    else:
        
        for i in range(len(string_single_labels)):
            if i == 0:
                if string_single_labels[i] not in list(reference_dict.keys()):
                    IOBES_string_single_labels.append('None')
                else:
                    if string_single_labels[i] == string_single_labels[i + 1]:
                        IOBES_string_single_labels.append(f'B-{string_single_labels[i]}')
                    else:
                        IOBES_string_single_labels.append(f'S-{string_single_labels[i]}')
            elif i > 0 and i < len(string_single_labels) - 1:
                if string_single_labels[i] not in list(reference_dict.keys()):
                    IOBES_string_single_labels.append('None')
                else:
                    if string_single_labels[i] != string_single_labels[i - 1] and string_single_labels[i] == \
                            string_single_labels[i + 1]:
                        IOBES_string_single_labels.append(f'B-{string_single_labels[i]}')
                    if string_single_labels[i] != string_single_labels[i - 1] and string_single_labels[i] != \
                            string_single_labels[i + 1]:
                        IOBES_string_single_labels.append(f'S-{string_single_labels[i]}')
                    if string_single_labels[i] == string_single_labels[i - 1] and string_single_labels[i] == \
                            string_single_labels[i + 1]:
                        IOBES_string_single_labels.append(f'I-{string_single_labels[i]}')
                    if string_single_labels[i] == string_single_labels[i - 1] and string_single_labels[i] != \
                            string_single_labels[i + 1]:
                        IOBES_string_single_labels.append(f'E-{string_single_labels[i]}')
            elif i == len(string_single_labels) - 1:
                if string_single_labels[i] not in list(reference_dict.keys()):
                    IOBES_string_single_labels.append('None')
                else:
                    if string_single_labels[i] != string_single_labels[i - 1]:
                        IOBES_string_single_labels.append(f'S-{string_single_labels[i]}')
                    if string_single_labels[i] == string_single_labels[i - 1]:
                        IOBES_string_single_labels.append(f'E-{string_single_labels[i]}')
    
    for label in IOBES_string_single_labels:
        support_dict = output_dict.copy()
        support_dict[label] = 1
        IOBES_array_labels.append(list(support_dict.values()))
    
    return IOBES_array_labels


def tokens2bert(list_of_flat_sentences, sentence_length=210):
    '''
    Converts list of lists into numpy arrays.
       Adds a CLS token at the beginning and the SEP token at the end of the sentence.
       Produces 3 numpy ndarrays that later become an input for BERT:
       - token ids array, masks array and segment array.
       For reference go here: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
       '''
    all_token_ids = []
    all_masks = []
    all_segments = []
    for list_of_token_ids in list_of_flat_sentences:
        list_of_token_ids = list_of_token_ids[
                            :(sentence_length - 2)]
        sent = [101] + list_of_token_ids + [102]  # ids' of CLS and SEP
        pad_len = sentence_length - len(sent)
        sent_ids_max = sent + [0] * pad_len
        pad_masks = [1] * len(sent) + [0] * pad_len
        segment_ids = [0] * sentence_length
        all_token_ids.append(sent_ids_max)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        

    return np.array(all_token_ids), np.array(all_masks), np.array(all_segments)



def singleLabels2bert(single_labels, reference_dict, sentence_length=210):
    '''Converts label list of lists into numpy ndarrays that are fed into Bert model.
    Adds PAD masks where the sentence is shorter than the sentence_lenght argument'''
    
    labels_vector = []
    bools_vector = []
    
    for labels_list in single_labels:
        outside_label = [0] * len(list(reference_dict.keys()))
        outside_label[-1] = 1
        ones_label = [1] * len(list(reference_dict.keys()))
        labels_list = labels_list[:sentence_length - 2]
        complete_labels = [outside_label] + labels_list + [outside_label]
        labels_to_mask = [ones_label] + labels_list + [ones_label]
        to_PAD = sentence_length - len(complete_labels)
        complete_labels_padded = complete_labels + [outside_label] * to_PAD
        complete_labels_to_mask = labels_to_mask + [ones_label] * to_PAD
        labels_boolean = [True if 0 in i else False for i in complete_labels_to_mask]
        labels_vector.append(complete_labels_padded)
        bools_vector.append(labels_boolean)
    
    return np.array(labels_vector), np.array(bools_vector)


def singleLabelsBIO2bert(single_BIO_labels, reference_dict, sentence_length=210):
    '''Converts label list of lists (in BIO scheme) into numpy ndarrays that are fed into Bert model.
    Adds PAD masks where the sentence is shorter than the sentence_lenght argument'''
    
    labels_vector = []
    bools_vector = []
    
    for labels_list in single_BIO_labels:
        outside_label = [0] * len(list(reference_dict.keys()))
        outside_label[-1] = 1
        ones_label = [1] * len(list(reference_dict.keys()))
        labels_list = labels_list[:sentence_length - 2]
        complete_labels = [outside_label] + labels_list + [outside_label]
        labels_to_mask = [ones_label] + labels_list + [ones_label]
        to_PAD = sentence_length - len(complete_labels)
        complete_labels_padded = complete_labels + [outside_label] * to_PAD
        complete_labels_to_mask = labels_to_mask + [ones_label] * to_PAD
        labels_boolean = [True if 0 in i else False for i in complete_labels_to_mask]
        labels_vector.append(complete_labels_padded)
        bools_vector.append(labels_boolean)
    
    return np.array(labels_vector), np.array(bools_vector)


def singleLabelsIOBES2bert(single_IOBES_labels, reference_dict, sentence_length=210):
    '''Converts label list of lists (in IOBES scheme) into numpy ndarrays that are fed into Bert model.
       Adds PAD masks where the sentence is shorter than the sentence_lenght argument'''
    
    labels_vector = []
    bools_vector = []
    
    for labels_list in single_IOBES_labels:
        outside_label = [0] * len(list(reference_dict.keys()))
        outside_label[-1] = 1
        ones_label = [1] * len(list(reference_dict.keys()))
        labels_list = labels_list[:sentence_length - 2]
        complete_labels = [outside_label] + labels_list + [outside_label]
        labels_to_mask = [ones_label] + labels_list + [ones_label]
        to_PAD = sentence_length - len(complete_labels)
        complete_labels_padded = complete_labels + [outside_label] * to_PAD
        complete_labels_to_mask = labels_to_mask + [ones_label] * to_PAD
        labels_boolean = [True if 0 in i else False for i in complete_labels_to_mask]
        labels_vector.append(complete_labels_padded)
        bools_vector.append(labels_boolean)
    
    return np.array(labels_vector), np.array(bools_vector)