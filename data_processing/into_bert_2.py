from data_processing.processing import output_tokens_and_tags
from data_processing.into_bert import tokens2bert
from data_processing.into_bert import singleLabel2array
from data_processing.into_bert import singleLabelBIO2array
from data_processing.into_bert import singleLabels2bert
from data_processing.into_bert import singleLabelsBIO2bert
from data_processing.into_bert import singleLabelsIOBES2bert
from data_processing.into_bert import singleLabelIOBESarray
import tensorflow as tf


def prepare_single_label_input_arrays(texts_path, labels_path, vocab_path, categories_dict, sequence_length=200, nested=False,
                                      only_double_enter=False, parsing=False):

    """Prepare input arrays for BERT model:
        (1) Matrix of token id vectors according to BERT vocabulary file,
        (2) Matrix of 0-1s vectors indicating if there is a valid token or a [PAD],
        (3) Matrix of 0-1s vectors  indicating that it's not question-answering scenario
        (4) Matrix of labels encoded into 0-1 vectors
        (5) Matrix of vectors indicating if labels attends to a valid token
        """
    

    tokens, labels, parses = output_tokens_and_tags(texts_path=texts_path, labels_path=labels_path,
                                                    vocab_path=vocab_path, sentence_length=sequence_length,
                                                    nested=nested, only_double_enter=only_double_enter,
                                                    parsing=parsing)

    vectorized_labels = [singleLabel2array(sentence_labels, categories_dict) for sentence_labels in labels]
    labels_vector_array, label_booleans_array = singleLabels2bert(single_labels=vectorized_labels,
                                                                  reference_dict=categories_dict,
                                                                  sentence_length=sequence_length)
    word_ids_array, word_masks_array, input_types_array, parse_array = tokens2bert(tokens, parses,
                                                                                   sentence_length=sequence_length)
    if parsing:
        return word_ids_array, word_masks_array, input_types_array, labels_vector_array, label_booleans_array, parse_array
    else:
        return word_ids_array, word_masks_array, input_types_array, labels_vector_array, label_booleans_array


def prepare_single_label_BIO_input_arrays(texts_path, labels_path, vocab_path, input_dict, output_dict,
                                          sequence_length=200, nested=False,
                                          only_double_enter=False):

    """Prepare input arrays for BERT model:
            (1) Matrix of token id vectors according to BERT vocabulary file,
            (2) Matrix of 0-1s vectors indicating if there is a valid token or a [PAD],
            (3) Matrix of 0-1s vectors  indicating that it's not question-answering scenario
            (4) Matrix of labels encoded into 0-1 vectors
            (5) Matrix of vectors indicating if labels attends to a valid token
            """

    tokens, labels = output_tokens_and_tags(texts_path=texts_path, labels_path=labels_path,
                                            vocab_path=vocab_path, sentence_length=sequence_length,
                                            nested=nested, only_double_enter=only_double_enter)

    vectorized_labels = [singleLabelBIO2array(sentence_labels, input_dict, output_dict) for sentence_labels in labels]
    labels_vector_array, label_booleans_array = singleLabelsBIO2bert(single_BIO_labels=vectorized_labels,
                                                                     sentence_length=sequence_length,
                                                                     reference_dict=output_dict)
    word_ids_array, word_masks_array, input_types_array = tokens2bert(tokens, sentence_length=sequence_length)
    return word_ids_array, word_masks_array, input_types_array, labels_vector_array, label_booleans_array


def prepare_single_label_IOBES_input_arrays(texts_path, labels_path, vocab_path, input_dict, output_dict,
                                            sequence_length=200, nested=False,
                                            only_double_enter=False):
    """Prepare input arrays for BERT model:
            (1) Matrix of token id vectors according to BERT vocabulary file,
            (2) Matrix of 0-1s vectors indicating if there is a valid token or a [PAD],
            (3) Matrix of 0-1s vectors  indicating that it's not question-answering scenario
            (4) Matrix of labels encoded into 0-1 vectors
            (5) Matrix of vectors indicating if labels attends to a valid token
            """

    tokens, labels = output_tokens_and_tags(texts_path=texts_path, labels_path=labels_path,
                                            vocab_path=vocab_path, sentence_length=sequence_length,
                                            nested=nested, only_double_enter=only_double_enter)

    vectorized_labels = [singleLabelIOBESarray(sentence_labels, input_dict, output_dict) for sentence_labels in labels]
    labels_vector_array, label_booleans_array = singleLabelsIOBES2bert(single_IOBES_labels=vectorized_labels,
                                                                       sentence_length=sequence_length,
                                                                       reference_dict=output_dict)
    word_ids_array, word_masks_array, input_types_array = tokens2bert(tokens, sentence_length=sequence_length)

    return word_ids_array, word_masks_array, input_types_array, labels_vector_array, label_booleans_array


def createZipDataset(word_ids_array, word_masks_array, input_types_array, labels_vector_array, label_booleans_array,
                     parses_array=None):
    print('Przerabiam ndarraye w tf.Datasety i sklejam je zipem')
    dataset_input_ids = tf.data.Dataset.from_tensor_slices(word_ids_array)
    dataset_input_masks = tf.data.Dataset.from_tensor_slices(word_masks_array)
    dataset_segment_ids = tf.data.Dataset.from_tensor_slices(input_types_array)
    dataset_label_vecs = tf.data.Dataset.from_tensor_slices(labels_vector_array)
    dataset_label_masks = tf.data.Dataset.from_tensor_slices(label_booleans_array)
    if parses_array is None:
        zip_data = tf.data.Dataset.zip(
            (dataset_input_ids, dataset_input_masks, dataset_segment_ids, dataset_label_vecs, dataset_label_masks))
    else:
        dataset_parses = tf.data.Dataset.from_tensor_slices(parses_array)
        zip_data = tf.data.Dataset.zip(
            (dataset_input_ids, dataset_input_masks, dataset_segment_ids, dataset_label_vecs, dataset_label_masks,
             dataset_parses))
    print('OK zrobione! \n')
    return zip_data

