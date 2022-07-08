from data_processing.into_bert_2 import prepare_single_label_input_arrays, createZipDataset
from data_processing.into_bert import find_file, PIO_categories_dict, PIO_dict_with_None
from data_processing.converting_results import SingleLabelPreds2semeval
from data_processing.processing import output_artIDs_tokens_offsets
from task_flc_statistics.saving_statistics import output_multi_stats, multi_update
from task_flc_statistics.recall_precision_f1 import pio_categories
from Models.bert import BertPTC_sl
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from fastprogress import master_bar, progress_bar
import tensorflow as tf
import numpy as np
import math
import json
import warnings
import os
import logging
import sys

warnings.simplefilter("ignore")
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', stream=sys.stdout, level=logging.INFO)


bert_model_path = r'/home/users/piotrmp/TF_models/bert_base' # Replace it with your local model path
bert_model_name = os.path.basename(bert_model_path)
output_folder = os.path.join(r'/home/users/piotrmp/out', bert_model_name)# Replace it with your local outputs path
stats_folder = os.path.join(output_folder, 'statistics')
preds_folder = os.path.join(output_folder, 'predictions')
os.makedirs(stats_folder, exist_ok=True)
os.makedirs(preds_folder, exist_ok=True)


train_abstracts_path = 'PICO_9999/train_abstracts' # Replace it with local path to train_abstracts dir from unzipped PICO_dataset.zip
dev_abstracts_path = 'PICO_9999/dev_abstracts'# Replace it with local path to dev_abstracts dir from unzipped PICO_dataset.zip
test_abstracts_path = 'PICO_9999/test_abstracts'# Replace it with local path to test_abstracts dir from unzipped PICO_dataset.zip
train_labels_path = 'PICO_9999/train_labels.txt' # Replace it with local path to train_labels file from unzipped PICO_dataset.zip
dev_labels_path = 'PICO_9999/dev_labels.txt' # Replace it with local path to dev_labels file from unzipped PICO_dataset.zip
test_labels_path = 'PICO_9999/test_labels.txt' # Replace it with local path to test_labels file from unzipped PICO_dataset.zip

# You can manipulate those values
sequence_length = 200
batch_size = 4
num_epochs = 50

# Vocabulary and labels files
logging.info('Loading vocab file')
vocabulary = find_file('vocab.txt', bert_model_path)

# Data processing part
logging.info('Loading & processing training data')

train_pred_meta = output_artIDs_tokens_offsets(texts_path=train_abstracts_path, vocab_path=vocabulary, nested=False,
                                               only_double_enter=False, sentence_length=sequence_length)

train_input_ids, train_input_masks, train_input_type, train_labels, train_booleans = prepare_single_label_input_arrays(
    texts_path=train_abstracts_path, labels_path=train_labels_path,
    vocab_path=vocabulary,
    sequence_length=sequence_length,
    nested=False,
    only_double_enter=False,
    categories_dict=PIO_dict_with_None)

zip_train_data = createZipDataset(train_input_ids,
                                  train_input_masks,
                                  train_input_type,
                                  train_labels,
                                  train_booleans)

zip_train_data = zip_train_data.batch(batch_size)


logging.info('Loading & processing validation data')
dev_pred_meta = output_artIDs_tokens_offsets(texts_path=dev_abstracts_path, vocab_path=vocabulary, nested=False,
                                             only_double_enter=False, sentence_length=sequence_length)

dev_input_ids, dev_input_masks, dev_input_type, dev_labels, dev_booleans = prepare_single_label_input_arrays(
    texts_path=dev_abstracts_path, labels_path=dev_labels_path,
    vocab_path=vocabulary,
    sequence_length=sequence_length,
    nested=False,
    only_double_enter=False,
    categories_dict=PIO_dict_with_None)

zip_dev_data = createZipDataset(dev_input_ids,
                                dev_input_masks,
                                dev_input_type,
                                dev_labels,
                                dev_booleans)

zip_dev_data = zip_dev_data.batch(batch_size)


logging.info('Loading & processing test data')
test_pred_meta = output_artIDs_tokens_offsets(texts_path=test_abstracts_path, vocab_path=vocabulary, nested=False,
                                              only_double_enter=False, sentence_length=sequence_length)

test_input_ids, test_input_masks, test_input_type, test_labels, test_booleans = prepare_single_label_input_arrays(
    texts_path=test_abstracts_path, labels_path=test_labels_path,
    vocab_path=vocabulary,
    sequence_length=sequence_length,
    nested=False,
    only_double_enter=False,
    categories_dict=PIO_dict_with_None)


zip_test_data = createZipDataset(test_input_ids,
                                 test_input_masks,
                                 test_input_type,
                                 test_labels,
                                 test_booleans)

zip_test_data = zip_test_data.batch(batch_size)





# Define the model
logging.info('Loading Bert model')
bert_body = BertPTC_sl(float_type=tf.float32, num_labels=len(PIO_dict_with_None), max_seq_length=sequence_length,
                       hub_path=bert_model_path)
SL_Bert = bert_body.getSingleLabelModel()
SL_Bert.summary()

# Define model statistics and metrics
loss_metric = tf.keras.metrics.Mean()
optimizer = Adam(learning_rate=3e-5)
CC_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)



def train_step(input_ids, input_mask, type_ids, label_vec, label_mask):
    with tf.GradientTape() as tape:
        logits = SL_Bert([input_ids, input_mask, type_ids], training=True)
        # label mask masks CLS, SEP and PADs
        important_labels = tf.boolean_mask(label_vec, label_mask)
        important_logits = tf.boolean_mask(logits, label_mask)
        loss = CC_loss(important_labels, important_logits)
        important_logits_argmax = tf.argmax(important_logits, axis=-1)
        important_labels_argmax = tf.argmax(important_labels, axis=-1)
        numpy_mask = np.logical_or(important_labels_argmax != len(PIO_categories_dict),
                                   important_logits_argmax != len(PIO_categories_dict))
        masked_labels_np = tf.boolean_mask(important_labels_argmax, numpy_mask)
        masked_logits_np_argmax = tf.boolean_mask(important_logits_argmax, numpy_mask)
        if np.logical_and(np.any(masked_labels_np), np.any(masked_logits_np_argmax)):
            batch_accuracy = accuracy_score(masked_labels_np, masked_logits_np_argmax)
        else:
            batch_accuracy = float(0)
    grads = tape.gradient(loss, SL_Bert.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, SL_Bert.trainable_variables)))
    return batch_accuracy, loss, logits


def validation_step(input_ids, input_mask, type_ids, label_vec, label_mask):
    logits = SL_Bert([input_ids, input_mask, type_ids], training=False)
    # label mask masks CLS, SEP and PADs
    important_labels = tf.boolean_mask(label_vec, label_mask)
    important_logits = tf.boolean_mask(logits, label_mask)
    loss = CC_loss(important_labels, important_logits)
    important_logits_argmax = tf.argmax(important_logits, axis=-1)
    important_labels_argmax = tf.argmax(important_labels, axis=-1)
    numpy_mask = np.logical_or(important_labels_argmax != len(PIO_categories_dict),
                               important_logits_argmax != len(PIO_categories_dict))
    masked_labels_np = tf.boolean_mask(important_labels_argmax, numpy_mask)
    masked_logits_np_argmax = tf.boolean_mask(important_logits_argmax, numpy_mask)
    if np.logical_and(np.any(masked_labels_np), np.any(masked_logits_np_argmax)):
        batch_accuracy = accuracy_score(masked_labels_np, masked_logits_np_argmax)
    else:
        batch_accuracy = float(0)
    return batch_accuracy, loss, logits



# MODEL RUN
train_pb_len = math.ceil(len(zip_train_data))
dev_pb_len = math.ceil(len(zip_dev_data))
test_pb_len = math.ceil(len(zip_test_data))
epoch_bar = master_bar(range(num_epochs))

train_results_dict = {}
dev_results_dict = {}
test_results_dict = {}
dev_results_print= []
test_results_print = []


logging.info('Starting train, validate & test loop')
for epoch in epoch_bar:
    # Collect statistics per epoch
    mean_train_loss = tf.keras.metrics.Mean()
    mean_train_accuracy = tf.keras.metrics.Mean()
    mean_dev_loss = tf.keras.metrics.Mean()
    mean_dev_accuracy = tf.keras.metrics.Mean()
    mean_test_loss = tf.keras.metrics.Mean()
    mean_test_accuracy = tf.keras.metrics.Mean()
    train_losses = []
    train_accuracies = []
    train_predictions = []
    dev_losses = []
    dev_accuracies = []
    dev_predictions = []
    test_losses = []
    test_accuracies = []
    test_predictions = []

    # TRAINING
    for (input_ids, input_mask, segment_ids,
         label_vec, label_masks) in progress_bar(zip_train_data, total=train_pb_len,
                                                 parent=epoch_bar):
        batch_train_acu, batch_train_loss, batch_train_sequence = train_step(input_ids, input_mask, segment_ids,
                                                                             label_vec, label_masks)
        mean_train_loss.update_state(batch_train_loss)
        mean_train_accuracy.update_state(batch_train_acu)
        epoch_bar.child.comment = f'loss: {round(mean_train_loss.result().numpy().astype(float), 3)}, ' \
                                  f'train accuracy: {round(mean_train_accuracy.result().numpy().astype(float), 3)}'

        train_losses.append(mean_train_loss.result().numpy())
        train_accuracies.append(mean_train_accuracy.result().numpy())
        train_predictions.append(batch_train_sequence)

    # VALIDATION on dev
    for (input_ids, input_mask, segment_ids,
         label_vec, label_masks) in progress_bar(zip_dev_data, total=dev_pb_len,
                                                 parent=epoch_bar):
        batch_dev_acu, batch_dev_loss, batch_dev_sequence = validation_step(input_ids, input_mask, segment_ids,
                                                                            label_vec, label_masks)
        mean_dev_loss.update_state(batch_dev_loss)
        mean_dev_accuracy.update_state(batch_dev_acu)
        epoch_bar.child.comment = f'loss: {round(mean_dev_loss.result().numpy().astype(float), 3)},' \
                                  f'dev accuracy: {round(mean_dev_accuracy.result().numpy().astype(float), 3)}'

        dev_losses.append(mean_dev_loss.result().numpy())
        dev_accuracies.append(mean_dev_accuracy.result().numpy())
        dev_predictions.append(batch_dev_sequence)

    # VALIDATION on test
    for (input_ids, input_mask, segment_ids,
         label_vec, label_masks) in progress_bar(zip_test_data, total=test_pb_len,
                                                 parent=epoch_bar):
        batch_test_acu, batch_test_loss, batch_test_sequence = validation_step(input_ids, input_mask, segment_ids,
                                                                               label_vec, label_masks)
        mean_test_loss.update_state(batch_test_loss)
        mean_test_accuracy.update_state(batch_test_acu)
        epoch_bar.child.comment = f'loss: {round(mean_test_loss.result().numpy().astype(float), 3)},' \
                                  f'test accuracy: {round(mean_test_accuracy.result().numpy().astype(float), 3)}'

        test_losses.append(mean_test_loss.result().numpy())
        test_accuracies.append(mean_test_accuracy.result().numpy())
        test_predictions.append(batch_test_sequence)

    # Per epoch predictions for semeval
    train_semeval_preds = SingleLabelPreds2semeval(arrayed_logits_list=train_predictions,
                                                   label_dict=PIO_dict_with_None,
                                                   label_masks_array=train_booleans,
                                                   token_metas=train_pred_meta,
                                                   output_path=os.path.join(preds_folder,
                                                                            f'train_preds_epoch_{epoch + 1}.txt'),
                                                   merge_spans=True)

    dev_semeval_preds = SingleLabelPreds2semeval(arrayed_logits_list=dev_predictions,
                                                 label_dict=PIO_dict_with_None,
                                                 label_masks_array=dev_booleans,
                                                 token_metas=dev_pred_meta,
                                                 output_path=os.path.join(preds_folder,
                                                                          f'dev_preds_epoch_{epoch + 1}.txt'),
                                                 merge_spans=True)

    test_semeval_preds = SingleLabelPreds2semeval(arrayed_logits_list=test_predictions,
                                                  label_dict=PIO_dict_with_None,
                                                  label_masks_array=test_booleans,
                                                  token_metas=test_pred_meta,
                                                  output_path=os.path.join(preds_folder,
                                                                           f'test_preds_epoch_{epoch + 1}.txt'),
                                                  merge_spans=True)

    train_stats, dev_stats, test_stats, train_mm, dev_mm, test_mm = output_multi_stats(
        os.path.join(stats_folder, f'train_preds_epoch_{epoch + 1}.txt'),
        os.path.join(stats_folder, f'dev_preds_epoch_{epoch + 1}.txt'),
        os.path.join(stats_folder, f'test_preds_epoch_{epoch + 1}.txt'),
        train_labels_path,
        dev_labels_path,
        test_labels_path,
        pio_categories)
    # Add Per epoch statistics to final statistics
    train_stats_2 = {'train_losses': str(round(train_losses[-1], 4)),
                     'train_accs': str(round(train_accuracies[-1], 4))}
    dev_stats_2 = {'dev_losses': str(round(dev_losses[-1], 4)),
                   'dev_accs': str(round(dev_accuracies[-1], 4))}
    test_stats_2 = {'test_losses': str(round(test_losses[-1], 4)),
                    'test_accs': str(round(test_accuracies[-1], 4))}

    dev_results_print.append(str(dev_mm))
    test_results_print.append(str(test_mm))

    train_results_dict = multi_update(train_results_dict, train_stats, train_stats_2, train_mm)
    dev_results_dict = multi_update(dev_results_dict, dev_stats, dev_stats_2, dev_mm)
    test_results_dict = multi_update(test_results_dict, test_stats, test_stats_2, test_mm)

    with open(os.path.join(stats_folder, f'train_stats_epoch_{epoch + 1}.txt'), 'w') as outfile:
        json.dump(train_results_dict, outfile, indent=3)
    with open(os.path.join(stats_folder, f'dev_stats_epoch_{epoch + 1}.txt'), 'w') as outfile:
        json.dump(dev_results_dict, outfile, indent=3)
    with open(os.path.join(stats_folder, f'test_stats_epoch_{epoch + 1}.txt'), 'w') as outfile:
        json.dump(test_results_dict, outfile, indent=3)


logging.info("Looking for the best epoch according to F-score on dev.")
bestF = -10
bestEpoch = -10
for i in range(len(dev_results_dict['dev_micro_fscore'])):
    if float(dev_results_dict['dev_micro_fscore'][i]) > bestF:
        bestF = float(dev_results_dict['dev_micro_fscore'][i])
        bestEpoch = i

logging.info("Results on dev for best epoch (" + str(bestEpoch + 1) + "): " + str(dev_results_print[bestEpoch]))
logging.info("Results on test for best epoch (" + str(bestEpoch + 1) + "): " + str(test_results_print[bestEpoch]))




