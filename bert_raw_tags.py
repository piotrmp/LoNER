from data_processing.into_bert_2 import prepare_single_label_input_arrays, createZipDataset
from data_processing.into_bert import PTC_dict_with_None
from data_processing.processing import output_artIDs_tokens_offsets
from task_flc_statistics.saving_statistics import output_multi_stats, multi_update
from data_processing.converting_results import SingleLabelPreds2semeval
from data_processing.into_bert import find_file
from Models.hub_based_single_label_bert import BertPTC_sl
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from fastprogress import master_bar, progress_bar
import tensorflow as tf
import numpy as np
import math
import json
import warnings
import os


warnings.simplefilter("ignore")

sequence_length = 200
batch_size = 4
num_epochs = 50

bert_model_path = r'/home/users/piotrmp/TF_models/bert_base'
bert_model_name = os.path.basename(bert_model_path)
output_folder = os.path.join(r'/home/users/piotrmp/out', bert_model_name)
stats_folder = os.path.join(output_folder, 'statistics')
preds_folder = os.path.join(output_folder, 'predictions')
os.makedirs(stats_folder, exist_ok=True)
os.makedirs(preds_folder, exist_ok=True)


train_articles_path = 'PTC/train_articles'
dev_articles_path = 'PTC/dev_articles'
test_articles_path = 'PTC/test_articles'

train_labels_path = 'PTC/train_labels.txt'
dev_labels_path = 'PTC/dev_labels.txt'
test_labels_path = 'PTC/test_labels.txt'


# Vocabulary and lables files
print('Loading vocab file \n')
vocabulary = find_file('vocab.txt', bert_model_path)

# Data processing part
train_pred_meta = output_artIDs_tokens_offsets(texts_path=train_articles_path, vocab_path=vocabulary, nested=False,
                                               only_double_enter=False, sentence_length=sequence_length)
train_input_ids, train_input_masks, train_input_type, train_labels, train_booleans = prepare_single_label_input_arrays(
    texts_path=train_articles_path, labels_path=train_labels_path,
    vocab_path=vocabulary,
    sequence_length=sequence_length,
    nested=False,
    only_double_enter=False,
    categories_dict=PTC_dict_with_None)

zip_train_data = createZipDataset(train_input_ids,
                                  train_input_masks,
                                  train_input_type,
                                  train_labels,
                                  train_booleans)

zip_train_data = zip_train_data.batch(batch_size)

dev_pred_meta = output_artIDs_tokens_offsets(texts_path=dev_articles_path, vocab_path=vocabulary, nested=False,
                                             only_double_enter=False, sentence_length=sequence_length)
dev_input_ids, dev_input_masks, dev_input_type, dev_labels, dev_booleans = prepare_single_label_input_arrays(
    texts_path=dev_articles_path, labels_path=dev_labels_path,
    vocab_path=vocabulary,
    sequence_length=sequence_length,
    nested=False,
    only_double_enter=False,
    categories_dict=PTC_dict_with_None)

zip_dev_data = createZipDataset(dev_input_ids,
                                dev_input_masks,
                                dev_input_type,
                                dev_labels,
                                dev_booleans)

zip_dev_data = zip_dev_data.batch(batch_size)

test_pred_meta = output_artIDs_tokens_offsets(texts_path=test_articles_path, vocab_path=vocabulary, nested=False,
                                              only_double_enter=False, sentence_length=sequence_length)

test_input_ids, test_input_masks, test_input_type, test_labels, test_booleans = prepare_single_label_input_arrays(
    texts_path=test_articles_path, labels_path=test_labels_path,
    vocab_path=vocabulary,
    sequence_length=sequence_length,
    nested=False,
    only_double_enter=False,
    categories_dict=PTC_dict_with_None)

zip_test_data = createZipDataset(test_input_ids,
                                 test_input_masks,
                                 test_input_type,
                                 test_labels,
                                 test_booleans)

zip_test_data = zip_test_data.batch(batch_size)

# Define the model
bert_body = BertPTC_sl(float_type=tf.float32, num_labels=15, max_seq_length=sequence_length,
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
        numpy_mask = np.logical_or(important_labels_argmax != 14,
                                   important_logits_argmax != 14)  # None label is the last one
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
    numpy_mask = np.logical_or(important_labels_argmax != 14,
                               important_logits_argmax != 14)  # None label is the last one
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

dev_results_print = []
test_results_print = []

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
    print('Now producing train preds \n')
    train_semeval_preds = SingleLabelPreds2semeval(arrayed_logits_list=train_predictions,
                                                   label_masks_array=train_booleans,
                                                   token_metas=train_pred_meta,
                                                   output_path=os.path.join(preds_folder,
                                                                            f'train_preds_epoch_{epoch + 1}.txt'),
                                                   merge_spans=True)
    print('Now producing dev preds \n')
    dev_semeval_preds = SingleLabelPreds2semeval(arrayed_logits_list=dev_predictions,
                                                 label_masks_array=dev_booleans,
                                                 token_metas=dev_pred_meta,
                                                 output_path=os.path.join(preds_folder,
                                                                          f'dev_preds_epoch_{epoch + 1}.txt'),
                                                 merge_spans=True)
    print('Now producing test preds \n')
    test_semeval_preds = SingleLabelPreds2semeval(arrayed_logits_list=test_predictions,
                                                  label_masks_array=test_booleans,
                                                  token_metas=test_pred_meta,
                                                  output_path=os.path.join(preds_folder,
                                                                           f'test_preds_epoch_{epoch + 1}.txt'),
                                                  merge_spans=True)
    print('Counting f-scores\n')

    train_stats, dev_stats, test_stats, train_mm, dev_mm, test_mm = output_multi_stats(os.path.join(stats_folder, f'train_preds_epoch_{epoch + 1}.txt'),
                                                                                       os.path.join(stats_folder, f'dev_preds_epoch_{epoch + 1}.txt'),
                                                                                       os.path.join(stats_folder, f'test_preds_epoch_{epoch + 1}.txt'),
                                                                                       train_labels_path,
                                                                                       dev_labels_path,
                                                                                       test_labels_path)
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

    print('Outputing json with statistics')
    with open(os.path.join(stats_folder, f'train_stats_epoch_{epoch + 1}.txt'), 'w') as outfile:
        json.dump(train_results_dict, outfile, indent=3)
    with open(os.path.join(stats_folder, f'dev_stats_epoch_{epoch + 1}.txt'), 'w') as outfile:
        json.dump(dev_results_dict, outfile, indent=3)
    with open(os.path.join(stats_folder, f'test_stats_epoch_{epoch + 1}.txt'), 'w') as outfile:
        json.dump(test_results_dict, outfile, indent=3)

print("Looking for the best epoch according to F-score on dev.")
bestF = -10
bestEpoch = -10
for i in range(len(dev_results_dict['dev_micro_fscore'])):
    if float(dev_results_dict['dev_micro_fscore'][i]) > bestF:
        bestF = float(dev_results_dict['dev_micro_fscore'][i])
        bestEpoch = i

print("Results on dev for best epoch ("+str(bestEpoch+1)+"): "+str(dev_results_print[bestEpoch]))
print("Results on test for best epoch ("+str(bestEpoch+1)+"): "+str(test_results_print[bestEpoch]))

print('Thanks and bye bye !!!')

