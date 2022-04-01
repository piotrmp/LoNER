"""Quick funcs for combining & saving output statistics during training"""
from task_flc_statistics.recall_precision_f1 import precision_recall_fscore_dataset
from task_flc_statistics.slaner_evaluation import output_stats_dict

def output_multi_stats(train_preds_path, dev_preds_path, test_preds_path,
                     train_labels_path, dev_labels_path, test_labels_path, categories):
    train_stats_dict = precision_recall_fscore_dataset(train_preds_path, train_labels_path, categories)
    dev_stats_dict = precision_recall_fscore_dataset(dev_preds_path, dev_labels_path, categories)
    test_stats_dict = precision_recall_fscore_dataset(test_preds_path, test_labels_path, categories)
    train_micro_macro = output_stats_dict(train_preds_path, train_labels_path, bin_size=5)
    dev_micro_macro = output_stats_dict(dev_preds_path, dev_labels_path, bin_size=5)
    test_micro_macro = output_stats_dict(test_preds_path, test_labels_path, bin_size=5)

    return train_stats_dict, dev_stats_dict, test_stats_dict, train_micro_macro, dev_micro_macro, test_micro_macro

def update_stats(final_dict, epoch_dict):
    for statistic in list(epoch_dict.keys()):
        if statistic not in list(final_dict.keys()):
            final_dict[statistic] = [epoch_dict[statistic]]
        else:
            final_dict[statistic].append(epoch_dict[statistic])
    return final_dict

def multi_update(final_dict, epoch_dict_1, epoch_dict_2, epoch_dict_3):
    final_dict = update_stats(final_dict, epoch_dict_1)
    final_dict = update_stats(final_dict, epoch_dict_2)
    final_dict = update_stats(final_dict, epoch_dict_3)
    return final_dict






