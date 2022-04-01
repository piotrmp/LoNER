from task_flc_statistics.recall_precision_f1 import precision_recall_fscore_dataset
from task_flc_statistics.recall_precision_f1 import labels2dict
from task_flc_statistics.recall_precision_f1 import harsh_precision_recall_fscore_dataset
from data_processing.evaluations_2 import KLdivergence, KLdivergence_with_None
from copy import deepcopy
import os
import re


def recalls_precs_fscores(preds_path, gold_path):
    '''Produce a dict with evaluation statistics for a semeval-styled predictions file'''
    
    per_category_and_micro = precision_recall_fscore_dataset(preds_path, gold_path)
    per_category = per_category_and_micro.copy()
    del per_category['Overall result']
    macro_recall = round(sum([float(per_category[k]['recall']) for k in per_category.keys()])/len(per_category.keys()), 4)
    macro_precision = round(sum([float(per_category[k]['precision']) for k in per_category.keys()]) / len(per_category.keys()), 4)
    macro_fscore = round(sum([float(per_category[k]['f-score']) for k in per_category.keys()]) / len(per_category.keys()), 4)
    
    output_dict = {
        'micro_recall': per_category_and_micro['Overall result']['recall'],
        'micro_precision': per_category_and_micro['Overall result']['precision'],
        'micro_fscore': per_category_and_micro['Overall result']['f-score'],
        'macro_recall' : macro_recall,
        'macro_precision' : macro_precision,
        'macro_fscore' : macro_fscore
    }
    
    return output_dict


def matching_recalls_precs_fscores(preds_path, gold_path):
    '''Produce a dict with evaluation statistics for a semeval-styled predictions file,
        counting only perfect-matched propaganda spans'''

    per_category_and_micro = harsh_precision_recall_fscore_dataset(preds_path, gold_path)
    per_category = per_category_and_micro.copy()
    del per_category['Overall result']
    macro_recall = round(
        sum([float(per_category[k]['recall']) for k in per_category.keys()]) / len(per_category.keys()), 4)
    macro_precision = round(
        sum([float(per_category[k]['precision']) for k in per_category.keys()]) / len(per_category.keys()), 4)
    macro_fscore = round(
        sum([float(per_category[k]['f-score']) for k in per_category.keys()]) / len(per_category.keys()), 4)

    output_dict = {
        'micro_recall': per_category_and_micro['Overall result']['recall'],
        'micro_precision': per_category_and_micro['Overall result']['precision'],
        'micro_fscore': per_category_and_micro['Overall result']['f-score'],
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'macro_fscore': macro_fscore
    }

    return output_dict


def recalls_precs_fscores_name(preds_path, gold_path, prefix):
    '''Produce a dict with evaluation statistics for a semeval-styled predictions file'''

    per_category_and_micro = precision_recall_fscore_dataset(preds_path, gold_path)
    per_category = per_category_and_micro.copy()
    del per_category['Overall result']
    macro_recall = round(
        sum([float(per_category[k]['recall']) for k in per_category.keys()]) / len(per_category.keys()), 4)
    macro_precision = round(
        sum([float(per_category[k]['precision']) for k in per_category.keys()]) / len(per_category.keys()), 4)
    macro_fscore = round(
        sum([float(per_category[k]['f-score']) for k in per_category.keys()]) / len(per_category.keys()), 4)

    output_dict = {
        f'{prefix}_micro_recall': per_category_and_micro['Overall result']['recall'],
        f'{prefix}_micro_precision': per_category_and_micro['Overall result']['precision'],
        f'{prefix}_micro_fscore': per_category_and_micro['Overall result']['f-score'],
        f'{prefix}_macro_recall': macro_recall,
        f'{prefix}_macro_precision': macro_precision,
        f'{prefix}_macro_fscore': macro_fscore
    }

    return output_dict

def spans2lengths(span_list):
    '''converts a list of span tuples to a list of lengths'''
    
    return [int(i[1]) - int(i[0]) for i in span_list]

def KLs(preds_path, gold_path, bin_size: int):
    """Count Kullback–Leibler divergence between span-length
        distributions of predictions and labels"""
    preds = labels2dict(preds_path)
    labels = labels2dict(gold_path)
    
    cat_spans_dikt = {}
    for art in preds.keys():
        for category in preds[art].keys():
            if category not in cat_spans_dikt.keys():
                cat_spans_dikt[category] = {'pred_spans' : spans2lengths(deepcopy(preds[art][category]))}
            else:
                cat_spans_dikt[category]['pred_spans'] += spans2lengths(deepcopy(preds[art][category]))

    for art in labels.keys():
        for category in labels[art].keys():
            if category not in cat_spans_dikt.keys():
                cat_spans_dikt[category] = {'label_spans': spans2lengths(deepcopy(labels[art][category])), 'pred_spans' : []}
            elif category in cat_spans_dikt.keys() and 'label_spans' not in cat_spans_dikt[category].keys():
                cat_spans_dikt[category]['label_spans'] = spans2lengths(deepcopy(labels[art][category]))
            elif category in cat_spans_dikt.keys() and 'label_spans' in cat_spans_dikt[category].keys():
                cat_spans_dikt[category]['label_spans'] += spans2lengths(deepcopy(labels[art][category]))
                
    all_pred_lengths = []
    all_label_lengths = []

    for cat in cat_spans_dikt.keys():
        all_label_lengths += deepcopy(cat_spans_dikt[cat]['label_spans'])
        all_pred_lengths += deepcopy(cat_spans_dikt[cat]['pred_spans'])
        
    macro_KL = round(KLdivergence(all_label_lengths, all_pred_lengths, bin_size = bin_size), 4)
    
    
    category_KLs = [KLdivergence_with_None(deepcopy(cat_spans_dikt[cat]['pred_spans']), deepcopy(cat_spans_dikt[cat]['label_spans']),
                                 bin_size=bin_size) for cat in cat_spans_dikt]
    category_KLs = [i for i in category_KLs if i != None]
    micro_KL = round(sum(category_KLs)/len(category_KLs), 4)
    
    return {'micro averaged KL' : micro_KL, 'macro averaged KL' : macro_KL}


def KLs_name(preds_path, gold_path, bin_size: int, prefix: str):
    """Count Kullback–Leibler divergence between span-length
            distributions of predictions and labels."""
    preds = labels2dict(preds_path)
    labels = labels2dict(gold_path)

    cat_spans_dikt = {}
    for art in preds.keys():
        for category in preds[art].keys():
            if category not in cat_spans_dikt.keys():
                cat_spans_dikt[category] = {'pred_spans': spans2lengths(deepcopy(preds[art][category]))}
            else:
                cat_spans_dikt[category]['pred_spans'] += spans2lengths(deepcopy(preds[art][category]))

    for art in labels.keys():
        for category in labels[art].keys():
            if category not in cat_spans_dikt.keys():
                cat_spans_dikt[category] = {'label_spans': spans2lengths(deepcopy(labels[art][category])),
                                            'pred_spans': []}
            elif category in cat_spans_dikt.keys() and 'label_spans' not in cat_spans_dikt[category].keys():
                cat_spans_dikt[category]['label_spans'] = spans2lengths(deepcopy(labels[art][category]))
            elif category in cat_spans_dikt.keys() and 'label_spans' in cat_spans_dikt[category].keys():
                cat_spans_dikt[category]['label_spans'] += spans2lengths(deepcopy(labels[art][category]))

    all_pred_lengths = []
    all_label_lengths = []

    for cat in cat_spans_dikt.keys():
        try:
            all_label_lengths += deepcopy(cat_spans_dikt[cat]['label_spans'])
        except KeyError:
            pass
        try:
            all_pred_lengths += deepcopy(cat_spans_dikt[cat]['pred_spans'])
        except KeyError:
            pass

    macro_KL = round(KLdivergence(all_label_lengths, all_pred_lengths, bin_size=bin_size), 4)


    category_KLs = [KLdivergence_with_None(deepcopy(cat_spans_dikt[cat]['pred_spans']),
                                           deepcopy(cat_spans_dikt[cat]['label_spans']),
                                           bin_size=bin_size) for cat in cat_spans_dikt if 'label_spans' in cat_spans_dikt[cat].keys()]

    category_KLs = [i for i in category_KLs if i != None]
    micro_KL = round(sum(category_KLs) / len(category_KLs), 4)

    return {f'{prefix}_micro averaged KL': micro_KL, f'{prefix}_macro averaged KL': macro_KL}


def output_stats_dict(pred_path, gold_path, bin_size):
    prefix = ''
    if 'dev' in os.path.basename(gold_path):
        prefix += 'dev'
    elif 'train' in os.path.basename(gold_path):
        prefix += 'train'
    elif 'test' in os.path.basename(gold_path):
        prefix += 'test'

    epoch_stats = recalls_precs_fscores_name(pred_path, gold_path, prefix=prefix)
    epoch_kls = KLs_name(pred_path, gold_path, bin_size=bin_size, prefix=prefix)
    epoch_stats.update(epoch_kls)
    return epoch_stats


def save_stats(submissions_folder_path, gold_path, bin_size, output_path):
    """Saves statistics for all epochs to file"""

    prefix = ''
    if 'dev' in gold_path:
        prefix += 'dev'
    elif 'train'in gold_path:
        prefix += 'train'
    elif 'test' in gold_path:
        prefix += 'test'

    output_folder = os.path.join(output_path, f'{prefix}_predictions_reports')
    os.makedirs(output_folder, exist_ok=True)
    pred_file_names = sorted(os.listdir(submissions_folder_path), key=lambda x: int(re.findall(r'\d+', x)[0]))

    with open(os.path.join(output_folder, f'{prefix}_statistics.txt'), 'w', encoding='utf8') as stats_file:
        
        for file_name in pred_file_names:
            print(file_name)
            pred_path = os.path.join(submissions_folder_path, file_name)
            epoch_stats = recalls_precs_fscores(pred_path, gold_path)
            epoch_kls = KLs(pred_path, gold_path, bin_size=bin_size)
            
            stats_file.write(f'For {file_name}: \n')
            stats_file.write(f'micro recall = {epoch_stats["micro_recall"]} \n')
            stats_file.write(f'micro precision = {epoch_stats["micro_precision"]} \n')
            stats_file.write(f'micro fscore = {epoch_stats["micro_fscore"]} \n')
            stats_file.write(f'macro recall = {epoch_stats["macro_recall"]} \n')
            stats_file.write(f'macro precision = {epoch_stats["macro_precision"]} \n')
            stats_file.write(f'macro fscore = {epoch_stats["macro_fscore"]} \n')
            stats_file.write(f'micro KL = {epoch_kls["micro averaged KL"]} \n')
            stats_file.write(f'macro KL = {epoch_kls["macro averaged KL"]} \n\n')
    return None
