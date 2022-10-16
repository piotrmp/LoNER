'''A different take on FLC-task statistics'''
import itertools

propaganda_categories = ['Bandwagon,Reductio_ad_hitlerum',
                         'Thought-terminating_Cliches',
                         'Black-and-White_Fallacy',
                         'Whataboutism,Straw_Men,Red_Herring',
                         'Slogans',
                         'Appeal_to_Authority',
                         'Causal_Oversimplification',
                         'Flag-Waving',
                         'Appeal_to_fear-prejudice',
                         'Exaggeration,Minimisation',
                         'Doubt',
                         'Repetition',
                         'Name_Calling,Labeling',
                         'Loaded_Language']

pio_categories = ['p_Age',
                  'p_Sex',
                  'p_Sample size',
                  'p_Condition',
                  'i_Surgical',
                  'i_Physical',
                  'i_Pharmacological',
                  'i_Educational',
                  'i_Psychological',
                  'i_Other',
                  'i_Control',
                  'o_Physical',
                  'o_Pain',
                  'o_Mortality',
                  'o_Adverse effects',
                  'o_Mental',
                  'o_Other']


def flatten(t):
    '''Make a flat list out of list of lists'''
    return [item for sublist in t for item in sublist]


def labels2list(labels_path):
    """Load in text file with labels"""
    with open(labels_path, 'r', encoding='utf8') as labels:
        label_lines = labels.readlines()
        label_lines = [i.rstrip('\n').split('\t') for i in label_lines]
        label_lines = [[int(i[0]), i[1], (int(i[2]), int(i[3]))] for i in label_lines]
    return label_lines


def labels2dict(labels_path):
    '''Outputs a dict of dicts. The outer dict has article ids as keys.
    Values of the outer dict are dicts with propaganda categories as keys
    and lists of spans [(sart,end) tuples] as values
    Use this function to load both gold labels and predictions'''
    
    with open(labels_path, 'r', encoding='utf8') as labels:
        label_lines = labels.readlines()
        label_lines = [i.rstrip('\n').split('\t') for i in label_lines]
        label_lines = [[int(i[0]), i[1], (int(i[2]), int(i[3]))] for i in label_lines]
    
    by_article = [list(g) for k, g in itertools.groupby(label_lines, lambda x: x[0])]
    
    big_dikt = {}
    
    for article_labels in by_article:
        article_dikt = {}
        for span in article_labels:
            if span[1] not in article_dikt:
                article_dikt[span[1]] = [span[2]]
            else:
                article_dikt[span[1]].append(span[2])
        big_dikt[int(article_labels[0][0])] = article_dikt
    
    return big_dikt

def matching_spans2dikt(labels_path: str, preds_path: str) -> dict:
    """Matches lables and predictions and outputs a dictionary used for counting evaluation statistics"""
    labels = labels2list(labels_path)
    preds = labels2list(preds_path)
    label_lines = [i for i in labels if i in preds]
    by_article = [list(g) for k, g in itertools.groupby(label_lines, lambda x: x[0])]
    big_dikt = {}
    for article_labels in by_article:
        article_dikt = {}
        for span in article_labels:
            if span[1] not in article_dikt:
                article_dikt[span[1]] = [span[2]]
            else:
                article_dikt[span[1]].append(span[2])
        big_dikt[int(article_labels[0][0])] = article_dikt

    return big_dikt




def articleCategoryNums(label_span_list, pred_span_list):
    '''
    For a single propaganda category in a single article:
    Outputs a tuple with:
       1.list of C(s,t,|s|) values in string format (for precision)
       2.list of C(s,t,|t|)values in string format (for recall)
       3. number of all predicted spans (S)
       4. number of all spans in labels
    '''
    
    Csts_s  = [] # list of all C-funcs for precision
    Cstt_s = [] # list of all C-funcs for recall
    S = len(pred_span_list) # number of submitted spans
    T = len(label_span_list) #number of spans in labels
    # srl stands for span range list
    # list of all indices within spans in labels
    all_label_srls = flatten([list(range(sp[0], sp[1])) for sp in label_span_list])
    # list of all indices within spans in predictions
    all_pred_srls = flatten([list(range(sp[0], sp[1])) for sp in pred_span_list])
    # for precision - counting C(s,t,|s|) values for all spans
    for span in pred_span_list:
        srl = list(range(span[0], span[1])) # list of all indices from the span
        srl_ints = set(srl).intersection(set(all_label_srls))
        Csts_s.append(f'{len(srl_ints)}/{len(srl)}')
    # for recall - counting C(s,t,|t|) values for all spans
    for span in label_span_list:
        srl = list(range(span[0], span[1]))
        srl_ints = set(srl).intersection(set(all_pred_srls))
        Cstt_s.append(f'{len(srl_ints)}/{len(srl)}')
    
    return Csts_s, Cstt_s, S, T


def HarshArticleCategoryNums(label_span_list, pred_span_list, matching_pred_span_list):
    '''
    For a single propaganda category in a single article:
    Outputs a tuple with:
       1.list of C(s,t,|s|) values in string format (for precision)
       2.list of C(s,t,|t|)values in string format (for recall)
       3. number of all predicted spans (S)
       4. number of all spans in labels
    '''

    Csts_s = []  # list of all C-funcs for precision
    Cstt_s = []  # list of all C-funcs for recall
    S = len(pred_span_list)  # number of submitted spans
    T = len(label_span_list)  # number of spans in labels
    # srl stands for span range list
    # list of all indices within spans in labels
    all_label_srls = flatten([list(range(sp[0], sp[1])) for sp in label_span_list])
    # list of all indices within spans in predictions !(only those that match 100%)!
    all_pred_srls = flatten([list(range(sp[0], sp[1])) for sp in pred_span_list])
    matching_pred_srls = flatten([list(range(sp[0], sp[1])) for sp in matching_pred_span_list])
    # for precision - counting C(s,t,|s|) values for all spans
    for span in matching_pred_span_list:
        srl = list(range(span[0], span[1]))  # list of all indices from the span
        srl_ints = set(srl).intersection(set(all_label_srls))
        Csts_s.append(f'{len(srl_ints)}/{len(srl)}')
    # for recall - counting C(s,t,|t|) values for all spans
    for span in label_span_list:
        srl = list(range(span[0], span[1]))
        srl_ints = set(srl).intersection(set(matching_pred_srls))
        Cstt_s.append(f'{len(srl_ints)}/{len(srl)}')

    return Csts_s, Cstt_s, S, T


def perArticleNums(article_label_dict:dict, article_pred_dict:dict, categories:list):
    '''
    Takes 2 dicts with propaganda categories as keys and span lists as values
    Outputs a dict with porpaganda categories as keys and lists with
        1. list of precision C-func values for given category
        2. list of recall c-func values for given category
        3. num of predicted spans
        4. num of spans in labels
    If for a given article some categories were present neither in preds nor in labels,
    the value for this key is None
    '''
    # union of predicted propaganda categories and those labeled
    all_categories = list(set(list(article_pred_dict.keys())).union(set(list(article_label_dict.keys()))))
    nums_dict = {}
    general_nums = [[], [], 0, 0] # Csts_s, Cstt_s, S, T
    for cat in categories:
        if cat in all_categories:
            try:
                label_spans = article_label_dict[cat]
            except KeyError:
                label_spans = []
            try:
                pred_spans = article_pred_dict[cat]
            except KeyError:
                pred_spans = []
            category_nums = articleCategoryNums(label_spans, pred_spans)
            nums_dict[cat] = [category_nums]
            # For the 'general' category we sum all Csts_s, Cstt_s, S, T
            general_nums[0] += category_nums[0]
            general_nums[1] += category_nums[1]
            general_nums[2] += category_nums[2]
            general_nums[3] += category_nums[3]
        else:
            nums_dict[cat] = None
    nums_dict['Overall result'] = [general_nums]
    return nums_dict


def HarshPerArticleNums(article_label_dict:dict, article_pred_dict:dict, matching_article_pred_dict: dict, categories:list):
    '''
    Takes 2 dicts with propaganda categories as keys and span lists as values
    Outputs a dict with porpaganda categories as keys and lists with
        1. list of precision C-func values for given category
        2. list of recall c-func values for given category
        3. num of predicted spans
        4. num of spans in labels
    If for a given article some categories were present neither in preds nor in labels,
    the value for this key is None
    '''
    # union of predicted propaganda categories and those labeled
    all_categories = list(set(list(article_pred_dict.keys())).union(set(list(article_label_dict.keys()))))
    nums_dict = {}
    general_nums = [[], [], 0, 0] # Csts_s, Cstt_s, S, T
    for cat in categories:
        if cat in all_categories:
            try:
                label_spans = article_label_dict[cat]
            except KeyError:
                label_spans = []
            try:
                pred_spans = article_pred_dict[cat]
            except KeyError:
                pred_spans = []
            try:
                matching_pred_spans = matching_article_pred_dict[cat]
            except KeyError:
                matching_pred_spans = []
            category_nums = HarshArticleCategoryNums(label_spans, pred_spans, matching_pred_spans)
            nums_dict[cat] = [category_nums]
            # For the 'general' category we sum all Csts_s, Cstt_s, S, T
            general_nums[0] += category_nums[0]
            general_nums[1] += category_nums[1]
            general_nums[2] += category_nums[2]
            general_nums[3] += category_nums[3]
        else:
            nums_dict[cat] = None
    nums_dict['Overall result'] = [general_nums]
    return nums_dict
    
def perDatasetNums(list_of_article_dicts, categories):
    '''
    Takes in a list of per article dicts resulting from use of perArticleNums func
    '''
    categories = categories + ['Overall result']
    final_dict = {}
    for cat in categories:
        Csts_s = []  # list of all C-funcs for precision
        Cstt_s = []  # list of all C-funcs for recall
        S = 0 # number of submitted spans
        T = 0
        category_lists = [dikt[cat] for dikt in list_of_article_dicts if dikt[cat] != None]
        for art_cat_list in category_lists:
            Csts_s += art_cat_list[0][0]
            Cstt_s += art_cat_list[0][1]
            S += art_cat_list[0][2]
            T += art_cat_list[0][3]
        final_dict[cat] = [Csts_s, Cstt_s, S, T]
        
    return final_dict


def precision(Csts_s, S):
    try:
        prec = sum([eval(i) for i in Csts_s])/S
    except ZeroDivisionError:
        prec = 0
    return prec

def recall(Cstt_s, T):
    '''Count recall based on string values of C-functions'''
    try:
        rec = sum([eval(i) for i in Cstt_s])/T
    except ZeroDivisionError:
        rec = 0
    return rec

def fscore(num_recall, num_precision):
    '''Count fscore base don numerical values of recall and precision'''
    try:
        f = 2*(num_recall*num_precision)/(num_recall + num_precision)
    except ZeroDivisionError:
        f = 0
    return f

def precision_recall_fscore_dataset(predictions_path: str, labels_path: str, categories: list):
    '''
    Takes paths for semeval 2019,2020 style predictions and labels
    and outputs a dictionary with 14 propaganda categories + 'general'
    as keys and dicts {recal, precision, fscore} as values
    '''
    article_num_dicts = []
    final_dict = {}
    all_labels = labels2dict(labels_path)
    all_preds = labels2dict(predictions_path)
    # Union of article ids prsent in either labels or predictions
    all_art_ids = list(set(list(all_preds.keys())).union(set(list(all_labels.keys()))))
    for id in all_art_ids:
        try:
            label_dict = all_labels[id]
        except KeyError:
            label_dict = {}
        try:
            pred_dict = all_preds[id]
        except KeyError:
            pred_dict = {}
        article_num_dicts.append(perArticleNums(label_dict,pred_dict, categories))
        
    final_nums = perDatasetNums(article_num_dicts, categories)
    
    for category in list(final_nums.keys()):
        num_recall = recall(final_nums[category][1], final_nums[category][3])
        num_precision = precision(final_nums[category][0], final_nums[category][2])
        num_fscore = fscore(num_recall, num_precision)
        
        final_dict[category] = {
            'precision': str(round(num_precision, 4)),
            'recall' : str(round(num_recall, 4)),
            'f-score' : str(round(num_fscore, 4))
        }
  
    return final_dict

def harsh_precision_recall_fscore_dataset(predictions_path, labels_path, categories):
    '''
    Takes paths for semeval 2019,2020 style predictions and labels
    and outputs a dictionary with 14 propaganda categories + 'general'
    as keys and dicts {recal, precision, fscore} as values
    '''
    article_num_dicts = []
    final_dict = {}
    all_labels = labels2dict(labels_path)
    all_preds = labels2dict(predictions_path)
    matching_preds = matching_spans2dikt(labels_path=labels_path, preds_path=predictions_path)
    # Union of article ids present in either labels or predictions
    all_art_ids = list(set(list(all_preds.keys())).union(set(list(all_labels.keys()))))
    for id in all_art_ids:
        try:
            label_dict = all_labels[id]
        except KeyError:
            label_dict = {}
        try:
            pred_dict = all_preds[id]
        except KeyError:
            pred_dict = {}
        try:
            matching_pred_dict = matching_preds[id]
        except KeyError:
            matching_pred_dict = {}

        article_num_dicts.append(HarshPerArticleNums(label_dict, pred_dict, matching_pred_dict, categories))

    final_nums = perDatasetNums(article_num_dicts, categories)

    for category in list(final_nums.keys()):
        num_recall = recall(final_nums[category][1], final_nums[category][3])
        num_precision = precision(final_nums[category][0], final_nums[category][2])
        num_fscore = fscore(num_recall, num_precision)

        final_dict[category] = {
            'precision': str(round(num_precision, 4)),
            'recall': str(round(num_recall, 4)),
            'f-score': str(round(num_fscore, 4))
        }

    return final_dict
