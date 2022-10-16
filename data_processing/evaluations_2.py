import math

def KLdivergence(g_lengths_list, p_lengths_list, bin_size: int = 5):
    """Computes Kullback–Leibler divergence between true and predicted lenghts of propaganda spans"""

    all_possible_lengths = list(set(g_lengths_list + p_lengths_list))
    sorted_lengths = sorted(all_possible_lengths)
    num_bins = int((sorted_lengths[-1] - sorted_lengths[0]) / bin_size)
    predicted_distribution = []
    gold_distribution = []
    for bin_id in range(num_bins):
        gold_lengths = [i for i in g_lengths_list if
                        (sorted_lengths[0] + bin_id * bin_size) <= i < sorted_lengths[0] + (bin_id + 1) * bin_size]
        predicted_lengths = [i for i in p_lengths_list if
                             (sorted_lengths[0] + bin_id * bin_size) <= i < sorted_lengths[0] + (bin_id + 1) * bin_size]
        predicted_distribution.append(len(predicted_lengths))
        gold_distribution.append(len(gold_lengths))
    # smoothing
    smoothed_gold_distribution = [i if i != 0 else 0.1 for i in gold_distribution]
    smoothed_golds = sum(smoothed_gold_distribution)
    smoothed_predicted_distribution = [i if i != 0 else 0.1 for i in predicted_distribution]
    smoothed_preds = sum(smoothed_predicted_distribution)
    final_gold_distribution = [i / smoothed_golds for i in smoothed_gold_distribution]
    final_predicted_distribution = [i / smoothed_preds for i in smoothed_predicted_distribution]
    # KL
    re = []
    for p, g in zip(final_predicted_distribution, final_gold_distribution):
        re.append(float(p * math.log2(p / g)))
    KLD = sum(re)

    return KLD


def KLdivergence_with_None(g_lengths_list, p_lengths_list, bin_size: int = 5):
    """Computes Kullback–Leibler divergence between true and predicted lenghts of propaganda spans
        None option added for debugging purposes"""

    if len(p_lengths_list) == 0:
        return None
    else:
        # first gold then predicted
        all_possible_lengths = list(set(g_lengths_list + p_lengths_list))
        sorted_lengths = sorted(all_possible_lengths)
        num_bins = int((sorted_lengths[-1] - sorted_lengths[0]) / bin_size)
        predicted_distribution = []
        gold_distribution = []
        for bin_id in range(num_bins):
            gold_lengths = [i for i in g_lengths_list if
                            (sorted_lengths[0] + bin_id * bin_size) <= i < sorted_lengths[0] + (bin_id + 1) * bin_size]
            predicted_lengths = [i for i in p_lengths_list if
                                 (sorted_lengths[0] + bin_id * bin_size) <= i < sorted_lengths[0] + (
                                             bin_id + 1) * bin_size]
            predicted_distribution.append(len(predicted_lengths))
            gold_distribution.append(len(gold_lengths))
        # smoothing
        smoothed_gold_distribution = [i if i != 0 else 0.1 for i in gold_distribution]
        smoothed_golds = sum(smoothed_gold_distribution)
        smoothed_predicted_distribution = [i if i != 0 else 0.1 for i in predicted_distribution]
        smoothed_preds = sum(smoothed_predicted_distribution)
        final_gold_distribution = [i / smoothed_golds for i in smoothed_gold_distribution]
        final_predicted_distribution = [i / smoothed_preds for i in smoothed_predicted_distribution]
        # print(sum(final_gold_distribution))
        # print(sum(final_predicted_distribution))
        # KL
        re = []
        for p, g in zip(final_predicted_distribution, final_gold_distribution):
            re.append(float(p * math.log2(p / g)))
        KLD = sum(re)

        return KLD

