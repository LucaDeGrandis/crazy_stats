# WARNING: due to time constraints, this part is a bit messy and not well-organized since it is taken directly from the original repo https://github.com/psunlpgroup/MACSum
from nltk.corpus import stopwords
from collections import defaultdict
import multiprocessing
import nltk
import math


nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


st_words = stopwords.words('english')
SPE_METRICS = 'weighted'


def cal_intra(bucket_gold, bucket_pred, ordered_specificity_keys, relative=True):
    diffs = []

    for key in ordered_specificity_keys:
        length = len(bucket_gold[key])
        for i in range(length):
            gold = bucket_gold[key][i]
            pred = bucket_pred[key][i]
            diff = math.fabs(gold - pred)
            if relative:
                diff /= gold
            diffs.append(diff)

    ret = sum(diffs) / len(diffs)
    return ret


def get_specificity_value(target, metrics=SPE_METRICS):
    num_sent = len(nltk.sent_tokenize(target))
    target = nltk.word_tokenize(target.lower())
    target = [x for x in target if x not in st_words]
    target_pos = nltk.pos_tag(target)

    tot = len(target_pos)
    nn_words = [x for x, y in target_pos if y == 'NN']
    # vb_words = [x for x, y in target_pos if y == 'VB']
    vbg_words = [x for x, y in target_pos if y == 'VBG']
    cd_words = [x for x, y in target_pos if y == 'CD']
    nn = len(nn_words)
    # vb = len(vb_words)
    cd = len(cd_words)
    vbg = len(vbg_words)

    metrics = (0.1 * vbg + 0.2 * tot + 0.3 * nn + 0.4 * cd) / num_sent
    return metrics


def get_specificity_values(target, metrics=SPE_METRICS, multicore=False):
    if multicore:  # some bugs in this options
        # Get all cores
        cores = multiprocessing.cpu_count()
        # start a pool
        pool = multiprocessing.Pool(processes=cores)
        tasks = [(x, SPE_METRICS) for x in target]
        # do parallel calculate
        data = pool.starmap(get_specificity_value, tasks)
        data = [x for x in data if x is not None]

        return data
    else:
        return [get_specificity_value(x) for x in target]


def compute_cer_specificity(data, ordered_specificity_keys):
    bucket_len = defaultdict(list)
    bucket_gold = defaultdict(list)
    for i, sample in enumerate(data):
        bucket_len[sample['specificity']].append(get_specificity_value(sample['prediction']))
        bucket_gold[sample['specificity']].append(get_specificity_value(sample['summary']))

    intra_score = cal_intra(bucket_gold, bucket_len, ordered_specificity_keys)
    return intra_score


def get_spe_value_one_sample(sample):
    return get_specificity_value(sample['prediction'])


def cal_cc(pre_sample, cur_sample, class_dict):
    pre_len = class_dict[pre_sample['specificity']]
    cur_len = class_dict[cur_sample['specificity']]
    pre_score = get_spe_value_one_sample(pre_sample)
    cur_score = get_spe_value_one_sample(cur_sample)
    return (pre_score - cur_score) / (pre_len - cur_len)


def compute_cc_specificity(samples, class_dict):
    spe_cvs = []

    for i, sample in enumerate(samples):
        if i == 0:
            continue
        previous_doc = samples[i - 1]['text_in']
        previous_tpk = samples[i - 1]['topic']
        previous_spe = samples[i - 1]['specificity']
        cur_doc = sample['text_in']
        cur_tpk = sample['topic']
        cur_spe = sample['specificity']

        if previous_tpk != cur_tpk or previous_doc != cur_doc:
            continue

        if previous_spe != cur_spe:
            spe_cvs.append(cal_cc(sample, samples[i - 1], class_dict))

    return sum(spe_cvs) / len(spe_cvs)
