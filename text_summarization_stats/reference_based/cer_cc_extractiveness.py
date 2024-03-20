import os
import math
cwd = os.getcwd()
os.chdir('/content/crazy_stats/MACSum/metric')
from rouge.evaluator import EvaluateTool
os.chdir(cwd)


METRICS = ['rouge-2', 'rouge-3']


def get_extractiveness_values(target, source, metrics=None):
    if metrics is None:
        metrics = ['rouge-1', 'rouge-2']

    rouge_evaluator = EvaluateTool()
    target_values = []
    for pred, source in zip(target, source):
        # check if precision is correct
        cur_rouge_score = rouge_evaluator.evaluate_list_fast([pred], [source], metrics=metrics)
        cur_rouge_avg = []
        for x in cur_rouge_score:
            all_metrics_scores = [x[metric]['p'] for metric in metrics]
            cur_rouge_avg.append(sum(all_metrics_scores)/len(all_metrics_scores))
        target_values.append(cur_rouge_avg[0])
    return target_values


def cal_diff(gold_list, pred_list, relative=True):
    diffs = []
    for gold, pred in zip(gold_list, pred_list):
        diff = math.fabs(gold - pred)
        if relative:
            diff /= (gold + 0.1)
        diffs.append(diff)
    ret = sum(diffs) / len(diffs)
    return ret


def get_ext_score(samples):
    bucket_gold = []
    for sample in samples:
        bucket_gold.append(get_extractiveness_values([sample['summary']], [sample['text_in']], metrics=METRICS)[0])
    bucket_pred = []
    for sample in samples:
        bucket_pred.append(get_extractiveness_values([sample['prediction']], [sample['text_in']], metrics=METRICS)[0])

    score = cal_diff(bucket_gold, bucket_pred)

    return score
