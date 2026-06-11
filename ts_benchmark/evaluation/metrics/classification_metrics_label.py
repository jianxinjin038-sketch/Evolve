# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics

from ts_benchmark.evaluation.metrics.affiliation.generics import convert_vector_to_events
from ts_benchmark.evaluation.metrics.vus_metrics import metricor, generate_curve
from ts_benchmark.evaluation.metrics.affiliation.metrics import pr_from_events
from ts_benchmark.evaluation.metrics.vus_metrics import metricor
from ts_benchmark.evaluation.metrics.utils import get_list_anomaly, find_length
from sklearn.metrics import accuracy_score

__all__ = [
    "accuracy",
    "f_score",
    "precision",
    "recall",
    "adjust_accuracy",
    "adjust_f_score",
    "adjust_precision",
    "adjust_recall",
    "rrecall",
    "rprecision",
    "rf",
    "affiliation_f",
    "affiliation_precision",
    "affiliation_recall",

    "auc_roc", 
    "auc_pr", 
    "R_AUC_ROC", 
    "R_AUC_PR", 
    "VUS_ROC", 
    "VUS_PR"
]


metricor_grader = metricor()


def adjust_predicts(actual: np.ndarray, predicted: np.ndarray, **kwargs) -> np.ndarray:
    """
    调整检测结果
    异常检测算法在一个异常区间检测到某点存在异常，则认为算法检测到整个异常区间的所有异常点
    先从检测到的异常点从后往前调整检测结果，随后再从该点从前往后调整检测结果，直到真实的异常为False
    退出异常状态，结束当前区间的调整

    :param actual: 真实的异常。
    :param predicted: 检测所得的异常。
    :return: 调整后的异常检测结果。
    """
    predicted = predicted.copy()
    anomaly_state = False
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if actual[j] == 0:
                    break
                else:
                    if predicted[j] == 0:
                        predicted[j] = 1
            for j in range(i, len(actual)):
                if actual[j] == 0:
                    break
                else:
                    if predicted[j] == 0:
                        predicted[j] = 1
        elif actual[i] == 0:
            anomaly_state = False
        if anomaly_state:
            predicted[i] = 1
    return predicted


# score = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
# label = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# print(adjust_predicts(score, label))

#####改，参考merlion



# def adjust_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     predicted = adjust_predicts(actual, predicted)
#     Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
#         actual, predicted, zero_division=0
#     )
#     return Precision[1]


# def adjust_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     predicted = adjust_predicts(actual, predicted)
#     Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
#         actual, predicted, zero_division=0
#     )
#     return Recall[1]


# def adjust_f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     predicted = adjust_predicts(actual, predicted)
#     Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
#         actual, predicted, zero_division=0
#     )
#     return F[1]


# def adjust_accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     predicted = adjust_predicts(actual, predicted)
#     accuracy = accuracy_score(actual, predicted)
#     return accuracy


# def precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
#         actual, predicted, zero_division=0
#     )
#     return Precision[1]


# def recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
#         actual, predicted, zero_division=0
#     )
#     return Recall[1]


# def f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
#         actual, predicted, zero_division=0
#     )
#     return F[1]


# def accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     accuracy = accuracy_score(actual, predicted)
#     return accuracy


# def rrecall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     (
#         AUC_ROC,
#         Precision,
#         Recall,
#         F,
#         Rrecall,
#         ExistenceReward,
#         OverlapReward,
#         Rprecision,
#         RF,
#         Precision_at_k,
#     ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
#     return Rrecall


# def rprecision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     (
#         AUC_ROC,
#         Precision,
#         Recall,
#         F,
#         Rrecall,
#         ExistenceReward,
#         OverlapReward,
#         Rprecision,
#         RF,
#         Precision_at_k,
#     ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
#     return Rprecision


# def rf(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     (
#         AUC_ROC,
#         Precision,
#         Recall,
#         F,
#         Rrecall,
#         ExistenceReward,
#         OverlapReward,
#         Rprecision,
#         RF,
#         Precision_at_k,
#     ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
#     return RF


# def precision_at_k(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     (
#         AUC_ROC,
#         Precision,
#         Recall,
#         F,
#         Rrecall,
#         ExistenceReward,
#         OverlapReward,
#         Rprecision,
#         RF,
#         Precision_at_k,
#     ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
#     return Precision_at_k

# def affiliation_f(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     events_pred = convert_vector_to_events(predicted)
#     events_label = convert_vector_to_events(actual)
#     Trange = (0, len(predicted))

#     result = pr_from_events(events_pred, events_label, Trange)
#     P = result['precision']
#     R = result['recall']
#     F = 2 * P * R / (P + R)

#     return F
# def affiliation_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     events_pred = convert_vector_to_events(predicted)
#     events_label = convert_vector_to_events(actual)
#     Trange = (0, len(predicted))

#     result = pr_from_events(events_pred, events_label, Trange)
#     P = result['precision']
#     R = result['recall']
#     F = 2 * P * R / (P + R)

#     return P

# def affiliation_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     events_pred = convert_vector_to_events(predicted)
#     events_label = convert_vector_to_events(actual)
#     Trange = (0, len(predicted))

#     result = pr_from_events(events_pred, events_label, Trange)
#     P = result['precision']
#     R = result['recall']
#     F = 2 * P * R / (P + R)

#     return R


# # score = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# # label = np.array([0, 1, 1, 0, 1, 1, 1, 1])
# #
# #
# # print("precision:", precision(label, score,))
# # print("recall:", recall(label, score))
# # print("f_score:", f_score(label, score,))
# # print("rrecall:", rrecall(label, score,))
# # print("rprecision:", rprecision(label, score,))
# # print("rf:", rf(label, score,))
# # print("precision_at_k:", precision_at_k(label, score,))



def adjust_precision(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    predicted = adjust_predicts(actual, predicted)
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Precision[1]


def adjust_recall(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    predicted = adjust_predicts(actual, predicted)
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Recall[1]


def adjust_f_score(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    predicted = adjust_predicts(actual, predicted)
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return F[1]


def adjust_accuracy(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    predicted = adjust_predicts(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    return accuracy


def precision(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Precision[1]


def recall(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Recall[1]


def f_score(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return F[1]


def accuracy(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    accuracy = accuracy_score(actual, predicted)
    return accuracy


def rrecall(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Rrecall


def rprecision(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Rprecision


def rf(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return RF


def precision_at_k(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Precision_at_k

def affiliation_f(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return F
def affiliation_precision(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return P

def affiliation_recall(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return R



def auc_roc(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    if ratio == 1 or type(ratio) is not int:
        return metrics.roc_auc_score(actual, another)
    return 0


def auc_pr(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    if ratio == 1 or type(ratio) is not int:
        return metrics.average_precision_score(actual, another)
    return 0


def R_AUC_ROC(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    if ratio == 1 or type(ratio) is not int:
        slidingWindow = int(np.median(get_list_anomaly(actual)))
        # slidingWindow = 100
        R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
            labels=actual, score=another, window=slidingWindow, plot_ROC=True
        )
        return R_AUC_ROC
    return 0


def R_AUC_PR(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    if ratio == 1 or type(ratio) is not int:
        slidingWindow = int(np.median(get_list_anomaly(actual)))
        # slidingWindow = 100
        R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
            labels=actual, score=another, window=slidingWindow, plot_ROC=True
        )
        return R_AUC_PR
    return 0


def VUS_ROC(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    if ratio == 1 or type(ratio) is not int:
        slidingWindow = int(np.median(get_list_anomaly(actual)))
        # slidingWindow = 100

        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
            actual, another, 2 * slidingWindow
        )
        return VUS_ROC
    return 0


def VUS_PR(ratio: int, actual: np.ndarray, predicted: np.ndarray, another: np.ndarray, **kwargs):
    if ratio == 1 or type(ratio) is not int:
        slidingWindow = int(np.median(get_list_anomaly(actual)))
        # slidingWindow = 100

        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
            actual, another, 2 * slidingWindow
        )
        return VUS_PR
    return 0