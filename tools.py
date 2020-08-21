# Adopted from a code base from https://github.com/cetinsamet
# --------------------------------------------------
import random
random.seed(123)
import numpy as np
np.random.seed(123)

import scipy.io as sio
from sklearn.metrics import confusion_matrix
import json


def load_json(file_path):
    with open(file_path) as f:
        params = json.load(f)

    print(json.dumps(params, indent=4))
    return params

def load_data(data, dataName):
    """ Load data from .mat files """

    dataContent = sio.loadmat(data)
    dataContent = dataContent[dataName]

    return dataContent

def normalized_accuracy_zsl(all_predictions, all_labels):

    cm_gzsl = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm_gzsl.astype('float') / cm_gzsl.sum(axis=1)[:, np.newaxis]
    cm_overall_normalized_acc = np.sum(cm_normalized.diagonal()) / float(len(cm_normalized.diagonal()))

    return cm_overall_normalized_acc

def harmonic_score_gzsl(all_predictions, all_labels, unique_lab_seen, unique_lab_unseen):

    cm_gzsl = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm_gzsl.astype('float') / cm_gzsl.sum(axis=1)[:, np.newaxis]

    diagonal_vals = cm_normalized[:].diagonal()
    known_class_range = unique_lab_seen.astype(int)
    unknown_class_range = unique_lab_unseen.astype(int)

    known_classes_avg_accuracy = np.sum(diagonal_vals[known_class_range]) / float(len(diagonal_vals[known_class_range]))
    unknown_classes_avg_accuracy = np.sum(diagonal_vals[unknown_class_range]) / float(len(diagonal_vals[unknown_class_range]))

    h_score = (known_classes_avg_accuracy * unknown_classes_avg_accuracy  * 2) / (known_classes_avg_accuracy + unknown_classes_avg_accuracy)

    return known_classes_avg_accuracy, unknown_classes_avg_accuracy, h_score


