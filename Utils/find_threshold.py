import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, \
    RocCurveDisplay, ConfusionMatrixDisplay


def find_threshold(rmse_real, rmse_fake):
    """
    Finds the best threshold for determining real vs fake data based on Root Mean Squared Error (RMSE) values.
    The function calculates the ROC AUC score and accuracy for different threshold values, and prints the best
    threshold along with its ROC AUC score and accuracy. Also plots the histograms for the RMSE of real and fake data.
        :param rmse_real: List of RMSE values for real data
        :param rmse_fake: List of RMSE values for fake data
    """
    all = rmse_real + rmse_fake

    true_labels = []
    for i in range(0, len(rmse_fake)):
        true_labels.append(0)
    for i in range(0, len(rmse_real)):
        true_labels.append(1)

    num_all = len(rmse_real) + len(rmse_fake)

    dd = np.arange(min(all), max(all), 0.001)
    best_roc_auc_score = 0.0
    best_acc = 0.0
    best_t = -1

    for threshold in dd:
        right_pred = 0
        pred_labels = []

        for j in rmse_fake:
            if j > threshold:
                right_pred += 1
                pred_labels.append(0)
            else:
                pred_labels.append(1)

        for i in rmse_real:
            if i < threshold:
                right_pred += 1
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        if roc_auc_score(true_labels, pred_labels) > best_roc_auc_score:
            best_roc_auc_score = roc_auc_score(true_labels, pred_labels)
            best_t = threshold
            best_acc = right_pred / num_all

    print(best_roc_auc_score, best_t, best_acc)

    n, bins, patches = plt.hist(
        x=rmse_real, bins='auto', color='#0504aa',
        alpha=0.7, rwidth=0.85
        )
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.title('Histogramm of mean RMSE for sequences of Real Videos')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    plt.figure()
    n, bins, patches = plt.hist(
        x=rmse_fake, bins='auto', color='#0504aa',
        alpha=0.7, rwidth=0.85
        )
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.title('Histogramm of mean RMSE for sequences of Fake Videos')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    plt.show()
