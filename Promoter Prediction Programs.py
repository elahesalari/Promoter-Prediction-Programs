import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import time
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression




def preprocess():
        promoter = pd.read_excel('Dataset/promoter.xls')
        non_promoter = pd.read_excel('Dataset/non_promoter.xls')
        promoter = np.array(promoter['Promoter sequences'])
        non_promoter = np.array(non_promoter['non Promoter sequences'])

        n_feature = len(non_promoter[0])


        new = []
        for i in range(promoter.shape[0]):
            if 'N' in promoter[i]:
                promoter[i] = promoter[i].replace('N', '')
            if len(promoter[i]) >= n_feature:
                new.append(promoter[i][:n_feature])

        promoter = np.array(new)

        bendability = pd.read_csv('Conversion table/bendability.tsv', sep='\t', header=None)
        bendability = np.array(bendability)

        # Create numeric dataset and sliding window
        window_size = 10

        sliding_promoter = []
        for i in range(promoter.shape[0]):
            num_promoter = []
            sliding_promoter.append([])
            for j in range(len(promoter[i]) - 2):
                sep = promoter[i][j:j + 3].lower()
                for k in range(bendability.shape[0]):
                    if sep in bendability[k, 0]:
                        num_promoter.append(bendability[k, 1])

            for w in range(len(num_promoter) - (window_size - 1)):
                win_pro = np.sum(num_promoter[w:w + window_size])
                sliding_promoter[i].append(float(f"{win_pro:.3f}"))

        sliding_promoter = np.array(sliding_promoter)
        print(sliding_promoter.shape)
        np.save('Sliding window/Sliding_promoter10.npy', sliding_promoter)

        sliding_non = []
        for i in range(non_promoter.shape[0]):
            num_non = []
            sliding_non.append([])
            for j in range(len(non_promoter[i]) - 2):
                sep = non_promoter[i][j:j + 3].lower()
                for k in range(bendability.shape[0]):
                    if sep in bendability[k, 0]:
                        num_non.append(bendability[k, 1])

            for w in range(len(num_non) - (window_size - 1)):
                win_non = np.sum(num_non[w:w + window_size])
                sliding_non[i].append(float(f"{win_non:.3f}"))

        sliding_non = np.array(sliding_non)
        print(sliding_non.shape)
        np.save('Sliding window/Sliding_non-promoter10.npy', sliding_non)


def classifier_SVM(x_train, x_test, y_train, y_test):
        svm_classifier = SVC(kernel='rbf')
        svm_classifier.fit(x_train, y_train)
        print('finish training')
        y_pred = svm_classifier.predict(x_test)
        print(f'time for each fold:', time.perf_counter() // 60 )
        accuracy = metrics.accuracy_score(y_test, y_pred) * 100
        print('accuracy:', accuracy)
        conf = confusion_matrix(y_test, y_pred)
        TN = conf[0, 0]
        FP = conf[0, 1]
        FN = conf[1, 0]
        TP = conf[1, 1]
        print('confusion:', conf)
        precision = TP / (TP + FN)
        recall = TP / (TP + FP)
        f_score = 2 * ((precision * recall) / (precision + recall))

        return accuracy, precision, recall, f_score

def logistic_regression(x_train, x_test, y_train, y_test):
        model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print('accuracy:', accuracy)
        conf = confusion_matrix(y_test, y_pred)
        TN = conf[0, 0]
        FP = conf[0, 1]
        FN = conf[1, 0]
        TP = conf[1, 1]
        print('confusion:', conf)
        precision = TP / (TP + FN)
        recall = TP / (TP + FP)
        f_score = 2 * ((precision * recall) / (precision + recall))
        return accuracy, precision, recall, f_score

def train():
        promoter = np.load('Sliding window/Sliding_promoter.npy')
        non_promoter = np.load('Sliding window/Sliding_non-promoter.npy')
        promoter = np.hstack((promoter, np.ones((promoter.shape[0], 1), dtype=promoter.dtype)))
        non_promoter = np.hstack((non_promoter, - np.ones((non_promoter.shape[0], 1), dtype=non_promoter.dtype)))

        dataset = np.concatenate((promoter, non_promoter), axis=0)

        # Dimentionality Reduction with PCA
        x = dataset[:, :-1]

        pca = PCA(n_components=0.9)
        reduce = pca.fit_transform(x)

        X = reduce
        Y = dataset[:, -1]

        print(X.shape)
        accuracy = []
        precision = []
        recall = []
        f_score = []
        # 5 Fold cross validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        for train_id, test_id in kfold.split(X):
            x_train, x_test = X[train_id], X[test_id]
            y_train, y_test = Y[train_id], Y[test_id]

            # SVM Classification
            acc, pre, rec, f = classifier_SVM(x_train, x_test, y_train, y_test)

            # Logistic regression
            # acc, pre, rec, f = logistic_regression(x_train, x_test, y_train, y_test)

            accuracy.append(acc)
            precision.append(pre)
            recall.append(rec)
            f_score.append(f)
        accuracy = np.mean(accuracy)
        precision = np.mean(precision)
        recall = np.mean(recall)
        f_score = np.mean(f_score)

        print('accuracy:', accuracy)
        print('precision:', precision)
        print('recall:', recall)
        print('f_score:', f_score)




if __name__ == '__main__':
    preprocess()
    train()
