import csv

import numpy
import pandas as pd
from LDA import LDA
from Decidion_Tree import Decidion_Tree

if __name__ == '__main__':

    data = [[2.771244718, 1.784783929, 0],
            [1.728571309, 1.169761413, 0],
            [3.678319846, 2.81281357, 0],
            [3.961043357, 2.61995032, 0],
            [5.999208922, 2.209014212, 0],
            [1.728571309, 1.169761413, 0],
            [4.678319846, 2.81281357, 0],
            [1.961043357, 2.61995032, 0],
            [3.769208922, 2.209014212, 0],
            [7.497545867, 3.162953546, 1],
            [9.00220326, 3.339047188, 1],
            [7.444542326, 0.476683375, 1],
            [10.12493903, 3.234550982, 1],
            [6.642287351, 3.319983761, 1],
            [7.497545867, 3.162953546, 1],
            [9.00220326, 3.339047188, 1],
            [5.444542326, 0.476683375, 1],
            [9.12493903, 3.234550982, 1],
            [8.642287351, 3.319983761, 1]
            ]

    chunks = 6
    max_tree_lvl = 4
    min_tree_size = 10
    binary_tree = Decidion_Tree(max_tree_lvl, min_tree_size)
    chunks = binary_tree.split_into_validation_data(data, chunks)
    class_scores = list()
    for i in range(2):
        train_data = list(chunks)
        train_data.remove(chunks[i])
        train_data = sum(train_data, [])
        test_data = list()
        for row in chunks[i]:
            row_copy = list(row)
            test_data.append(row_copy)
            row_copy[-1] = None
        predicted, tree = binary_tree.decision_tree_classifier(train_data, test_data)
        actual = [row[-1] for row in chunks[i]]
        accuracy = binary_tree.prediction_accuracy(actual, predicted)
        class_scores.append(accuracy)

    print('Class Scores: %s' % class_scores)
    print('Mean Accuracy: %.3f%%' % (sum(class_scores) / float(len(class_scores))))


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def setWineQualityClasses(data):
    labels = []
    for i in range(len(data)):
        if data[i][-1] <= 5:
            labels.append(1)
        elif data[i][-1] > 5 and data[i][-1] <= 6:
            labels.append(2)
        else:
            labels.append(3)
    return labels

def addLabels(data, labels):
    for i in range(len(data)):
        data[i].append(labels[i])


if __name__ == '__main__':


    lda = LDA(data2, descriptors=3)
    traindata, testdata = lda.fit()

    lda.gaussian_modeling()
    trainerror = lda.calculate_score_gaussian(traindata) / float(traindata.shape[0])
    testerror = lda.calculate_score_gaussian(testdata) / float(testdata.shape[0])
    print(trainerror)
    print(testerror)
    lda.plot_proj_2D(data2)

