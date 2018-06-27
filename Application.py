import csv
import random

import pandas as pd
from LDA import LDA
from Decidion_Tree import Decidion_Tree

if __name__ == '__main__':
    data = []
    filename = 'iris_data.csv'

    chunks = 6
    max_tree_lvl = 4
    min_tree_size = 10
    binary_tree = Decidion_Tree(max_tree_lvl, min_tree_size)
    binary_tree.loadDataSet(filename, data)
    chunks = binary_tree.split_into_validation_data(data, chunks)
    class_scores = list()
    for i in range(6):
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
    #print(random.seed(1))
    print('Class Scores: %s' % class_scores)
    print('Mean Accuracy: %.3f%%' % (sum(class_scores) / float(len(class_scores))))

if __name__ == '__main__':
    data = pd.read_csv('digits.csv')
    data = data[data['0.29'] <= 4]
    lda = LDA(data, descriptors=3)
    traindata, testdata = lda.fit()

    lda.gaussian_modeling()
    trainerror = lda.calculate_score_gaussian(traindata) / float(traindata.shape[0])
    testerror = lda.calculate_score_gaussian(testdata) / float(testdata.shape[0])
    print(trainerror)
    print(testerror)
    lda.plot_proj_2D(data)
