import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from mpl_toolkits.mplot3d import Axes3D

split_ratio = 0.9

class LDA:
    def __init__(self, data, descriptors=1, label_column=-1):
        self.data = data
        self.descriptors = descriptors
        self.label_column = label_column

    def remove_label_column(self, data, col):
        return data.drop(data.columns[[col]], axis=1)

    def estimate_lda_params(self, data):
        means = {}
        for c in self.classes:
            # calculate for columns
            tmp_data_mean = self.remove_label_column(self.class_data_samples[c], self.label_column).mean(axis=0)
            means[c] = np.array(tmp_data_mean)

        overall_mean = np.array(self.remove_label_column(data, self.label_column).mean(axis=0))

        # calculate between class covariance matrix
        S_B = np.zeros((data.shape[1] - 1, data.shape[1] - 1))
        for c in means.keys():
            N = len(self.class_data_samples[c])
            scatter_mean = np.outer((means[c] - overall_mean), (means[c] - overall_mean))
            S_B += np.multiply(N, scatter_mean)

        # calculate within class covariance matrix
        S_W = np.zeros(S_B.shape)
        for c in self.classes:
            class_data_transpose = self.remove_label_column(self.class_data_samples[c], self.label_column).T
            tmp_data = np.subtract(class_data_transpose, np.expand_dims(means[c], axis=1))  # taking away mean value
            cov_mat = np.cov(tmp_data)
            S_W = np.add(cov_mat, S_W)

        invSw_bySb = np.dot(np.linalg.pinv(S_W), S_B)
        eigvals, eigvecs = np.linalg.eig(invSw_bySb)
        eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

        # sort the eigvals in decreasing order
        eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

        # take the first descriptors eigvectors
        w = np.array([eiglist[i][1] for i in range(self.descriptors)])

        self.w = w
        self.means = means
        return

    def fit(self):
        traindata = []
        testdata = []
        grouped = self.data.groupby(self.data.ix[:, self.label_column])

        self.classes = [c for c in grouped.groups.keys()]
        self.class_data_samples = {}
        for c in self.classes:
            self.class_data_samples[c] = grouped.get_group(c)
            index_list = list(self.class_data_samples[c].index)
            splitting_border = int(self.class_data_samples[c].shape[0] * split_ratio)
            rows = random.sample(index_list, splitting_border)
            traindata.append(self.class_data_samples[c].ix[rows])
            testdata.append(self.class_data_samples[c].drop(rows))

        traindata = pd.concat(traindata)
        testdata = pd.concat(testdata)

        # estimate the LDA parameters
        self.estimate_lda_params(traindata)

        return traindata, testdata

    def gaussian_modeling(self):
        self.priors = {}
        self.gaussian_means = {}
        self.gaussian_cov = {}

        for c in self.means.keys():
            input_data = self.remove_label_column(self.class_data_samples[c], self.label_column)
            projection = np.dot(self.w, input_data.T).T
            self.priors[c] = input_data.shape[0] / float(self.data.shape[0])
            self.gaussian_means[c] = np.mean(projection, axis=0)
            self.gaussian_cov[c] = np.cov(projection, rowvar=False)

    def pdf(self, sample, mean, cov):
        cons = 1. / ((2 * np.pi) ** (len(sample) / 2.) * np.linalg.det(cov) ** (-0.5))
        exponent = np.exp(-np.dot(np.dot((sample - mean), np.linalg.inv(cov)), (sample - mean).T) / 2.)
        return cons * exponent

    def calculate_score_gaussian(self, data):
        classes = sorted(list(self.means.keys()))
        input_data = self.remove_label_column(data, self.label_column)
        projection = np.dot(self.w, input_data.T).T
        # calculate the likelihoods for each class based on the gaussian models
        likelihoods = np.array([
            [self.priors[c] * self.pdf(
                [x[index] for index in range(len(x))],
                self.gaussian_means[c],
                self.gaussian_cov[c])
             for c in classes]
            for x in projection])
        labels = np.argmax(likelihoods, axis=1)
        errors = np.sum(labels != data.ix[:, self.label_column])
        return errors

    def plot_proj_3D(self, data):
        classes = list(self.means.keys())
        color_map = cm.rainbow(np.linspace(0, 1, len(classes)))
        plotlabels = {classes[c]: color_map[c] for c in range(len(classes))}

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, row in data.iterrows():
            projection = np.dot(self.w, row[:self.label_column])
            projection = np.real(projection)
            ax.scatter(projection[0],projection[1],projection[2],color=plotlabels[row[self.label_column]])
        plt.show()

    def plot_proj_2D(self, data):
        classes = list(self.means.keys())
        color_map = cm.rainbow(np.linspace(0, 1, len(classes)))
        plotlabels = {classes[c]: color_map[c] for c in range(len(classes))}

        fig = plt.figure()
        for i, row in data.iterrows():
            projection = np.dot(self.w, row[:self.label_column])
            projection = np.real(projection)
            plt.scatter(projection[0], projection[1], color=plotlabels[row[self.label_column]])
        plt.show()