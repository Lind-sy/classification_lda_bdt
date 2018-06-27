import csv
from random import seed
from random import randrange


class Decidion_Tree():

    def __init__(self, max_tree_lvl, min_tree_size):
        self.max_tree_lvl = max_tree_lvl
        self.min_tree_size = min_tree_size
        seed(1)
        print('SEED',seed(1))

    def split_into_groups(self, index, value, data):
        first, second = list(), list()
        for row in data:
            if row[index] > value:
                second.append(row)
            else:
                first.append(row)
        return first, second

    def calculate_gini_index(self, groups, classes):
        sample_size = float(sum([len(group) for group in groups]))
        gini_index = 0.0
        for group in groups:
            group_sample_size = float(len(group))
            score = 0.0
            if group_sample_size != 0:
                for class_sample in classes:
                    p = [row[-1] for row in group].count(class_sample) / group_sample_size
                    score += p * p
                gini_index += (1.0 - score) * (group_sample_size / sample_size)
            else:
                continue
        return gini_index

    def split_into_classes(self, data):
        class_values = list(set(row[-1] for row in data))
        best_index = 100
        best_value = 100
        best_score = 100
        best_groups = None
        gini_index = 0
        for index in range(len(data[0]) - 1):
            for row in data:
                groups = self.split_into_groups(index, row[index], data)
                gini_index = self.calculate_gini_index(groups, class_values)
                if gini_index < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini_index, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups, 'gini': gini_index}

    def set_terminal_value(self, group):
        result = [row[-1] for row in group]
        return max(set(result), key=result.count)

    def create_tree_structure(self, node, level):
        left, right = node['groups']
        del (node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.set_terminal_value(left + right)
            return
        if level >= self.max_tree_lvl:
            node['left'], node['right'] = self.set_terminal_value(left), self.set_terminal_value(right)
            return
        if len(left) <= self.min_tree_size:
            node['left'] = self.set_terminal_value(left)
        else:
            node['left'] = self.split_into_classes(left)
            self.create_tree_structure(node['left'], level + 1)
        if len(right) <= self.min_tree_size:
            node['right'] = self.set_terminal_value(right)
        else:
            node['right'] = self.split_into_classes(right)
            self.create_tree_structure(node['right'], level + 1)

    def build_decision_tree(self, train):
        root_data = self.split_into_classes(train)
        self.create_tree_structure(root_data, 1)
        return root_data

    def predict_class(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_class(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_class(node['right'], row)
            else:
                return node['right']

    def decision_tree_classifier(self, train, test):
        tree = self.build_decision_tree(train)
        predictions = list()
        for row in test:
            prediction = self.predict_class(tree, row)
            predictions.append(prediction)
        return predictions, tree

    def split_into_validation_data(self, data, chunks):
        data_split = list()
        data_copy = list(data)
        chunk_size = int(len(data) / chunks)
        for i in range(chunks):
            chunk = list()
            while len(chunk) < chunk_size:
                index = randrange(len(data_copy))
                chunk.append(data_copy.pop(index))
            data_split.append(chunk)
        return data_split

    def prediction_accuracy(self, actual_labels, predicted_labels):
        counter = 0
        for i in range(len(actual_labels)):
            if actual_labels[i] == predicted_labels[i]:
                counter += 1
        return counter / float(len(actual_labels)) * 100.0

    def loadDataSet(self, filename, data=[]):
        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)
            for x in range(len(dataset)):
                dataset[x] = [float(i) for i in dataset[x]]
                data.append(dataset[x])
