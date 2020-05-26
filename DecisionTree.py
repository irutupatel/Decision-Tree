import sys
from math import sqrt


def getHackerrankInput():
    """
    :return:
    """
    input = list()
    train_data = list()
    test_data = list()
    for line in sys.stdin.readlines():
        sample = line.strip().split()
        refined_sample = list()
        for index, item in enumerate(sample):
            if index == 0:
                label = int(sample[0])
                refined_sample.append(label)
            else:
                attribute_to_value = item.split(":")
                value = float(attribute_to_value[1])
                refined_sample.append(value)
        if int(sample[0]) == 0:
            test_data.append(refined_sample)
        else:
            train_data.append(refined_sample)
    return train_data, test_data


def getInput():
    # train_data = [[1, 1.0, 1.0], [1, 1.0, 2.0], [1, 2.0, 1.0], [3, 2.0, 2.0], [1, 3.0, 1.0], [3, 3.0, 2.0],
    #               [3, 3.0, 3.0], [3, 4.5, 3.0]]
    # test_data = [[0, 1.0, 2.2], [0, 4.5, 1.0]]

    # train_data = [[2, 3, 3, 3, 2], [1, 3, 2, 4, 2], [2, 1, 2, 2, 2],
    #               [3, 1, 4, 2, 4], [2, 1, 2, 2, 2], [3, 2, 3, 4, 3],
    #               [3, 1, 5, 2, 1], [1, 3, 2, 3, 4], [2, 2, 5, 4, 3],
    #               [2, 2, 4, 3, 3], [2, 3, 5, 3, 4], [3, 2, 3, 2, 3],
    #               [3, 1, 5, 2, 2]]
    # test_data = [[0, 1, 2, 4, 4],[0, 3, 3, 1, 4]]


    train_data = [[4, 3, 3, 3, 3, 3], [3, 1, 3, 3, 3, 2], [3, 1, 3, 3, 3, 2],
                  [4, 3, 3, 3, 3, 3], [1, 1, 1, 3, 4, 3], [4, 3, 5, 3, 2, 2],
                  [3, 1, 3, 1, 4, 3], [3, 1, 5, 1, 2, 3], [4, 2, 2, 4, 4, 2],
                  [4, 2, 2, 3, 4, 2], [4, 1, 4, 4, 4, 3], [3, 1, 3, 3, 3, 2],
                  [3, 1, 3, 3, 3, 2], [3, 1, 3, 3, 3, 2]]
    test_data = [[0, 2, 3, 1, 1, 2], [0, 3, 5, 1, 4, 1], [0, 1, 3, 1, 2, 2],
                 [0, 2, 5, 2, 2, 3], [0, 3, 4, 2, 1, 3], [0, 2, 3, 4, 4, 2]]

    return train_data, test_data


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = dict()  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[0]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Question:
    """
    Class records attribute # and attribute value to create a question such as
    "If attribute X <= threshold theta, then left node, else right node"

    Match returns True for left node, False for right node.
    """

    def __init__(self, whichAttribute, value):
        self.whichAttribute = whichAttribute
        self.value = value

    def match(self, aSample):
        value = aSample[self.whichAttribute]
        return value <= self.value


def partition_dataset(dataset, question):
    """
    :param dataset:
    :param question:
    :return:
    """
    true_rows, false_rows = list(), list()
    for aSample in dataset:
        if question.match(aSample):
            true_rows.append(aSample)
        else:
            false_rows.append(aSample)
    return true_rows, false_rows


def gini(dataset):
    """
    Calculate the Gini Impurity for a list of rows.
    """

    label_counts = class_counts(dataset)
    numberOfRows = float(len(dataset))
    gini = 1

    for label in label_counts:
        numberOfLabelOccurence = label_counts[label]
        probability_of_label = float(numberOfLabelOccurence / numberOfRows)
        gini -= (probability_of_label**2)
    return gini


def calculate_gini_attribute(left_dataset, right_dataset):

    numberOfLeftRows = float(len(left_dataset))
    numberOfRightRows = float(len(right_dataset))
    total = float(numberOfLeftRows + numberOfRightRows)
    gini_attribute = float(float(numberOfLeftRows/total)*gini(left_dataset)) + float(float(numberOfRightRows/total)*gini(right_dataset))
    return gini_attribute


def get_midpoints(values):
    """
    Finds midpoints from the list of values
    :param values: list()
    :return: midpoints : set(); Sorted!
    """
    uniqueValues = set(values)
    uniqueSortedValues = sorted(uniqueValues)
    midpoints = list()
    if len(uniqueSortedValues) == 1:
        midpoints.append(uniqueSortedValues[0])
    else:
        for index in range(1, len(uniqueSortedValues)):
            midpoint = float((uniqueSortedValues[index-1]+uniqueSortedValues[index])/float(2.0))
            midpoints.append(midpoint)

    return midpoints

def find_best_split(dataset):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""

    lowestGini = float("inf")
    splitOnAttribute = None
    leftDataset = None
    rightDataset = None
    bestMidpoint = None

    numberOfAttributes = len(dataset[0]) - 1
    for whichAttribute in range(1, numberOfAttributes+1):

        attributeValues = [aRow[whichAttribute] for aRow in dataset]

        midpoints = get_midpoints(attributeValues)

        for aMidpoint in midpoints:

            splitCondition = Question(whichAttribute, aMidpoint)

            left_dataset, right_dataset = partition_dataset(dataset, question=splitCondition)

            gini_attribute = calculate_gini_attribute(left_dataset, right_dataset)

            if gini_attribute < lowestGini:
                lowestGini = gini_attribute
                splitOnAttribute = whichAttribute
                bestMidpoint = aMidpoint
                leftDataset = left_dataset
                rightDataset = right_dataset

    return lowestGini, splitOnAttribute, bestMidpoint, leftDataset, rightDataset


class Node:

    def __init__(self):
        self.dataset = None
        self.whichAttribute = None
        self.threshold = None
        self.leaf = False
        self.result = None
        self.left_branch = None
        self.right_branch = None
        self.level = 0


class DecisionTree:
    def __init__(self, depth =2):
        self.root = None
        self.depth = 2

    def build(self, dataset, level=0):

        node = Node()

        if level == self.depth:
            node.leaf = True
            node.result = self.get_result(dataset=dataset)
            return node

        gini, whichAttribute, threshold, leftDataset, rightDataset = find_best_split(dataset=dataset)

        node.dataset = dataset
        node.whichAttribute = whichAttribute
        node.threshold = threshold
        node.level = level
        node.left_branch = self.build(dataset=leftDataset,level=level+1)
        node.right_branch = self.build(dataset=rightDataset,level=level+1)

        return node

    def get_result(self, dataset):
        highest_count = 0
        which_label = None
        label_counts = class_counts(dataset)
        for label in sorted(label_counts.keys()):
            if highest_count < label_counts[label]:
                highest_count = label_counts[label]
                which_label = label

        return which_label


def predict(sample, root):
    if root:
        if root.leaf == True:
            sample[0] = root.result
            return
        else:
            if sample[root.whichAttribute] <= root.threshold:
                predict(sample, root.left_branch)
            else:
                predict(sample, root.right_branch)


if __name__ == '__main__':

    # trainData, testData = getHackerrankInput()
    trainData, testData = getInput()
    Tree = DecisionTree()
    my_tree = Tree.build(dataset=trainData)
    for sample in testData:
        predict(sample=sample, root=my_tree)
        print (sample[0])