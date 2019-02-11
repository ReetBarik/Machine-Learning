# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 01:51:06 2018

@author: Reet Barik
"""

import numpy as np
import pandas as pd



def load_data(train_path, test_path, dev_path):
    col_names = ["age", "workclass", "education", "relationship", "profession", "race", "gender", "workhours",
                 "nationality", "income"]
    train_data = pd.read_csv(train_path, header=None, names=col_names)
    test_data = pd.read_csv(test_path, header=None, names=col_names)
    dev_data = pd.read_csv(dev_path, header=None, names=col_names)


    print("Q.10. Training Data Loaded.")
    print("Total training instances:", len(train_data))
    print("Total testing instances:", len(test_data))
    print("Total dev instances:", len(dev_data), "\n")

    data = pd.concat([train_data, dev_data, test_data])
    X, Y = data.drop('income', axis=1), data['income']

    x_train, y_train = X[:len(train_data)], Y[:len(train_data)]
    x_dev, y_dev = X[len(train_data):len(dev_data) + len(train_data)], Y[len(train_data):len(dev_data) + len(train_data)]
    x_test, y_test = X[len(dev_data) + len(train_data):], Y[len(dev_data) + len(train_data):]

    df = x_train.duplicated(keep = False)
    y_train[df == True] = y_train.iloc[0]


    return x_train, y_train, x_test, y_test, x_dev, y_dev

# Calculate the total entropy of the dataset. Here 'data' contains the labels of the dataset
def calculate_entropy(data):

    values, frequency = np.unique(data, return_counts=True)
    probability = frequency / len(data)
    return -probability.dot(np.log2(probability))

# Calculate Gain = Entropy(X) - Entorpy(X | Z)
def calculate_information_gain(data, labels):
    values, frequencies = np.unique(data, return_counts=True)
    summation = 0.0
    for i in range(len(values)):
        summation += frequencies[i] * calculate_entropy(labels[data == values[i]])

    return calculate_entropy(labels) - summation / len(labels)


# Each object of this class stores 1 node of the Decision Tree
class id3:
    # prediction at each node. Stores the label if it is a leaf node. 'None' otherwise.
    prediction = None

    # attribute to split on based on the value of information gain. Is equal to 'None' in case of leaf nodes.
    split_attribute = None

    # the value of the split attribute for that branch. Contains the previous split_attribute_value in the parent node.
    split_attribute_value = None

    # list of child nodes
    children = None

    # Select the best attribute to split on based on the Information Gain
    def find_split_attribute(self, data, labels):
        highest_information_gain = -1
        for a in data.columns:
            information_gain = calculate_information_gain(data[a].values, labels.values)
            if (information_gain > highest_information_gain):
                highest_information_gain = information_gain
                self.split_attribute = a
                
                
    # Construct the Decision Tree: 
    def construct(self, data, labels):
        self.find_split_attribute(data, labels)
        
        self.children = []
        for val in data[self.split_attribute].unique().tolist():
            subset = data[data[self.split_attribute] == val]
            subset_labels = labels[data[self.split_attribute] == val]
            subset = subset.drop(columns=[self.split_attribute])

            self.children.append(id3(subset, subset_labels, at_val=val))
    
    
    # Predict the label of 1 example at a time
    def predict(self, data):
        if self.prediction is not None and self.children is None:
            return self.prediction
        
        else:
            for c in self.children:
                    if  data[self.split_attribute].values[0] == c.split_attribute_value:
                        var = c.predict(data)
                        return var

    # Predict the accuracy
    def accuracy(self, y_pred, y):
        return len([i for i, val in enumerate(y_pred) if val == y[i]])/len(y)  # Credit for this line goes to Sriyandas

    
    #Initialize the decision tree
    def __init__(self, data, labels, at_val=None):
        if at_val is not None:
            self.split_attribute_value = at_val

        if len(labels.unique()) == 1:
            self.prediction = labels.unique()[0]
            return
                    
        self.construct(data, labels)
        return


if __name__ == "__main__":
    
    # Load data
    x_train, y_train, x_test, y_test, x_dev, y_dev = load_data("income.train.txt", "income.test.txt", "income.dev.txt")
    
    # Learn the tree
    tree = id3(x_train, y_train)
    
    # Training Accuracy
    y_train_pred = []
    for i in range(x_train.shape[0]):
        y_train_pred.append(tree.predict(x_train.iloc[i:i+1]))

    train_accuracy = tree.accuracy(y_train_pred, y_train)
    
    # Testing Accuracy
    y_test_pred = []
    for i in range(x_test.shape[0]):
        y_test_pred.append(tree.predict(x_test.iloc[i:i+1]))

    test_accuracy = tree.accuracy(y_test_pred, y_test)
    
    # Dev Accuracy
    y_dev_pred = []
    for i in range(x_dev.shape[0]):
        y_dev_pred.append(tree.predict(x_dev.iloc[i:i+1]))

    dev_accuracy = tree.accuracy(y_dev_pred, y_dev)  
    
    print("\nID3 Decision Tree:\n")
    print("Training accuracy: " + str(train_accuracy) + "\n")
    print("Testing accuracy: " + str(test_accuracy) + "\n")
    print("Dev accuracy: " + str(dev_accuracy) + "\n")
