# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:06:23 2018

@author: Reet Barik
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
import matplotlib.pyplot as plt


# Load data from CSV
def load_data(train_path, test_path, dev_path):
    col_names = ["age", "workclass", "education", "relationship", "profession", "race", "gender", "workhours",
                 "nationality", "income"]
    train_data = pd.read_csv(train_path, header=None, names=col_names)
    test_data = pd.read_csv(test_path, header=None, names=col_names)
    dev_data = pd.read_csv(dev_path, header=None, names=col_names)

    print("Training Data Loaded.")
    print("Total training instances:", len(train_data))
    print("Total testing instances:", len(test_data))
    print("Total dev instances:", len(dev_data), "\n")

    return train_data, test_data, dev_data


# Standardize the numerical columns
def standardize_data(train_data, test_data, dev_data):
    # Fit scaler on train data only. Transform training and testing set
    numerical_col = ["age", "workhours"]
    scaler = StandardScaler()
    train_data[numerical_col] = scaler.fit_transform(train_data[numerical_col])
    test_data[numerical_col] = scaler.transform(test_data[numerical_col])
    dev_data[numerical_col] = scaler.transform(dev_data[numerical_col])

    return train_data, test_data, dev_data


# Split the data into features and labels
def split_data(train_data, test_data, dev_data):
    y_train = train_data["income"]
    x_train = train_data.drop("income", axis=1)

    y_test = test_data['income']
    x_test = test_data.drop("income", axis=1)

    y_dev = dev_data['income']
    x_dev = dev_data.drop("income", axis=1)

    return x_train, y_train, x_test, y_test, x_dev, y_dev


# One hot encoding of categorical data
def ohe_data(x_train, y_train, x_test, y_test, x_dev, y_dev):
    data = pd.concat([x_train, x_test, x_dev])
    
    age_group = []
    for age in data["age"]:
        if age < 0:
            age_group.append("age1")
        elif 0 <= age <= 2:
            age_group.append("age2")
        else:
            age_group.append("age3")
            
    data_ohe_age = data.copy()
    data_ohe_age["age_group"] = age_group
    del data_ohe_age["age"]
    
    workinghours = []
    for w in data_ohe_age["workhours"]:
        if age < -1.75:
            workinghours.append("w1")
        elif -1.75 <= age <= 0.5:
            workinghours.append("w2")
        elif 0.5 <= age <= 2.75:
            workinghours.append("w3")
        else:
            workinghours.append("w4")
            
    data_ohe = data_ohe_age.copy()
    data_ohe["workinghours"] = workinghours
    del data_ohe["workhours"]
    
    
    data_ohe = pd.get_dummies(data, drop_first=True)
    x_train_ohe = data_ohe[:len(x_train)]
    x_test_ohe = data_ohe[len(x_train):len(x_test) + len(x_train)]
    x_dev_ohe = data_ohe[len(x_test) + len(x_train):]

    y_train_ohe = y_train.replace([' <=50K', ' >50K'], [-1, 1])
    y_test_ohe = y_test.replace([' <=50K', ' >50K'], [-1, 1])
    y_dev_ohe = y_dev.replace([' <=50K', ' >50K'], [-1, 1])

    return x_train_ohe, y_train_ohe, x_test_ohe, y_test_ohe, x_dev_ohe, y_dev_ohe


# Bringing all pre-processing steps together
def pre_process_data(train_data, test_data, dev_data):
    train_data, test_data, dev_data = standardize_data(train_data, test_data, dev_data)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split_data(train_data, test_data, dev_data)
    x_train_ohe, y_train_ohe, x_test_ohe, y_test_ohe, x_dev_ohe, y_dev_ohe = ohe_data(x_train, y_train, x_test, y_test,
                                                                                      x_dev, y_dev)

    return x_train_ohe, y_train_ohe, x_test_ohe, y_test_ohe, x_dev_ohe, y_dev_ohe


# Plot Train, Test and Dev accuracy vs Size of ensemble/No. of Iterations for different Tree depths
def plot_accuracy(depth, bagging_tree_number, bagging_train_accuracy, bagging_test_accuracy, bagging_dev_accuracy, bagging = True):
    plt.plot(bagging_tree_number, bagging_train_accuracy, color='orange', label='Training', marker='o')
    plt.plot(bagging_tree_number, bagging_test_accuracy, color='green', label='Testing', marker='o')
    plt.plot(bagging_tree_number, bagging_dev_accuracy, color='blue', label='Dev', marker='o')
    plt.legend(loc='best')
    if(bagging is True):
        plt.xlabel('Size of ensemble')
        plt.title('Plot of Accuracy vs. size of ensemble of depth ' + str(depth))
        plt.ylabel('Bagging Accuracy')
    else: 
        plt.xlabel('Number of Iterations')
        plt.title('Plot of Accuracy vs. No. of Iterations of depth ' + str(depth))
        plt.ylabel('Boosting Accuracy')
    
    plt.show()


# Run the sklearn Bagging classifier
def do_bagging(x_train, y_train, x_test, y_test, x_dev, y_dev):
    
    # Hyperparameters
    bagging_tree_depth = [1,2,3,5,10]
    bagging_tree_number = [10,20,40,60,80,100]
    
    print("\na)Sklearn Bagging: \n")
    for depth in bagging_tree_depth:
        bagging_train_accuracy = []
        bagging_test_accuracy = []
        bagging_dev_accuracy = []
        for num in bagging_tree_number:        
            bagging = BaggingClassifier(DecisionTreeClassifier(max_depth = depth),max_samples=0.5,max_features=1.0,n_estimators = num)
            bagging = bagging.fit(x_train, y_train)
            train = bagging.score(x_train, y_train)
            test = bagging.score(x_test, y_test)
            dev = bagging.score(x_dev, y_dev)
            bagging_train_accuracy.append(train)
            bagging_test_accuracy.append(test)
            bagging_dev_accuracy.append(dev)
            
            print("Depth = " + str(depth) + ", Ensemble size = " + str(num) + "\n")
            print("Training Accuracy: " + str(round(train, 4)) + ",")
            print("Testing Accuracy: " + str(round(test, 4)) + ",")
            print("Dev Accuracy: " + str(round(dev, 4)))
            print("\n")
            
        plot_accuracy(depth, bagging_tree_number, bagging_train_accuracy, bagging_test_accuracy, bagging_dev_accuracy)
        print("\n")
            

# Run the sklearn AdaBoost classifier
def do_boosting(x_train, y_train, x_test, y_test, x_dev, y_dev):
    
    # Hyperparameters
    boosting_tree_depth = [1,2,3]
    boosting_iterations = [10,20,40,60,80,100]
    
    print("\nb)Sklearn AdaBoost: \n")
    for depth in boosting_tree_depth:
        boosting_train_accuracy = []
        boosting_test_accuracy = []
        boosting_dev_accuracy = []
        for itr in boosting_iterations: 
            boosting = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = depth),n_estimators = itr)
            boosting = boosting.fit(x_train, y_train)
            train = boosting.score(x_train, y_train)
            test = boosting.score(x_test, y_test)
            dev = boosting.score(x_dev, y_dev)
            boosting_train_accuracy.append(train)
            boosting_test_accuracy.append(test)
            boosting_dev_accuracy.append(dev)
            
            print("Depth = " + str(depth) + ", Number of Iterations = " + str(itr) + "\n")
            print("Training Accuracy: " + str(round(train, 4)) + ",")
            print("Testing Accuracy: " + str(round(test, 4)) + ",")
            print("Dev Accuracy: " + str(round(dev, 4)))
            print("\n")
            
        plot_accuracy(depth, boosting_iterations, boosting_train_accuracy, boosting_test_accuracy, boosting_dev_accuracy, False)
        print("\n")
              

if __name__ == "__main__":
    
    np.random.seed(10)
    
    # Loading the Training, Testing and Development Data
    train_data, test_data, dev_data = load_data("income-data/income.train.txt", "income-data/income.test.txt",
                                                "income-data/income.dev.txt")
    
    # Binary Conversion and Pre Processing of the Data
    x_train, y_train, x_test, y_test, x_dev, y_dev = pre_process_data(train_data, test_data, dev_data)

    # Sklearn Bagging
    do_bagging(x_train, y_train, x_test, y_test, x_dev, y_dev)
    
    # Sklearn Boosting
    do_boosting(x_train, y_train, x_test, y_test, x_dev, y_dev)
