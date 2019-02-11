# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:34:31 2018

@author: Reet Barik
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import warnings


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


# Loading the Training, Testing and Development Data
train_data, test_data, dev_data = load_data("income-data/income.train.txt", "income-data/income.test.txt",
                                                "income-data/income.dev.txt")
    
# Binary Conversion and Pre Processing of the Data
x_train, y_train, x_test, y_test, x_dev, y_dev = pre_process_data(train_data, test_data, dev_data)


def plot_bo_performance(stat_func_values, bagging = True):
    x = list(range(1, len(stat_func_values) + 1))
    plt.plot(x, stat_func_values, color='b', label='f(x)', marker='o')
    plt.legend(loc='best')
    plt.xlabel('No. of BO iterations')
    plt.ylabel('f(x)')
    if(bagging is True): 
        plt.title('Performance vs No. of BO iterations for Bagging')
    else: 
        plt.title('Performance vs No. of BO iterations for Boosting')
    plt.show()
    
# Objective function for Bagging    
def bagging_evaluate(depth, num):

    bagging = BaggingClassifier(DecisionTreeClassifier(max_depth = int(depth)),max_samples=0.5,max_features=1.0,n_estimators = int(num))
    bagging = bagging.fit(x_train, y_train)    

    return bagging.score(x_dev, y_dev)

# Bayesian Optimization for Bagging
def bagging_optimize():
    
    print("Searching for Hyperparameters for Bagging via Bayesian Optimization:\n")
    opt = BayesianOptimization(bagging_evaluate, {'depth': (1, 10), 'num': (10, 100)})    
    opt.maximize(init_points=1, n_iter=49, acq='ei')
    
    return opt

# Objective function for Boosting   
def boosting_evaluate(depth, itr):
    
    boosting = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = int(depth)),n_estimators = int(itr))
    boosting = boosting.fit(x_train, y_train)
    
    return boosting.score(x_dev, y_dev)


# Bayesian Optimization for Boosting
def boosting_optimize():
    
    print("Searching for Hyperparameters for Boosting via Bayesian Optimization:\n")
    opt = BayesianOptimization(boosting_evaluate, {'depth': (1, 3), 'itr': (10, 100)})    
    opt.maximize(init_points=1, n_iter=49, acq='ei')
    
    return opt



# Output Bayesian Optimization Results for Bagging/Boosting  
def bo_result(bayes_opt_obj, bagging = True):
    
    if (bagging is True):
        plot_bo_performance(bayes_opt_obj.Y)
    else:
        plot_bo_performance(bayes_opt_obj.Y, False)
    print('\n')
    
    if (bagging is True):
        print("List of candidate Hyper-parameters for Bagging:\n")
    else:
        print("List of candidate Hyper-parameters for Boosting:\n")    
    
    for i in range(len(bayes_opt_obj.res['all']['params'])):
        print('Depth: ' + str(round(bayes_opt_obj.res['all']['params'][i]['depth'])))
        if (bagging is True):
            print('Size of ensemble: ' + str(round(bayes_opt_obj.res['all']['params'][i]['num'])))
        else:
            print('Number of iterations: ' + str(round(bayes_opt_obj.res['all']['params'][i]['itr'])))
        print('f(x): ' + str(bayes_opt_obj.res['all']['values'][i]))
        print('\n')
    
    if (bagging is True):
        print('Hyperparameters chosen for Bagging by Bayesian Optimization:\n')
    else:
        print('Hyperparameters chosen for Boosting by Bayesian Optimization:\n')
    print('Depth: ' + str(round(bayes_opt_obj.res['max']['max_params']['depth'])))
    if (bagging is True):
        print('Size of ensemble: ' + str(round(bayes_opt_obj.res['max']['max_params']['num'])))
    else: 
        print('Number of iterations: ' + str(round(bayes_opt_obj.res['max']['max_params']['itr'])))
    print('\n')



if __name__ == "__main__":
    
    np.random.seed(10)
    
    # Bayesian Optimization for Bagging
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bayes_opt_obj = bagging_optimize()
        
    bo_result(bayes_opt_obj)
    
    # Bayesian Optimization for Boosting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bayes_opt_obj = boosting_optimize()
        
    bo_result(bayes_opt_obj, False)   