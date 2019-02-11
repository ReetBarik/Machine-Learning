import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from sklearn import svm

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


# def clean_data(train_data, test_data, dev_data):
#
#     # Replace all " ?" with NaN and then drop rows where NaN appears
#     train_clean = train_data.replace(' ?', np.nan).dropna()
#     test_clean = test_data.replace(' ?', np.nan).dropna()
#     dev_clean = dev_data.replace(' ?', np.nan).dropna()
#
#     print("Number of training instances removed:", len(train_data) - len(train_clean))
#     print("Number of testing instances removed:", len(test_data) - len(test_clean))
#     print("Number of dev instances removed:", len(dev_data) - len(dev_clean))
#     print("Total training instances:", len(train_clean))
#     print("Total testing instances:", len(test_clean))
#     print("Total dev instances:", len(dev_clean), "\n")
#
#     return train_clean, test_clean, dev_clean


def standardize_data(train_data, test_data, dev_data):
    # Fit scaler on train data only. Transform training and testing set
    numerical_col = ["age", "workhours"]
    scaler = StandardScaler()
    train_data[numerical_col] = scaler.fit_transform(train_data[numerical_col])
    test_data[numerical_col] = scaler.transform(test_data[numerical_col])
    dev_data[numerical_col] = scaler.transform(dev_data[numerical_col])

    return train_data, test_data, dev_data


def split_data(train_data, test_data, dev_data):
    y_train = train_data["income"]
    x_train = train_data.drop("income", axis=1)

    y_test = test_data['income']
    x_test = test_data.drop("income", axis=1)

    y_dev = dev_data['income']
    x_dev = dev_data.drop("income", axis=1)

    return x_train, y_train, x_test, y_test, x_dev, y_dev


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


def remove_zero_cols(x_train_ohe, x_test_ohe, x_dev_ohe):
    x_train_ohe = x_train_ohe.replace(0, np.nan)
    x_train_ohe = x_train_ohe.dropna(axis=1, thresh=500)
    x_train_ohe = x_train_ohe.replace(np.nan, 0)

    cols = x_train_ohe.columns
    x_test_ohe = x_test_ohe[cols]
    x_dev_ohe = x_dev_ohe[cols]

    return x_train_ohe, x_test_ohe, x_dev_ohe


def pre_process_data(train_data, test_data, dev_data):
    train_data, test_data, dev_data = standardize_data(train_data, test_data, dev_data)
    x_train, y_train, x_test, y_test, x_dev, y_dev = split_data(train_data, test_data, dev_data)
    x_train_ohe, y_train_ohe, x_test_ohe, y_test_ohe, x_dev_ohe, y_dev_ohe = ohe_data(x_train, y_train, x_test, y_test,
                                                                                      x_dev, y_dev)
    x_train_ohe, x_test_ohe, x_dev_ohe = remove_zero_cols(x_train_ohe, x_test_ohe, x_dev_ohe)

    return x_train_ohe, y_train_ohe, x_test_ohe, y_test_ohe, x_dev_ohe, y_dev_ohe


def kernelized_perceptron(iterations, x_train, y_train):
    alpha = np.zeros(x_train.shape[0])
    K = np.zeros((x_train.shape[0],x_train.shape[0]))
    
    mistakes = []
    mistake_count = 0
    
    K = (1 + np.dot(x_train, x_train.T)) ** 2
    
    for i in range(iterations):
        count = 0
        for x, y in zip(x_train.values, y_train.values):
            y_hat = np.sign(np.sum(alpha * y_train.values * K[:][count]))
            
            if y_hat != y:
                alpha[count] = alpha[count] + 1
                mistake_count += 1
                            
            count += 1
        mistakes.append(mistake_count)
        mistake_count = 0
    return alpha, mistakes


def kernelized_prediction(alpha, x_train, y_train, x_test, y_test, x_dev, y_dev):
    # Training accuracy
    for x in x_train.values:
        y_hat_train = np.sign(np.sum(alpha * y_train.values * ((1 + np.dot(x_train.values, x)) ** 2)))
    correct_train = np.sum((y_hat_train == np.array(y_train)).astype(int))
    kernelized_training_accuracy = round((float(correct_train) / x_train.shape[0]) * 100, 2)
    
    # Testing accuracy
    for x in x_test.values:
        y_hat_test = np.sign(np.sum(alpha * y_train.values * ((1 + np.dot(x_train.values, x)) ** 2)))
    correct_test = np.sum((y_hat_test == np.array(y_test)).astype(int))
    kernelized_testing_accuracy = round((float(correct_test) / x_test.shape[0]) * 100, 2)
    
    # Dev accuracy
    for x in x_dev.values:
        y_hat_dev = np.sign(np.sum(alpha * y_train.values * ((1 + np.dot(x_train.values, x)) ** 2)))
    correct_dev = np.sum((y_hat_dev == np.array(y_dev)).astype(int))
    kernelized_dev_accuracy = round((float(correct_dev) / x_dev.shape[0]) * 100, 2)
    
    return kernelized_training_accuracy, kernelized_testing_accuracy, kernelized_dev_accuracy
    

def plot_kernel_mistakes(kernel_mistakes):
    
    iterations = [1,2,3,4,5]
    plt.plot(iterations, kernel_mistakes, color='orange', marker='o')
    plt.xlabel('No. of iterations')
    plt.ylabel('No. of mistakes')
    plt.title('No. of mistakes on training set as a function of the iterations')
    plt.show()

def plot_sv_count(sv_count_negative, sv_count_positive):
    C = ["10**-4", "10**-3", "10**-2", "10**-1", "10**0", "10**1", "10**2", "10**3", "10**4"]
    plt.plot(C, sv_count_negative, color='orange', label='<=50K', marker='o')
    plt.plot(C, sv_count_positive, color='b', label='>50K', marker='o')
    plt.legend(loc='best')
    plt.xlabel('C Parameter Values')
    plt.ylabel('No. of Support Vectors')
    plt.title('No. of Support Vectors as a function of the C Value')
    plt.show()
    
    
def plot_accuracy(train_accuracy, test_accuracy, dev_accuracy):
    C = ["10**-4", "10**-3", "10**-2", "10**-1", "10**0", "10**1", "10**2", "10**3", "10**4"]
    plt.plot(C, train_accuracy, color='orange', label='Training', marker='o')
    plt.plot(C, test_accuracy, color='b', label='Testing', marker='o')
    plt.plot(C, dev_accuracy, color='g', label='Dev', marker='o')
    plt.legend(loc='best')
    plt.xlabel('C Parameter Values')
    plt.ylabel('Accuracy')
    plt.title('Training, Testing and Dev Accuracy as a function of the C Value')
    plt.show()
    

def sv_count_mixed(alg, sv_count_negative_mixed, sv_count_positive_mixed):
    plt.plot(alg, sv_count_negative_mixed, color='orange', label='<=50K', marker='o')
    plt.plot(alg, sv_count_positive_mixed, color='b', label='>50K', marker='o')
    plt.legend(loc='best')
    plt.xlabel('Kernels')
    plt.ylabel('No. of Support Vectors')
    plt.title('No. of Support Vectors for SVM Classifiers with different Kernels')
    plt.show()
    

def plot_accuracy_mixed(alg, train_accuracy, test_accuracy, dev_accuracy):
    plt.plot(alg, train_accuracy, color='orange', label='Training', marker='o')
    plt.plot(alg, test_accuracy, color='b', label='Testing', marker='o')
    plt.plot(alg, dev_accuracy, color='g', label='Dev', marker='o')
    plt.legend(loc='best')
    plt.xlabel('Kernels')
    plt.ylabel('Accuracy')
    plt.title('Training, Testing and Dev Accuracy for SVM Classifiers with different Kernels ')
    plt.show()
    

if __name__ == "__main__":
    
    np.random.seed(10)
    
    # Loading the Training, Testing and Development Data
    train_data, test_data, dev_data = load_data("income-data/income.train.txt", "income-data/income.test.txt",
                                                "income-data/income.dev.txt")
    
    # Binary Conversion and Pre Processing of the Data
    x_train, y_train, x_test, y_test, x_dev, y_dev = pre_process_data(train_data, test_data, dev_data)
    
    train_accuracy = []
    test_accuracy = []
    dev_accuracy = []
    sv_count_negative = []
    sv_count_positive = []
    
    # C Parameter Values
    C = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
    
    # Learning Linear SVM classifiers for different C values
    for i in C:
        clf = svm.SVC(C = i, kernel='linear')
        clf.fit(x_train, y_train)
        train_accuracy.append(clf.score(x_train, y_train))
        test_accuracy.append(clf.score(x_test, y_test))
        dev_accuracy.append(clf.score(x_dev, y_dev))
        sv_count_negative.append(clf.n_support_[0])
        sv_count_positive.append(clf.n_support_[1])
        
    # Plot of No. of Support Vectors as a function of C values
    plot_sv_count(sv_count_negative, sv_count_positive)
    
    # Plot Training, Testing, Dev Accuracy as a function of C values
    plot_accuracy(train_accuracy, test_accuracy, dev_accuracy)
    
    # Combined Set of Training and Validation Data set
    x_train_dev = pd.concat([x_train, x_dev])
    y_train_dev = pd.concat([y_train, y_dev])
    
    # Best C = 10 ^ -2 based on accuracy on validation set
    clf = svm.SVC(C = 10**-2, kernel='linear')
    clf.fit(x_train_dev, y_train_dev)
    
    # Accuracy on Testing Data
    combined_testing_accuracy = clf.score(x_test, y_test)
    
    # Predictions on Testing Data
    predictions = clf.predict(x_test)
    
    train_accuracy_mixed = []
    test_accuracy_mixed = []
    dev_accuracy_mixed = []
    
    sv_count_negative_mixed = []
    sv_count_positive_mixed = []
    
    # Linear Kernel SVM with best C = 10^-2
    clf = svm.SVC(C = 10**-2, kernel='linear')
    clf.fit(x_train, y_train)
    
    train_accuracy_mixed.append(clf.score(x_train, y_train))
    test_accuracy_mixed.append(clf.score(x_test, y_test))
    dev_accuracy_mixed.append(clf.score(x_dev, y_dev))
    
    sv_count_negative_mixed.append(clf.n_support_[0])
    sv_count_positive_mixed.append(clf.n_support_[1])
    
    # Polynomial Kernel SVM with best C = 10^-2 and Degree = 2
    clf = svm.SVC(C = 10**-2, kernel='poly', degree = 2)
    clf.fit(x_train, y_train)
    
    train_accuracy_mixed.append(clf.score(x_train, y_train))
    test_accuracy_mixed.append(clf.score(x_test, y_test))
    dev_accuracy_mixed.append(clf.score(x_dev, y_dev))
    
    sv_count_negative_mixed.append(clf.n_support_[0])
    sv_count_positive_mixed.append(clf.n_support_[1])
    
    # Polynomial Kernel SVM with best C = 10^-2 and Degree = 3
    clf = svm.SVC(C = 10**-2, kernel='poly', degree = 3)
    clf.fit(x_train, y_train)
    
    train_accuracy_mixed.append(clf.score(x_train, y_train))
    test_accuracy_mixed.append(clf.score(x_test, y_test))
    dev_accuracy_mixed.append(clf.score(x_dev, y_dev))
    
    sv_count_negative_mixed.append(clf.n_support_[0])
    sv_count_positive_mixed.append(clf.n_support_[1])
    
    # Polynomial Kernel SVM with best C = 10^-2 and Degree = 4
    clf = svm.SVC(C = 10**-2, kernel='poly', degree = 4)
    clf.fit(x_train, y_train)
    
    train_accuracy_mixed.append(clf.score(x_train, y_train))
    test_accuracy_mixed.append(clf.score(x_test, y_test))
    dev_accuracy_mixed.append(clf.score(x_dev, y_dev))
    
    sv_count_negative_mixed.append(clf.n_support_[0])
    sv_count_positive_mixed.append(clf.n_support_[1])
    
    # Mixed Support Vector Count Plot
    alg = ['Linear', 'Degree 2', 'Degree 3', 'Degree 4']
    sv_count_mixed(alg, sv_count_negative_mixed, sv_count_positive_mixed)
    
    # Accuracy plots for different Kernels
    plot_accuracy_mixed(alg, train_accuracy_mixed, test_accuracy_mixed, dev_accuracy_mixed)
    
    # General learning curve for Kernelized perceptron
    alpha , kernelized_mistakes = kernelized_perceptron(5, x_train, y_train)
    plot_kernel_mistakes(kernelized_mistakes)
    k_train_accuracy, k_test_accuracy, k_dev_accuracy = kernelized_prediction(alpha, x_train, y_train, x_test, y_test, x_dev, y_dev)
    
    
    with open('output.txt', 'w') as f:
        print('Answer 14', file=f)
        print(' ', file=f)
        print('Training accuracy as a function of C: ', train_accuracy,file=f)
        print('Testing accuracy as a function of C: ', test_accuracy,file=f)
        print('Dev accuracy as a function of C: ', dev_accuracy,file=f)
        print('Support Vectors as a function of C: ', dev_accuracy,file=f)
        print(' ', file=f)
        print('Testing accuracy on combined set of Train and Dev data: ', combined_testing_accuracy,file=f)
        print(' ', file=f)
        print('Confusion Matrix: ', confusion_matrix(y_test, predictions), file=f)
        print(' ', file=f)
        print('Training accuracy for different Kernels: ', train_accuracy_mixed,file=f)
        print('Testing accuracy for different Kernels: ', test_accuracy_mixed,file=f)
        print('Dev accuracy for different Kernels: ', dev_accuracy_mixed,file=f)
        print('Support Vectors as a function of C: ', sv_count_negative_mixed,file=f)
        print(' ', file=f)
        print('Training Accuracy of Kernelized Perceptron: ', k_train_accuracy, file=f)
        print('Training Accuracy of Kernelized Perceptron: ', k_test_accuracy, file=f)
        print('Training Accuracy of Kernelized Perceptron: ', k_dev_accuracy, file=f)
        print(' ', file=f)
        print(' ', file=f)
    

