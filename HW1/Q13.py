import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

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
    # x_train_ohe, x_test_ohe, x_dev_ohe = remove_zero_cols(x_train_ohe, x_test_ohe, x_dev_ohe)

    return x_train_ohe, y_train_ohe, x_test_ohe, y_test_ohe, x_dev_ohe, y_dev_ohe


def cal_accuracy(w, x, y, b=0):
    if b == 0:
        b = np.zeros_like(np.dot(w, x.T))

    y_hat = np.sign(np.dot(w, x.T) + b)
    correct = np.sum((y_hat == np.array(y)).astype(int))
    # print("No. of correct predictions:", correct, " out of ", x.shape[0], "\n")
    # print("Accuracy:", round((float(correct) / x.shape[0]) * 100, 2), "\n")
    return round((float(correct) / x.shape[0]) * 100, 2)


def perceptron_standard(iterations, x_train, y_train, x_test, y_test, x_dev, y_dev):
    train_accuracy = []
    test_accuracy = []
    dev_accuracy = []

    mistakes = []

    w = np.zeros(x_train.shape[1])
    tau = 1

    for i in range(iterations):
        count = 0

        for x, y in zip(x_train.values, y_train):
            y_hat = np.sign(np.dot(w, x))
            # print (y_hat)
            if y_hat != y:
                count += 1
                w = w + tau * y * x

        mistakes.append(count)
      
        train_accuracy.append(cal_accuracy(w, x_train, y_train))
        test_accuracy.append(cal_accuracy(w, x_test, y_test))
        dev_accuracy.append(cal_accuracy(w, x_dev, y_dev))
    
    return mistakes, train_accuracy, test_accuracy, dev_accuracy


def perceptron_pa(iterations, x_train, y_train, x_test, y_test, x_dev, y_dev):
    train_accuracy = []
    test_accuracy = []
    dev_accuracy = []

    mistakes = []

    w = np.zeros(x_train.shape[1])


    for i in range(iterations):
        count = 0

        for x, y in zip(x_train.values, y_train):
            y_hat = np.sign(np.dot(w, x))
            if y_hat != y:
                count += 1
                tau = float(1 - (y * np.dot(w, x))) / np.linalg.norm(x, ord=1)**2
                w = w + tau * y * x

        mistakes.append(count)

        train_accuracy.append(cal_accuracy(w, x_train, y_train))
        test_accuracy.append(cal_accuracy(w, x_test, y_test))
        dev_accuracy.append(cal_accuracy(w, x_dev, y_dev))
    
    return mistakes, train_accuracy, test_accuracy, dev_accuracy


def average_perceptron_naive(iterations, x_train, y_train, x_test, y_test, x_dev, y_dev):
    train_accuracy = []
    test_accuracy = []
    dev_accuracy = []
    
    w = np.zeros(x_train.shape[1])
    w_sum = np.zeros(x_train.shape[1])
    tau = 1
    count = 0
    
    for i in range(iterations):
        for x, y in zip(x_train.values, y_train):
            y_hat = np.sign(np.dot(w, x))
            if y_hat != y:
                w = w + tau * y * x
                w_sum += w
                count += 1
        w_avg = w_sum / count
        
        train_accuracy.append(cal_accuracy(w_avg, x_train, y_train))
        test_accuracy.append(cal_accuracy(w_avg, x_test, y_test))
        dev_accuracy.append(cal_accuracy(w_avg, x_dev, y_dev))
        
    return train_accuracy, test_accuracy, dev_accuracy

def average_perceptron_smart(iterations, x_train, y_train, x_test, y_test, x_dev, y_dev):
    train_accuracy = []
    test_accuracy = []
    dev_accuracy = []
    
    train_count = 0
    
    test_accuracy_general = []
    dev_accuracy_general = []
    
    w = np.zeros(x_train.shape[1])
    v = np.zeros(x_train.shape[1])
    b = 0
    a = 0
    c = 1
    
    for i in range(iterations):
        for x, y in zip(x_train.values, y_train):
            if y * (np.dot(w, x) + b) <= 0:
                w = w + y * x
                b = b + y
                v = v + y * c * x
                a = a + y * c
            c = c + 1
            
            train_count += 1
        
            if(train_count == (i + 1) * 5000):
                test_accuracy_general.append(cal_accuracy((w-(v / c)), x_test, y_test, (b-(a / c))))
                dev_accuracy_general.append(cal_accuracy((w-(v / c)), x_dev, y_dev, (b-(a / c))))
                
        train_count = 0
        
        train_accuracy.append(cal_accuracy((w-(v / c)), x_train, y_train, (b-(a / c))))
        test_accuracy.append(cal_accuracy((w-(v / c)), x_test, y_test, (b-(a / c))))
        dev_accuracy.append(cal_accuracy((w-(v / c)), x_dev, y_dev, (b-(a / c))))
        
    return train_accuracy, test_accuracy, dev_accuracy, test_accuracy_general, dev_accuracy_general


def online_learning_curve(mistakes_standard, mistakes_pa):
    iterations = [1,2,3,4,5]
    plt.plot(iterations, mistakes_standard, color='g', label='Standard Perceptron', marker='o')
    plt.plot(iterations, mistakes_pa, color='orange', label='PA Algorithm', marker='o')
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('No. of Mistakes')
    plt.title('Online Learning Curve of Standard Perceptron and PA Algorithm')
    plt.show()
    
    
def plot_accuracy(train_accuracy, test_accuracy, dev_accuracy, alg):
    iterations = [1,2,3,4,5]
    plt.plot(iterations, train_accuracy, color='g', label='Training', marker='o')
    plt.plot(iterations, test_accuracy, color='orange', label='Testing', marker='o')
    plt.plot(iterations, dev_accuracy, color='b', label='Dev')
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    if(alg == 1):
        plt.title('Accuracy of training, testing and dev data for Standard Perceptron')
    if(alg == 0):
        plt.title('Accuracy of training, testing and dev data for PA Algorithm')
    if(alg == 2):
        plt.title('Accuracy of training, testing and dev data for Averaged Classifier')
    plt.show()
    
    
def plot_general_learning(test_accuracy, dev_accuracy):
    x = [5000, 10000, 15000, 20000, 25000]
    plt.plot(x, test_accuracy, color='orange', label='Testing', marker='o')
    plt.plot(x, dev_accuracy, color='b', label='Dev', marker='o')
    plt.legend(loc='best')
    plt.xlabel('No. of training examples')
    plt.ylabel('Accuracy')
    plt.title('General Learning Curve of testing and dev data for Averaged Classifier')
    plt.show()
    


if __name__ == "__main__":
    
    np.random.seed(10)
    
    # Loading the Training, Testing and Development Data
    train_data, test_data, dev_data = load_data("income-data/income.train.txt", "income-data/income.test.txt",
                                                "income-data/income.dev.txt")
    
    # Binary Conversion and Pre Processing of the Data
    x_train, y_train, x_test, y_test, x_dev, y_dev = pre_process_data(train_data, test_data, dev_data)

    # Standard Perceptron
    mistakes_standard, train_accuracy_standard, test_accuracy_standard, dev_accuracy_standard = perceptron_standard(5, x_train, y_train, x_test, y_test, x_dev, y_dev)
    
    # Passive Aggressive Algorithm
    mistakes_pa, train_accuracy_pa, test_accuracy_pa, dev_accuracy_pa = perceptron_pa(5, x_train, y_train, x_test, y_test, x_dev, y_dev)
    
    # Online learning curve for both Perceptron and PA algorithm
    online_learning_curve(mistakes_standard, mistakes_pa)
    
    # Accuracy plot of Standard Perceptron
    plot_accuracy(train_accuracy_standard, test_accuracy_standard, dev_accuracy_standard, 1)
    
    # Accuracy plot of Passive Aggressive Algorithm
    plot_accuracy(train_accuracy_pa, test_accuracy_pa, dev_accuracy_pa, 0)
    
    # Naive Average Perceptron
    start_time = time.time()
    train_accuracy_naive, test_accuracy_naive, dev_accuracy_naive = average_perceptron_naive(5, x_train, y_train, x_test, y_test, x_dev, y_dev)
    time_naive = time.time() - start_time
    
    # Smart Average Perceptron
    start_time = time.time()
    train_accuracy_smart, test_accuracy_smart, dev_accuracy_smart, test_accuracy_general, dev_accuracy_general = average_perceptron_smart(5, x_train, y_train, x_test, y_test, x_dev, y_dev)
    time_smart = time.time() - start_time
    
    # Accuracy plot Average Perceptron
    plot_accuracy(train_accuracy_smart, test_accuracy_smart, dev_accuracy_smart, 2)
    
    # General Learning Curve of Average Perceptron
    plot_general_learning(test_accuracy_general, dev_accuracy_general)
    
    # Writing to output file: 
    
    with open('output.txt', 'w') as f:
        print('Answer 13', file=f)
        print(' ', file=f)
        print('Accuracy of Perceptron on training data: ', train_accuracy_standard.pop(),file=f)
        print('Accuracy of Perceptron on testing data: ', test_accuracy_standard.pop(),file=f)
        print('Accuracy of Perceptron on dev data: ', dev_accuracy_standard.pop(),file=f)
        print(' ', file=f)
        print('Accuracy of PA on training data: ', train_accuracy_pa.pop(),file=f)
        print('Accuracy of PA on testing data: ', test_accuracy_pa.pop(),file=f)
        print('Accuracy of PA on dev data: ', dev_accuracy_pa.pop(),file=f)
        print(' ', file=f)
        print('Accuracy of Averaged Perceptron on training data: ', train_accuracy_smart.pop(),file=f)
        print('Accuracy of Averaged Perceptron on testing data: ', test_accuracy_smart.pop(),file=f)
        print('Accuracy of Averaged Perceptron on dev data: ', dev_accuracy_smart.pop(),file=f)
        print(' ', file=f)
        print('Computation time for Naive Averaged Perceptron: ', time_naive, file=f)
        print('Computation time for Smart Averaged Perceptron: ', time_smart, file=f)
        print(' ', file=f)
        print(' ', file=f)
        
        