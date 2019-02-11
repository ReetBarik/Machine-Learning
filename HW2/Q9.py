# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:52:30 2018

@author: Reet Barik
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Pre-process the data (Note: Vectorizer uses the provided vocabulary for test data only)
def preprocess(train_data_path, train_label_path, stopwords_path, extern_vocab=None):
    
    with open(train_label_path, 'r') as file:
        train_labels_raw = file.read().splitlines()
        
    train_labels = list(map(int, train_labels_raw))
    
    with open(stopwords_path, 'r') as file:
        stoplist =  file.read().splitlines()
        
    stop_words = set()
    for line in stoplist:
        word = line
        stop_words.add(word)
        
    with open(train_data_path, 'r') as file:
        train_data_raw = file.read().splitlines()
        
    train_data = []
        
    # Removing stop-words
    for message in train_data_raw:
        message_words = message.split()
        resultwords  = [word for word in message_words if word.lower() not in stop_words]
        message = ''
        message = ' '.join(resultwords)
        train_data.append(message)
    
    if extern_vocab is None: # For training data
        vectorizer = CountVectorizer()
    else: # For testing data
        vectorizer = CountVectorizer(vocabulary = extern_vocab)
    
    # Feature extraction 
    train_bag_of_words = vectorizer.fit_transform(train_data).toarray().tolist()
    
    del train_data
    
    if extern_vocab is None:
        vocab = vectorizer.get_feature_names()
        return train_data_raw, vocab, train_bag_of_words, train_labels
    else:
        vocab = extern_vocab
        return train_data_raw, train_bag_of_words, train_labels


class NaiveBayesModel:
    model_features = []
    model_labels = []
    vocabulary = []
    
    def __init__(self, vocab):
        self.vocabulary = vocab
    
    # Get probability that the class label is 'is_label'
    def get_label_prob(self, feature, is_label):
        num_of_label = 0
        num_total_labels = len(self.model_labels)
        for past_label in self.model_labels:
            if past_label == is_label: 
                num_of_label += 1

        p_label = (num_of_label + 1) / (num_total_labels + 2)

        count_w_and_c = [0 for x in range(len(self.vocabulary))]

        for i in range(len(self.model_features)):
            f = self.model_features[i]
            l = self.model_labels[i]
            if l == is_label:
                # Counting number of times feature[j] has been in any f (for all f in model_features) 
                # such that l == is_label
                for j in range(len(f)):
                    if(f[j] == feature[j]):
                        count_w_and_c[j] += 1

        p_product = 1
        i = 0
        for c in count_w_and_c:
            prob_w_c = (c + 1) / (num_of_label + 2)
            p_product *= prob_w_c

        return p_label * p_product
    
    # Predict label after computing probability of both labels
    def predict(self, feature, label, train = True):

        prob_of_0 = self.get_label_prob(feature, 0)
        prob_of_1 = self.get_label_prob(feature, 1)

        predicted_label = 0
        if (prob_of_1 > prob_of_0): 
            predicted_label = 1

        if (train): # Add training features only to the model
            self.model_features.append(feature)
            self.model_labels.append(label)

        return predicted_label
    

# Train on the training data
def train(train_data, vocabulary, train_bag_of_words, train_labels):

    model = NaiveBayesModel(vocabulary)

    for i in range(len(train_data)):
        feature = train_bag_of_words[i]
        label = train_labels[i]
        
        model.predict(feature, label)

    return model

# Test on testing data using trained model
def test(model, test_bag_of_words, test_labels):
    
    num_mistakes = 0
    num_features = len(test_bag_of_words)

    for i in range(num_features):
        feature = test_bag_of_words[i]
        label = test_labels[i]
        pred_label = model.predict(feature, label, False)
        if (pred_label != label):
            num_mistakes += 1

    test_accuracy = (1 - (num_mistakes / num_features)) 
    
    return test_accuracy

# Scikit-learn Logistic Regression to compute training and testing accuracy
def scikit_logistic_regression(logisticRegr, train_bag_of_words, train_labels, test_bag_of_words, test_labels):
    
    logisticRegr.fit(train_bag_of_words, train_labels)
    
    logistic_train_accuracy = logisticRegr.score(train_bag_of_words, train_labels)
    logistic_test_accuracy = logisticRegr.score(test_bag_of_words, test_labels)
    
    return logistic_train_accuracy, logistic_test_accuracy 
    
    
if __name__ == "__main__":
    
    train_data, vocabulary, train_bag_of_words, train_labels = preprocess("traindata.txt", "trainlabels.txt", "stoplist.txt")
    model = train(train_data, vocabulary, train_bag_of_words, train_labels) 
    train_accuracy = test(model, train_bag_of_words, train_labels)
    
    test_data, test_bag_of_words, test_labels = preprocess("testdata.txt", "testlabels.txt", "stoplist.txt", vocabulary)
    test_accuracy = test(model, test_bag_of_words, test_labels)   
    
    logisticRegr = LogisticRegression()
    
    logistic_train_accuracy, logistic_test_accuracy = scikit_logistic_regression(logisticRegr, train_bag_of_words, train_labels, test_bag_of_words, test_labels)

    print("\nQ.9. Naive Bayes' Classifier:\n")
    print("Training accuracy: " + str(train_accuracy) + "\n")
    print("Testing accuracy: " + str(test_accuracy) + "\n")
    
    print("\nQ.9. Scikit-learn Logistic Regression Classifier:\n")
    print("Training accuracy: " + str(logistic_train_accuracy) + "\n")
    print("Testing accuracy: " + str(logistic_test_accuracy) + "\n")