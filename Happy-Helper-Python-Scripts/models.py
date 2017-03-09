#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:04:10 2017

@author: michelle
"""

import pickle
import os
import numpy as np
# Set working directory.
os.chdir('/home/michelle/Documents/MVP/clean code')
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

###############
## FUNCTIONS ##
###############

## The from_pickle() function takes stored pickles and imports them for use. It takes two arguments.
## 'names_pickled_objects' is a list of strings for the names of the imported pickled objects.
## 'pickled_filenames' is a list of filenames we want to import.
## The function essentially iterates over the lists, imports each pickle, and appends
## them to a dictionary for later use in this script.
def from_pickle(names_pickled_objects, pickle_filenames):
    if names_pickled_objects and pickle_filenames:
        object_dict = {}
        for i in range(len(names_pickled_objects)):
            pickle_file = open(pickle_filenames[i], 'rb')
            object_dict[names_pickled_objects[i]] = pickle.load(pickle_file)
            pickle_file.close()
        return object_dict
    else:
        print "Invalid arguments to import pickled files."
        
## The score() function takes a list of models, and conducts cross-validation on them. It takes three arguments.
## 'list_of_models' is a list of models. 'number_kfolds' is the number of folds you want to conduct for
## cross validation. 'modeling_objects' is the dictionary of x_train, y_train, x_test, y_test
## The function loops over the list of models, generates cross validation scores, and appends the mean
## of these scores to a list named 'scores'.
def scores(list_of_models, number_kfolds, modeling_objects): 
    scores = []
    for model in list_of_models:
        cv_scores =  cross_val_score(model, modeling_objects['x_train'], modeling_objects['y_train'], cv=number_kfolds, scoring='neg_mean_squared_error')
        scores.append(cv_scores.mean())
    scores = np.asarray(scores)
    return scores

## The model_fits() function generates model fit and model accuracy data. It takes three arguments.
## 'list_of_models' is a list of models (e.g., MultinomialBayes()). 'model_names' is a list of strings
## that provide short hand names for the models. 'modeling_objects' are the x_train, y_train...etc data.
## The function creates two dictionaries for model fits and accuracies. It loops over the list of models,
## and appends fit and accuracy data to separate lists.
def model_fits(list_of_models, model_names, modeling_objects):
    model_fit_dict = dict()
    model_acc_dict = dict()
    for i in range(len(list_of_models)):
        list_of_models[i] = list_of_models[i].fit(modeling_objects['x_train'], modeling_objects['y_train'])
        pred = list_of_models[i].predict(modeling_objects['x_test'])
        acc = accuracy_score(modeling_objects['y_test'], pred)
        model_fit_dict[model_name[i]] = pred
        model_acc_dict[model_name[i]] = acc
    return model_fit_dict, model_acc_dict


## The class_report() function returns classification reports (e.g., precision,
## recall, f1-score). It takes three arguments. 'y_data' is the actual y_test values
## 'model_fit' is the model fit object you want to get a report on, and 'label_names'
## are the appropriate names for the y_test values.
def class_report(y_data, model_fit, label_names):
    from sklearn.metrics import classification_report
    report = classification_report(y_data, model_fit, target_names = label_names)
    return report

if __name__ == "__main__":
    
    # Import pickled training and test data. Set it equal to a dictionary named 'modeling_objects'
    pickled_objects = ['x_train', 'x_test', 'y_train', 'y_test']
    pickle_filenames = ['x_train.pck1','x_test.pck1', 'y_train.pck1', 'y_test.pck1']
    
    modeling_objects = from_pickle(pickled_objects, pickle_filenames)
    
    # Call the four ML models used for multiclass classification.
    # These models are: Stochastic Gradient Descent using hinge loss (which equates to SVM), 
    # Random Forest, Naive Bayes, and Linear Discriminant Analysis.
    # Note, hyper parameters were obtained via grid search in a separate script.
    clf_sgd = SGDClassifier(alpha = 1.0000000000000001e-05, l1_ratio = 0.59999999999999998, loss = 'hinge', penalty = 'elasticnet')
    clf_rf = RandomForestClassifier(n_estimators=14, max_features = 19, max_depth = 4)
    clf_nb = MultinomialNB()
    clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage = 0)
    
    # Place models in a list.
    list_of_models = [clf_sgd, clf_rf, clf_nb, clf_lda]
    
    # Get cross validations scores.
    scores = scores(models, 5, modeling_objects)
    print scores
    
    # Fit the models and get their accuracy. Return fit and accuracy data to
    # 'fit_dict' and 'acc_dict'
    model_name = ['clf_sgd', 'clf_rf', 'clf_nb', 'clf_lda']
    fit_dict, acc_dict = model_fits(models, model_name, modeling_objects)
    
    # Get the classification reports for each model (precision, recall, f1_score)
    for i in fit_dict.keys():
        print "Model Metrics for %s " % i + '\n'
        print class_report(modeling_objects['y_test'], fit_dict[i], ['anxiety', 'depression', 'other'])
    
    # Export model_fit_dict for script to plot data.
    f = open('model_fits.pck1', 'wb')
    pickle.dump(fit_dict, f)
    f.close()