#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:25:16 2017

@author: michelle
"""
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
# Set working directory.
os.chdir('/home/michelle/Documents/MVP/clean code')

from sklearn.metrics import confusion_matrix

## Here, I reused a function to import multiple pickled files. This function was imported from 
## my modeling file, models.py.
from models import from_pickle

###############
## FUNCTIONS ##
###############


## The conf_matrix() function takes true y_values and our predicted_y_values (e.g., predicted categories)
## and creates a normalized confusion matrix (consisting of category proportions) and a raw confusion matrix
## (consisting of raw counts). It takes two argumements: 'true_y_values' are the true y_values / labels
## and 'predicted_y_values' are y_values produced by a classification model (e.g., random forest, SVM). 
## It returns both the normalized confusion matrix, and the raw confusion matrix.
def conf_matrix(true_y_values, predicted_y_values):
    conf_matrix = confusion_matrix(true_y_values, predicted_y_values)
    np.set_printoptions(precision=2)
    normalized_conf_matrix = conf_matrix / conf_matrix.astype(np.float).sum(axis=1)
    normalized_conf_matrix = normalized_conf_matrix.round(2)
    return normalized_conf_matrix, conf_matrix

## The plot_matrix() function takes the confusion matrices, and produces uniform seaborn plots. It takes
## 5 arguments: 'normalized_conf_matrix' is the normalized confusion matrix, 'raw_conf_matrix' is a raw confusion
## matrix with counts instead of proportions, 'ticklabels' is a list of strings that consist of labels for the plot
## 'plot_title' is the title of the plot, and 'output_filename' is the name of the file you want to output (e.g., figure.png)
def plot_matrix(normalized_conf_matrix, raw_conf_matrix, ticklabels, plot_title, output_filename):
    sns.set_style("whitegrid")
    sns.set(font_scale=2)
    fig = sns.heatmap(raw_conf_matrix, 
                      cmap = 'PuBu', 
                      annot = normalized_conf_matrix,
                      xticklabels = ticklabels,
                      yticklabels = ticklabels,
                      linewidths = .1,
                      annot_kws={"size":24})
    fig.set_title(plot_title)
    fig.set(xlabel='True Label', ylabel='Predicted Label')
    fig.figure.savefig(output_filename, dpi = 400)

if __name__ == "__main__":
    
    # Open the pickled model fits dictionary and y_test data (the TRUE y_values from our data)
    # Use the from_pickle function imported from models.py to load the pickles.
    pickled_objects = ['fit_dict', 'y_test']
    pickle_filenames = ['model_fits.pck1', 'y_test.pck1']
    plotting_objects = from_pickle(pickled_objects, pickle_filenames)       
    
    # Get normalized and raw count confusion matrices for each model.
    normalized_matrix_lda, raw_matrix_lda = conf_matrix(plotting_objects['y_test'], plotting_objects['fit_dict']['clf_lda'])
    normalized_matrix_naive_bayes, raw_matrix_nb = conf_matrix(plotting_objects['y_test'], plotting_objects['fit_dict']['clf_nb'])
    normalized_matrix_random_forest, raw_matrix_rf = conf_matrix(plotting_objects['y_test'], plotting_objects['fit_dict']['clf_rf'])
    normalized_matrix_sgd, raw_matrix_sgd = conf_matrix(plotting_objects['y_test'], plotting_objects['fit_dict']['clf_sgd'])
    
    # Use seaborn to plot all my confusion matrices. The models are plotting 
    labels = ['Anxiety', 'Depression', 'Other']
    plot_matrix(normalized_matrix_lda, raw_matrix_lda, labels, 'LDA Confusion Matrix', 'lda_confmat.png')
    plot_matrix(normalized_matrix_naive_bayes, raw_matrix_nb, labels, 'Naive Bayes Confusion Matrix', 'nb_confmat.png')
    plot_matrix(normalized_matrix_random_forest, raw_matrix_rf, labels, 'Random Forest Confusion Matrix', 'rf_confmat.png')
    plot_matrix(normalized_matrix_sgd, raw_matrix_sgd, labels, 'SVM-SGD Confusion Matrix', 'sgd_confmat.png')
