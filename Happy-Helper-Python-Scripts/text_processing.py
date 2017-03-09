#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:23:06 2017

@author: michelle
"""
import pandas as pd
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
 # Set working directory.
os.chdir('/home/michelle/Documents/MVP/clean code')
import pickle

from random import randint
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

###############
## FUNCTIONS ##
###############

## The import_csv function imports csvs from a file directory. It takes two arguments.
## "names_for_objects" is a list of strings you want to name your objects.
## "files_to_import" are a list of csvs you wan to import.
## This function loops the elements in each list, and appends the resulting dataframe
## objects to a list.
def import_csv(names_for_objects, files_to_import):
    dat_list = []
    names_for_objects = names_for_objects
    for i in range(len(files_to_import)):
        names_for_objects[i] = pd.read_csv(files_to_import[i])
        dat_list.append(names_for_objects[i])
    return dat_list

## The decode() function decodes text from UTF-8 formatting. It takes two arguments.
## 'df' is a dataframe with text. 'column_to_decode' is a column in that dataframe
## where the specific text resides. The function converts all text in that column into strings,
## sets them to all lower case, and decodes them from UTF-8.        
def decode(df, column_to_decode):
    df[column_to_decode] = df[column_to_decode].astype(str)
    df[column_to_decode] = df[column_to_decode].str.lower()
    df[column_to_decode] = df[column_to_decode].str.decode('utf-8', errors='strict')
    return df

## The split() function splits data into a training and test set. It takes three arguments.
## 'data_to_split' is data we want to split into training and test sets (e.g., our dataframe)
## 'size_of_split' denotes the proportion of data you want as the test set (e.g., .2 or 20%)
## 'y_categories' is the dataframe column that consist of the labels used for the actual y_labels.
## The function splits the data, then creates a y_train and y_test variable used for later
## predictive modeling.
def split(data_to_split, size_of_split, y_categories):
    train, test = train_test_split(data_to_split, test_size = size_of_split, random_state = randint(0,500))
    y_train = train[y_categories].values
    y_test = test[y_categories].values
    return train, test, y_train, y_test

## The custom_stop() function denotes a set of custom stop words to remove from our data.
## It takes the stop_words from the nltk package (minus negative words like "no" or "not)
## And combines those words with random punctuation. This can then be used to remove all irrelevant
## words and characters from the data.
def custom_stop():
    custom_stop = stopwords.words('english')
    del custom_stop[109:112]
    custom_stop = set(custom_stop)
    etc_stop = set(('\'ve', '[', ']', '\[\]', '\'s', '\'m', 'n\'t', '``', '\\n', '.', '\.', '...', '-', '\'\'', '(', ')', 'm', 's', 've', ',', ':', '*', '@', '!', '$', '%', '&', '?', '\'', '\"', '\"m', '\"n\'t\"', ' ','removed', 'deleted', '[]','0', 'te'))
    stop_words = custom_stop.union(etc_stop)
    return stop_words

## The stem() function takes a document (e.g, a corpus of strings), tokenizes, removes stop words, and
## stems words before returning the cleaned text. It takes one argument: 'document_to_stem' is
## a corpus of strings (e.g., a list of strings)
def stems(document_to_stem):
    stemmer = PorterStemmer()
    document_to_stem = document_to_stem.lower()
    doc_tokens = filter(lambda x : x.isalpha(), nltk.word_tokenize(document_to_stem))
    doc_stopped_tokens = filter(lambda x : x not in stop_words, doc_tokens) # remove stop tokens
    doc_stemmed_tokens =  map(lambda x : stemmer.stem(x), doc_stopped_tokens)
    cleaned_text = " ".join(doc_stemmed_tokens)
    return cleaned_text

## The corpus_append() function takes a data_frame, and creates a corpus with it. It takes
## two arguments. 'df' is a dataframe, 'column_to_append' is the dataframe target column with text.
## In the function, an empty list is created. Then it loops over elements in the target column
## appending the text to the corpus, **while simultaneously** tokenizing and stemming words using
## the stem() function above.
def corpus_append(df,column_to_append):
    corpus = []
    for selftext in df[column_to_append]:
        corpus.append(stems(selftext))
    return corpus


## The tfidf() function takes the cleaned corpus and vectorizes it for later modeling. It takes
## two arguments: 'corpus', which is the corpus we created, and 'number_of_features', the number of features
## we wish to extract from our text. This step is the starting point to what is commonly referred to as
## a "bag of words" model.
def tfidf(corpus, number_of_features):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='word',
                    min_df = 0, 
                    stop_words = None, 
                    max_features = number_of_features)
    data = vectorizer.fit_transform(corpus)
    data = data.toarray(corpus)
    return data

## The to_pickle() function pickles variables / objects for later use. It takes two arguments.
## 'objects_to_pickle' is a list of variables we want to pickle, and 'pickle_filenames' are
## what we want to name the eventual pickled files. In the function, if these two items exist,
## we iterate over the objects_to_pickle, set their filenames, and dumps them to the working directory.
## The loop will quit if no apporpriate item is listed in the function.
def to_pickle(objects_to_pickle, pickle_filenames):
    if objects_to_pickle and pickle_filenames:
        for i in range(len(objects_to_pickle)):
            pickled_file = open(pickle_filenames[i], 'wb')
            pickle.dump(objects_to_pickle[i], pickled_file)
            pickled_file.close()
    else:
        print "Invalid arguments for pickling."
  

if __name__ == "__main__":
   
    # Set dataframe and filenames
    object_names = ['anxiety', 'depression', 'other']
    csv_file_names = ['reddit_anxiety_2.csv', 'reddit_depress_2.csv', 'other.csv']
    
    #Import the data.
    imported_data_files = import_csv(object_names, csv_file_names)
    
    # Assign category labels to the data
    for i in range(len(imported_data_files)):
        imported_data_files[i]['category'] = i
       
    # Concatenate dataframes together.
    frames = [imported_data_files[0], imported_data_files[1], imported_data_files[2]]
    data = pd.concat(frames)
    
    # Decode the data frome UTF-8.
    data = decode(data, 'selftext')

    # Determine the test train splits for the data.
    train, test, y_train, y_test = split(data, .2, 'category')
   
    # Generate stop words.
    stop_words = custom_stop()
   
    # Remove stop words and stem all words. Append them to a newly created corpus.
    train2 = corpus_append(train, 'selftext')
    test2 = corpus_append(test, 'selftext')
   
    # Apply TFIDF to the corpus.
    x_train = tfidf(train2, 75)
    x_test = tfidf(test2, 75)
   
    # Pickle files to model in a separate script
    pickle_filenames = ['x_train.pck1', 'x_test.pck1', 'y_train.pck1', 'y_test.pck1']
    objects_to_pickle = [x_train, x_test, y_train, y_test]
    
    to_pickle(objects_to_pickle, pickle_filenames)
    