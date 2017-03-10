#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:52:49 2017

@author: michelle
"""

import os
# Set working directory.
os.chdir('/home/michelle/Documents/MVP/clean code')

import string
import nltk
import numpy as np
import praw
import pandas as pd

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

## The following two modules are imported from a previous script, text_processing.py
from text_processing import custom_stop
from text_processing import decode

###############
## FUNCTIONS ##
###############

## The reddit_api() function simply calls the reddit api (PRAW). It takes several arguments:
## 'client_id', 'secret_id' are inputs you are given when you register to use the api on reddit. 
## 'user_agent' is a descriptive title for the agent you are crawling reddit with.
## 'reddit_username' and 'reddit_password' are your username and password on reddit.
## For more information, read the PRAW api documentation: http://praw.readthedocs.io/en/latest/
def reddit_api(client_id, secret_id, user_agent, reddit_username, reddit_password):
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=secret_id,
                         user_agent=user_agent,
                         username=reddit_username,
                         password=reddit_password)
    return reddit

## The reddit_scraper() function is a custom function to pull reddit selftext, the reddit url for each post
## the post titles, and the number of comments. The function takes several arguments: 'subreddit' is the subreddit
## you want to scrape (e.g.,'news', 'cute', 'funny' etc), 'scraping_limit' is the max number of posts you want the api
## to return. The function does a for loop, looking for each submission in the subreddit you defined,
## and appending titles, urls, selftext, and number of comments to those respective lists. Those lists
## are then returned for later manipulation.
def reddit_scraper(subreddit, scraping_limit):
    self_text = []
    url = []
    post_title = []
    number_of_comments = []
    for submission in reddit.subreddit(subreddit).hot(limit=scraping_limit):
        if submission.selftext != '':
            post_title.append(submission.title)
            url.append(submission.url)
            self_text.append(submission.selftext)
            number_of_comments.append(submission.num_comments)
    return post_title, url, self_text, number_of_comments

## The df_processing() function tokenizes, and stems words from a dataframe of reddit data. It takes
## several arguments: 'df' is a dataframe with reddit data, 'column_to_tokenize' is the column that
## needs to be tokenized, 'column_of_tokens', is the name of a column where you want to store tokens,
## 'column_of_stems' is the name of the column where you want to store stemmed words. The function
## returns the cleaned dataframe.    
def df_processing(df, column_to_tokenize, column_of_tokens, column_of_stems):
    df[column_of_tokens] = df.apply(lambda row: nltk.word_tokenize(row[column_to_tokenize]), axis=1)
    df[column_of_tokens] = df[column_of_tokens].apply(lambda x: [item for item in x if item not in stop_words])
    df[column_of_stems] = df.apply(lambda row: nltk.word_tokenize(row[column_to_tokenize]), axis=1)
    return df

## The user_input_text function takes a string, strips the punctuation, tokenizes it, and returns it as a string.
## It takes one argument, 'user_input' is a text string of any kind. This function is used to mimic
## the user input in my Insight Project webapp (Happy Helper), where a user can input any type of text they want.
def user_input_text(user_input):
    text = user_input.translate(None, string.punctuation)
    text = nltk.word_tokenize(user_input)
    text = str(user_input)
    return text

## The cosine_sim() function takes two texts, and returns their cosine "angle" between them. This
## is otherwise known as a "cosine similarity" in natural language processing. The values range
## between 0 and 1, 0 are completely dissimlar texts, 1 represents and exact match. This function
## takes two arguments: 'text1' is  the first body of text to compare, and 'text2' is the second body
## of text to compare. Note that this function is used in a loop in the similarity_comparison() function.
def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 75)
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


## The similarity_comparison funtion compares user input to each reddit post and returns a cosine
## similarity. It takes three arguments 'user_input' is a string representing the user input
## 'df_of_posts' is the dataframe of reddit posts, 'column_to_compare' is the df column of reddit
## selftext you want to compare to user input. The function loops through individual posts in
## the df_of_posts, and produces a cosine similarity between user input and the reddit
## post self_text. The cosines are multiplied by 100 to create a "percent similarity" metric, eventually
## displayed to the user. These similarities are appended to a list.
def similarity_comparison(user_input, df_of_posts, column_to_compare):
    similarity = []
    for i in range(len(df_of_posts)):
        string_1 = str(user_input)
        string_2=  str(df_of_posts[column_to_compare][i])
        cosines = cosine_sim(string_1, string_2)
        cosines = cosines * 100
        similarity.append(cosines)
    return similarity

## The sort_posts() function takes a df and a column name, and sorts the df by that column.
## It specifically sorts in descending order (because we want the most similar posts up top) when
## eventually displayed on the webapp. The function takes two arguments: 'df_to_sort' is a
## df to sort, 'column_to_sortby' is a column you want to sort by. 
def sort_posts(df_to_sort, column_to_sortby):
    df = df_to_sort.reset_index(drop = True)
    df[column_to_sortby] = df[column_to_sortby].round(decimals = 1)
    df = df_to_sort.sort([column_to_sortby], ascending = False)
    return df

if __name__ == "__main__":
    
    ## Call the reddit api with the appropriate login information
    reddit = reddit_api('FaA1JN_8OF_5gA',
                         '6IrDUlhO7_XQrh6kMuNgitS_UxI',
                         'Mental Health Script by /u/tatinthehat',
                         'username',
                         'password')
    
    # Set the custom stop words, use function imported from text_processing.py
    stop_words = custom_stop()
    
    # Scrape reddit for posts (e.g., anxiety)
    self_text, url, post_title, number_of_comments = reddit_scraper('anxiety', 100)
    
    # Create dataframe based on post self_text, url, title, and number_of_comments
    df_of_posts = pd.DataFrame({'title': post_title, 'url': url, 'selftext': self_text, 'number': number_of_comments})
    
    # Decode the reddit post selftext from UTF, using function from text_processing.py
    df_of_posts = decode(df_of_posts, 'selftext')
    
    # Tokenize and stem text in the posts dataframe.
    df_of_posts = df_processing(df_of_posts, 'selftext', 'tokenized_selftext', 'stemmed_selftext')
    
    # Reset the index of the df_of_posts dataframe
    df_of_posts = df_of_posts.reset_index(drop = True)
    
    # Provide an example of user input text.
    example_user_input = 'Here\'s an example of user input that the app would take in. The app strips out all punctuation, tokenizes it, and evaluates it for length.'
    example_user_input = user_input_text(example_user_input)
    
    # Generate cosine similarities from user input
    similarities = similarity_comparison(example_user_input, df_of_posts, 'tokenized_selftext')
    
    # Create a similarity column in the df_of_posts dataframe
    df_of_posts['similarity'] = similarities
         
    # Sort the posts
    df_of_posts = sort_posts(df_of_posts, 'similarity')
    
    # Create an empty list, append all information to a new dictionary.
    # This will be used in jinja code within the Flask webapp framework.
    posts_list = []
    for i in range(len(df_of_posts)):
        posts_list.append(dict(title = df_of_posts.iloc[i]['title'],
                               url = df_of_posts.iloc[i]['url'],
                               number = df_of_posts.iloc[i]['number'],
                               similarities = df_of_posts.iloc[i]['similarity']))