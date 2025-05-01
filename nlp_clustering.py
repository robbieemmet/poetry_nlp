# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:00:19 2025

@author: Robert Emmet
"""
import os
import re
import pandas as pd
import numpy as np
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer

all_poems = PlaintextCorpusReader("C:/Users/rober/OneDrive/Desktop/camp_until/poetry_nlp/data", '.*\\.txt', encoding="cp1252")

poems_list = []
for fileid in all_poems.fileids():
    filename = fileid
    poems_list.append((filename, all_poems.raw(fileid)))

poems_df = pd.DataFrame(poems_list, columns=['filename', 'text'])

documents = []
for i in range(poems_df.shape[0]):
    item = poems_df["text"][i]
    documents.append(item)

stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    stems_comb = ' '.join(stems)
    return stems_comb

documents_stemmed = []
for i in range(poems_df.shape[0]):
    item=documents[i]
    item_stemmed = tokenize_and_stem(item)
    documents_stemmed.append(item_stemmed)

tfidf_vectorizer = TfidfVectorizer( stop_words='english',
                                 use_idf=True, ngram_range=(1,1))

tfidf_matrix = tfidf_vectorizer.fit_transform(documents_stemmed)

# TODO figure out how to make this a loop instead, possibly?
kmeans3 = KMeans(n_clusters=3, random_state=0)
clusters3 = kmeans3.fit_predict(tfidf_matrix)
poems_df["clusters3"] = clusters3
kmeans5 = KMeans(n_clusters=5, random_state=0)
clusters5 = kmeans5.fit_predict(tfidf_matrix)
poems_df["clusters5"] = clusters5
kmeans8 = KMeans(n_clusters=8, random_state=0)
clusters8 = kmeans8.fit_predict(tfidf_matrix)
poems_df["clusters8"] = clusters8
poems_df.iloc[0:60,[0,3,4]]
poems_df.iloc[61:120,[0,3,4]]

# Examine poems by cluster - 5 clusters vs. 8 clusters?

poems_df.loc[poems_df["clusters5"]==0, "filename"]
poems_df.loc[poems_df["clusters5"]==1, "filename"]
poems_df.loc[poems_df["clusters5"]==2, "filename"]
poems_df.loc[poems_df["clusters5"]==3, "filename"]
poems_df.loc[poems_df["clusters5"]==4, "filename"]

poems_df.loc[poems_df["clusters8"]==0, "filename"]
poems_df.loc[poems_df["clusters8"]==1, "filename"]
poems_df.loc[poems_df["clusters8"]==2, "filename"]
poems_df.loc[poems_df["clusters8"]==3, "filename"]
poems_df.loc[poems_df["clusters8"]==4, "filename"]
poems_df.loc[poems_df["clusters8"]==5, "filename"]
poems_df.loc[poems_df["clusters8"]==6, "filename"]
poems_df.loc[poems_df["clusters8"]==7, "filename"]

# Try some part of speech tagging, other context things
