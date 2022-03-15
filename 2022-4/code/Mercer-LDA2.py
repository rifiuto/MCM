#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hair_dryer_data = pd.read_excel('hair_dryer.xlsx')
microwave_data = pd.read_excel('microwave.xlsx')
pacifier_data = pd.read_excel('pacifier.xlsx')

import re


#
def clean_text(text):
    text = text.replace("<br />", " ")
    text = text.replace("<br", " ")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r"([.,!:?()])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace("-", " ")
    return text


hair_dryer_data['review_body'].apply(clean_text)

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

n_features = 1000
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df=0.5,
                                min_df=10)
tf = tf_vectorizer.fit_transform(microwave_data['review_headline'])

lda = LatentDirichletAllocation(n_components=15,
                                max_iter=150,
                                learning_method='online',
                                learning_offset=50, random_state=0)
lda.fit(tf)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


n_top_words = 7
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

#
import pyLDAvis
import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

first_level = ['great', 'pretty good', ]
