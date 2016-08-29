
from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required
import pandas as pd
import numpy as np
import cPickle as pickle
import re
import requests
import json
import nltk.data
import operator
import random
import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.summarization import summarize, keywords


from . import app, target_names, tfidf_vec_5 #estimator,


class PredictForm(Form):
    """URL to process"""
    URL = fields.TextField('URL:', validators=[Required()])

    submit = fields.SubmitField('Submit')

# Process the URL and return sentences

def get_candidates(sent_tokens, top_terms, n_sents=2, cutoff=0):
    sent_scores = {}
    for sent in sent_tokens:
        sent_scores[sent] = 0
    for key, value in sent_scores.items():
        sent_n = 0
        for term, weight in top_terms[cutoff:]:
            if term in key.lower():
                sent_scores[key] += weight
                sent_n += 1
        if sent_scores[key] > 0:
            sent_scores[key] = sent_scores[key]/(sent_n**.7)

    sorted_sentences = sorted(sent_scores.items(), key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in sorted_sentences[:n_sents]]

def compound_sents(sent_tokens, top_terms, cutoff=0):
    sia = SentimentIntensityAnalyzer()
    neg_scores = {}
    for sent in sent_tokens:
        neg_scores[sent] = sia.polarity_scores(sent)

    neg_sentences = []
    for key, value in neg_scores.items():
        if value['compound'] < 0:
            neg_sentences.append(key)
    candidates = get_candidates(neg_sentences, top_terms, cutoff=cutoff)
    return candidates

def neg_sents(sent_tokens, top_terms, cutoff=0):
    sia = SentimentIntensityAnalyzer()
    neg_scores = {}
    for sent in sent_tokens:
        neg_scores[sent] = sia.polarity_scores(sent)

    neg_sentences = []
    for key, value in neg_scores.items():
        if value['neg'] > 0:
            neg_sentences.append(key)
    candidates = get_candidates(neg_sentences, top_terms, cutoff=cutoff)
    return candidates




@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    prediction = 'prediction'
    results = []

    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data
        print(submitted_data)



        # Retrieve url value from form
        url = submitted_data['URL']
        r = []
        user_agent = {'User-agent': 'Mozilla/5.0'}
        id_ = re.search(r"\d+",url).group(0)
        api_request =  'https://api.airbnb.com/v2/reviews?client_id=3092nxybyb0otqw18e8nh5nty&locale=en-US&_limit=50&listing_id='+id_+'&role=all'

        # Get the first batch of 50 reviews
        first_r = requests.get(api_request, headers = user_agent)
        r.append(first_r)

        # Check if there are more reviews
        count = first_r.json()['metadata']['reviews_count'] #Get the count
        trips = count/50
        offset = 0

        # If there are more, get 'em!
        if count > 50:
            while trips > 0:
                offset += 50
                api_request = 'https://api.airbnb.com/v2/reviews?client_id=3092nxybyb0otqw18e8nh5nty&locale=en-US&_offset='+str(offset)+'&_limit=50&listing_id='+id_+'&role=all'
                trips -= 1
                next_r = requests.get(api_request, headers = user_agent)
                r.append(next_r)

        # Tokenize all sentences in all reviews for property
        reviews = ''
        for obj in r:
            for review in obj.json()['reviews']:
                reviews += review['comments']
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sent_tokens = tokenizer.tokenize(reviews)

        # Transform each review('comment') into a tf-idf vector
        if url[:6] == 'random':
            random.shuffle(sent_tokens)
            results = sent_tokens[:8]

        elif url[:6] == 'gensim':
            results = summarize(reviews, word_count=170)
            results = results.splitlines()

        elif url[:3] == 'neg':
            sia = SentimentIntensityAnalyzer()
            neg_scores = {}
            for sent in sent_tokens:
                neg_scores[sent] = sia.polarity_scores(sent)

            neg_sentences = []
            for key, value in neg_scores.items():
                if value['neg'] > 0:
                    neg_sentences.append(key)
            results = neg_sentences[:8]

        else:
            comment_arrays = []
            for obj in r:
                for review in obj.json()['reviews']:
                    #comment = tfidf.transform([review['comments']])
                    comment = tfidf_vec_5.transform([review['comments']])
                    comment_arrays.append(comment.toarray()[0])

            # Compute the average tf-idf vector across all reviews for the property
            ave_tfidf = np.true_divide(np.sum(comment_arrays, axis=0), len(comment_arrays))

            # Sort average vector and associated terms. Get top n terms
            top_terms = sorted(zip(tfidf_vec_5.get_feature_names(), ave_tfidf),key=lambda x: x[1], reverse=True)[:50]


            top = get_candidates(sent_tokens, top_terms)
            cutoffs = get_candidates(sent_tokens, top_terms,cutoff=15)
            negs = neg_sents(sent_tokens, top_terms, cutoff=4)
            negs_cutoff = neg_sents(sent_tokens, top_terms, cutoff=15)
            compound = compound_sents(sent_tokens, top_terms, cutoff=4)
            compound_cutoff = compound_sents(sent_tokens, top_terms, cutoff=15)
            for each in [top,cutoffs,negs_cutoff,compound_cutoff,]:
                for sent in each:
                    if sent not in results:
                        results.append(sent)
                #results.append('******')


    return render_template('index.html',
        results=results,
        form=form)

