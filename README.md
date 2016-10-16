# BnBot
A poorly named Airbnb review summarizer.

## Why
Airbnb currently does not offer summaries of the reviews for properties listed on it's site. Unlike almost all other sites that rely on user generated reviews to add significant value to their product (e.g. Yelp, Amazon, Google Restaurants), as an Airbnb user, I have to read or skim all the reviews to try and pick out important information. BnBot adds this summary functionality. By surfacing the most important sentences in the review corpus along with the most important _negative_ sentences, you can quickly get a clear picture of the property you're looking at. This streamlines the search experience for users and helps them avoid issues that would negatively impact their stay. And THAT leads to happy Airbnb-ers!

## How
The default backend is accessed by dropping a url into the url box. The current implementation uses Tf-idf and Vader sentiment analysis to vectorize and rank tokenized sentences. It returns those with the highest rank from a few different stacks, ideally providing a diverse set of informative summary sentences. 

Other backends (these might be broken at any given time!) available are accessed by prefixing the url with:
- 'random': this gives you a random selection of sentences from the reviews as a point of comparison
- 'gensim': this will give you a summary using the gensim summarize package
- 'neg': this will return only the most negative sentences according to Vader Sentiment Analysis with no additional analysis.

I'm currently working on another version that will cluster sentences using doc2vec distances as a sentence similarity metric and then use the same ranking algorithm as the current implementation. Hopefully this will ensure that important topics get surfaced more consistently.

A previous version used latent semantic analysis to extract topicality, but it proved to be pretty poor fit for this problem, probably due to the overwhelming similarity of most of the review content.


