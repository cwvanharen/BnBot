# BnBot
A simple Airbnb review summarizer using multiple backends.

The default backend is accessed by dropping a url into the box.

Other backends available are accessed by prefixing the url with:
- random: this gives you a random selection of sentences from the reviews as a point of comparison
- gensim: this will give you a summary using the gensim summarize package
- neg: this will return the most negative sentences according to Vader Sentiment Analysis.

The code for the app is in the 'views.py' file.

The code that generated the model used in the default backend is in the ipython notebook.
