# import nltk
# nltk.download('movie_reviews')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# If running for the first time, uncomment the above lines and run the script

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Extracting the fileids from the movie_reviews corpus
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')
# Extracting the raw text from the fileids
positive_reviews = [movie_reviews.raw(fileids=[f]) for f in positive_fileids]
negative_reviews = [movie_reviews.raw(fileids=[f]) for f in negative_fileids]

# Tokenize and preprocess the text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(t)
              for t in tokens if t not in stop_words and t.isalpha()]
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens


positive_tagged = [preprocess(r) for r in positive_reviews]
negative_tagged = [preprocess(r) for r in negative_reviews]
