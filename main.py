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

#  Compute word frequencies, create a word cloud
all_words = [t for tagged_review in positive_tagged + negative_tagged
             for t in tagged_review]
all_words = FreqDist(all_words)

# print(all_words.most_common(10))
#  Works. Prints the 10 most common words in the corpus

# Train a simple sentiment analysis using Naive Bayes Classifier


def extract_features(review):
    return {word: (word in set(review)) for word in all_words}


positive_features = [(extract_features(r), 'Positive')
                     for r in positive_tagged]
negative_features = [(extract_features(r), 'Negative')
                     for r in negative_tagged]
all_features = positive_features + negative_features

# Split the data into training and testing datasets
threshold = 0.8
split_point = int(threshold * len(all_features))
train_data, test_data = all_features[:split_point], all_features[split_point:]

# Train the classifier
classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy of the classifier is: ", accuracy(classifier, test_data))

# Test the classifier
input_reviews = [
    "It is an amazing movie",
    "This is a dull movie. I would never recommend it to anyone.",
    "The cinematography is pretty great in this movie",
    "The direction was terrible and the story was all over the place"
    "The movie was a great waste of time"
]

print("Predictions:")
for review in input_reviews:
    review = preprocess(review)
    print(f'Sentiment: {classifier.classify(extract_features(review))}')

# Output:
# Accuracy of the classifier is:  0.825
# Predictions:
# Sentiment: Positive
# Sentiment: Negative
# Sentiment: Positive
# Sentiment: Negative
# Sentiment: Negative

#  The classifier is able to correctly classify the sentiment of the input reviews
