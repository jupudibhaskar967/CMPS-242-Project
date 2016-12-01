from nltk.corpus import stopwords
import string

""" Constants """
REVIEW_DATA_FILE = 'yelp_academic_dataset_review.json'
# Fields in review data file
TEXT = 'text'
PUNCTUATIONS = set(string.punctuation)
STOP_WORDS = set(stopwords.words("english"))
