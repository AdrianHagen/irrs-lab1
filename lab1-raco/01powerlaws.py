# Utils script for the first exercise
import nltk
import pprint
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt

import urllib.request

target_url = (
    "https://fegalaz.usc.es/~gamallo/aulas/lingcomputacional/corpus/quijote-en.txt"
)

quijote_text = urllib.request.urlopen(target_url)

# tokenized with no pre process
tokenized_text = word_tokenize(quijote_text)

# remove stopwords
english_sw = set(stopwords.words("english") + list(string.punctuation))

filtered_tokenized_text = [
    w.lower() for w in tokenized_text if w.lower() not in english_sw
]

pprint.pprint(filtered_tokenized_text)
