
import nltk
import pickle
import json
from Chat_Bot import chat_bot
from keras.models import load_model
from nltk.stem.lancaster import LancasterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
        nltk.download('punkt')
stemmer = LancasterStemmer()
with open("./src/data.pickle", "rb") as f:
    words, classes, _, _ = pickle.load(f)

with open("./src/intents.json") as file:
    intents = json.load(file)
model = load_model('./src/saved_model.model')

ch_bot = chat_bot.Bot()
ch_bot.words = words
ch_bot.classes = classes
ch_bot.intents = intents
ch_bot.model = model
ch_bot.stemmer = stemmer
ch_bot.chat()