import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

df = pd.read_csv('./data/dataset.csv')

def preprocess(text):
    #Case folding
    text = text.lower()
    
    #non-alfabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    #Tokenisasi
    tokens = word_tokenize(text)
    
    #Stopword removal
    filtered = [word for word in tokens if word not in stopwords]
    
    #Stemming
    stemmed = [stemmer.stem(word) for word in filtered]
    
    return ' '.join(stemmed)

df['text'] = df['text'].apply(preprocess)
df.to_csv("./data/dataset_cleaned.csv", index=False)