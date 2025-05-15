import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')

# Stopwords & stemmer
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# Kamus normalisasi kata informal
normalization_dict = {
    "anak2": "anak-anak",
    "org2": "orang-orang",
    "temen2": "teman-teman",
    "tmn2": "teman-teman",
    "sdh": "sudah",
    "kek": "seperti",
    "yg": "yang",
    "rp": "rupiah",
    "dr": "dari",
    "bgn": "bagian",
    "dgn": "dengan",
    "utk": "untuk",
    "ga": "tidak",
    "gak": "tidak",
    "tdk": "tidak",
    "dpt": "dapat"
}

no_stem_words = {
    'bergizi', 'berhasil', 'mengurangi',
    'peningkatan', 'mengentaskan', 'kebijakan','diragukan'
}


# Normalisasi kata informal
def normalize_word(word):
    return normalization_dict.get(word, word)

# Hilangkan kata ulang seperti 'muntah-muntah' → 'muntah'
def remove_reduplication(word):
    match = re.match(r'^(\w+)-\1$', word)
    return match.group(1) if match else word

# Fungsi utama preprocessing
def preprocess(text):
    text = text.lower()

    # Hapus karakter non-huruf kecuali strip (untuk kata ulang)
    text = re.sub(r'[^\w\s\-\,\.\#]', ' ', text)
    
    #Tokenisasi
    tokens = word_tokenize(text)
    
    # Normalisasi dan reduplikasi
    tokens = [normalize_word(word) for word in tokens]
    tokens = [remove_reduplication(word) for word in tokens]

    #Kata ulang
    final_tokens = []
    for token in tokens:
        if '-' in token and not token.startswith('#'):
            final_tokens.extend(token.split('-'))
        else:
            final_tokens.append(token)

    # Stopword removal
    filtered = []
    for word in final_tokens:
        if word.startswith('#'):
            filtered.append(word)
        elif word not in stopwords and (word.isalpha() or word.isdigit()):
            filtered.append(word)
    
    # Stemming
    stemmed = []
    for word in filtered:
        if word in no_stem_words or word.startswith('#'):
            stemmed.append(word)
        else:
            stemmed.append(stemmer.stem(word))

    return ' '.join(stemmed)

# Proses CSV
df = pd.read_csv('./data/dataset.csv')
df['text'] = df['text'].apply(preprocess)
df.to_csv('./data/dataset_cleaned.csv', index=False)
