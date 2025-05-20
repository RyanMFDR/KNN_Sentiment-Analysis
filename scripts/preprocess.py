import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')

# Stopwords & stemmer
stop_factory = StopWordRemoverFactory()
default_stopwords = stop_factory.get_stop_words()
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
    "dr": "dari",
    "bgn": "bangun",
    "dgn": "dengan",
    "utk": "untuk",
    "ga": "tidak",
    "gak": "tidak",
    "tdk": "tidak",
    "dpt": "dapat",
    "min": "admin",
    "daripada": "dibanding",
    "lo": "kamu",
    "gemoy": "presiden",
    "%": "persen",
    "bangke":"bangkai"
}

custom_stopwords = ['fufufafa']
stopwords = set(default_stopwords + custom_stopwords)

no_stem_words = {
    'bergizi', 'berhasil', 'mengurangi',
    'peningkatan', 'mengentaskan', 'kebijakan',
    'diragukan', 'tujuan', 'ditingkatkan', 'atasan','pelajar','keterlaksanaan'
}

abbreviation_dict = {
    'makan bergizi gratis': 'mbg',
}

# Fungsi normalisasi kata informal
def normalize_word(word):
    return normalization_dict.get(word, word)

# Hilangkan kata ulang seperti 'muntah-muntah' → 'muntah'
def remove_reduplication(word):
    match = re.match(r'^(\w+)-\1$', word)
    return match.group(1) if match else word

# Gabungkan frasa angka + unit menjadi satu token
def merge_number_phrases(text):
    #99,99 persen → 99,99_persen
    text = re.sub(r'(\d{1,3}(?:,\d{1,3})?)\s+(persen)', r'\1_\2', text)

    #angka+ (juta|miliar|ribu)? + (orang|anak|korban|penduduk|warga|siswa|guru|napi)
    text = re.sub(
        r'(\d+)\s+(juta|miliar|ribu)?\s*(orang|anak|korban|penduduk|warga|siswa|guru|napi)',
        lambda m: '_'.join(filter(None, m.groups())),
        text
    )

    #"rp 71 triliun" → "rp_71_triliun"
    text = re.sub(r'(rp)\s+(\d+)\s+(juta|miliar|triliun)', r'\1_\2_\3', text)

    #institusi pendidikan → "smpn 35 bandung" → "smpn_35_bandung"
    text = re.sub(r'\b(smpn|sma|smk|sdn|mtsn)\s+(\d+)\s+(\w+)', r'\1_\2_\3', text)

    #angka + kata benda umum (ex: "342 siswa") → "342_siswa"
    text = re.sub(r'(\d+)\s+(siswa|guru|napi|tahanan|pegawai|pelajar)', r'\1_\2', text)

    return text

def normalize_hashtags(text):
    known_words = ['makan', 'bergizi', 'gratis', 'dukung', 'mbg', 'pemerataan', 'gizi', 'anak', 'gratis', 'sehat']

    hashtags = re.findall(r'#\w+', text)
    for tag in hashtags:
        clean_tag = tag[1:]
        lowered = clean_tag.lower()
        expanded = lowered
        for word in known_words:
            expanded = re.sub(rf'(?<!\w)({word})(?!\w)', r'\1 ', expanded)
        expanded = re.sub(r'\s+', ' ', expanded).strip()
        text = text.replace(tag, expanded)
    return text   

def split_compound_words(text, known_words):
    pattern = re.compile(r'\b\w+\b')
    words = pattern.findall(text)
    result = []
    for word in words:
        if word in known_words:
            result.append(word)
        else:
            split_word = try_split_word(word, known_words)
            if split_word:
                result.extend(split_word)
            else:
                result.append(word)
    return ' '.join(result)

def try_split_word(word, known_words):
    splits = []
    i = 0
    while i < len(word):
        found = False
        for j in range(len(word), i, -1):
            if word[i:j] in known_words:
                splits.append(word[i:j])
                i = j
                found = True
                break
        if not found:
            return None
    return splits

#preprocessing
def preprocess(text):
    text = text.lower() 
    text = normalize_hashtags(text)
    text = merge_number_phrases(text)

    known_words_for_split = ['makan', 'bergizi', 'gratis', 'dukung', 'pemerataan', 'gizi', 'anak', 'sehat']
    text = split_compound_words(text, known_words_for_split)

    # Ganti frasa dengan singkatan  (misal: makan bergizi gratis → mbg)
    for phrase, abbr in abbreviation_dict.items():
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', abbr, text)

    # Hapus karakter non-huruf kecuali strip (untuk kata ulang)
    text = re.sub(r'[^\w\s\-\,\.\#\%]', ' ', text)
    
    # Tokenisasi
    tokens = word_tokenize(text)

    # Normalisasi & hilangkan kata ulang
    tokens = [normalize_word(word) for word in tokens]
    tokens = [remove_reduplication(word) for word in tokens]

    # Tangani kata ulang eksplisit (contoh: "makan-makan")
    final_tokens = []
    for token in tokens:
        if '-' in token:
            final_tokens.extend(token.split('-'))
        else:
            final_tokens.append(token)

    # Stopword removal
    filtered = []
    for word in final_tokens:
        if word.startswith('#'):
            filtered.append(word)
        elif word not in stopwords and (word.isalpha() or word.isdigit() or '_' in word):
            filtered.append(word)

    # Stemming
    stemmed = []
    for word in filtered:
        if word in no_stem_words or word.startswith('#') or '_' in word:
            stemmed.append(word)
        else:
            stemmed.append(stemmer.stem(word))

    return ' '.join(stemmed)

# Proses file CSV
df = pd.read_csv('./data/dataset.csv')
df['text'] = df['text'].apply(preprocess)
df.to_csv('./data/dataset_cleaned.csv', index=False)
