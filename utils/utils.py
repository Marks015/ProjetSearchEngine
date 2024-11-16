import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def expand_french_contractions(text):
    """Expand common French contractions in the text."""
    contractions = {
        "c'est": "ce est", "j'ai": "je ai", "l'est": "le est", "qu'il": "que il",
        "qu'elle": "que elle", "qu'ils": "que ils", "qu'elles": "que elles",
        "j'suis": "je suis", "t'es": "tu es", "y'a": "il y a", "c'te": "cette"
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    return text

def preprocess_text(text, use_stemming=True, remove_stopwords=True, expand_contractions_option=True):
    """Preprocess the text for French: normalize, expand contractions, remove stop words, and apply stemming."""
    # 1. Lowercase the text
    text = text.lower()

    # 2. Expand contractions (if enabled)
    if expand_contractions_option:
        text = expand_french_contractions(text)

    # 3. Remove special characters and numbers (keeping only letters and spaces)
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', text)

    # 4. Tokenize the text into words
    words = word_tokenize(text)

    # 5. Remove stop words (if enabled)
    if remove_stopwords:
        stop_words = set(stopwords.words('french'))
        words = [word for word in words if word not in stop_words]

    # 6. Apply stemming (French-compatible) if enabled
    if use_stemming:
        stemmer = SnowballStemmer("french")
        words = [stemmer.stem(word) for word in words]

    # 7. Remove extra whitespace and join words into a single string
    return ' '.join(words).strip()


