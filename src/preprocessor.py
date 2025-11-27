from nltk.stem import WordNetLemmatizer
import regex as re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenizer(self, text):
        # Assuming text is already mostly preprocessed
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        text = re.sub(r"/^\s+|\s+$|\s+(?=\s)", "", text).strip()
        text = text.lower()
        tokens = text.split()
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return lemmatized_tokens