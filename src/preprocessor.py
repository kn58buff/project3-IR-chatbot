from nltk.stem import WordNetLemmatizer
import regex as re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessor:
    """Class to implement preprocessing with tokenization, stop word removal, and lemmatization."""
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenizer(self, text):
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text) # Remove special characters
        text = re.sub(r"/^\s+|\s+$|\s+(?=\s)", "", text).strip() # Remove extra spaces
        text = text.lower()
        tokens = text.split()
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words] # Remove stop words and lemmatize
        return lemmatized_tokens