import regex as re
import pandas as pd
import nltk
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('wordnet')



class Preprocessor:
    def __init__(self):
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        pass

    def tokenizer(self, text):
        text = text.lower()  # convert to lowercase
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text) # remove special chars
        text = re.sub(r"\s+", " ", text) # remove multiple spaces
        tokens = text.strip(" ").split(" ") # split by space
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words] # remove stopwords and perform lemmatization
        
        return tokens
    
class TopicClassifier:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.docs = pd.read_csv("./wikipedia_raw_data.csv")
        self.labels = list(set(self.docs['topic']))

    def _clean_text(self):
        self.docs["summary"] = self.docs["summary"].apply(self.preprocessor.tokenizer)
        self.docs = self.docs[self.docs["summary"].str.len() >= 200].reset_index(drop=True)
    
    def fit(self, labels):
        pass

TC = TopicClassifier()
TC._clean_text()
print(TC.docs.shape)
