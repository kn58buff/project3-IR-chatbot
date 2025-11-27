import pandas as pd
import numpy as np
import pickle
import ast

class TopicClassifier:
    def __init__(self):
        self.docs = pd.read_csv("./wikipedia_scraped_data.csv")
        self.docs["lemmatized_summary"] = self.docs["lemmatized_summary"].apply(ast.literal_eval)
        self.labels = list(self.docs['topic'].unique())
        self.wc = None
        self.log_prior = None
        self.log_likelihood = None
        self.unseen_log_likelihood = None
        self.word2idx = None
    
    def _retrieve_word_counts(self):
        data = self.docs.explode("lemmatized_summary")
        self.wc = (data.groupby(["lemmatized_summary", "topic"]).size().unstack(fill_value = 0).reindex(columns = self.labels, fill_value = 0))
    
    def _fit(self):
        self._retrieve_word_counts()

    def _save(self):
        with open("./NB_classifier.pkl", "wb") as f:
            pickle.dump(self, f)
        print(f"Naive Bayes Classifier successfuly saved.")

    @classmethod
    def load(cls, filepath):
        try:
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
            print(f"Successfully loaded Naive Bayes Classifier from {filepath}")
            return obj
        except Exception as e:
            print(f"Error loading: {e}")
            return None

    def NB_classify(self, words, k = 3):
        vocab_size = np.array(self.wc.sum()).sum()
        prod = np.ones(len(self.labels))
        
        for idx, topic in enumerate(self.labels):
            prod[idx] = self.wc[f"{topic}"].sum() / vocab_size
        prod = np.log(prod)

        for word in words:
            w = self.wc.loc[word] if word in self.wc.index else np.zeros(shape = len(self.labels))

            p = (w+1) / (self.wc.sum() + len(self.wc))    

            prod += np.log(np.array(p))

        ind = np.argpartition(prod, -1 * k)[-1 * k:]
        topk = np.flip(ind[np.argsort(prod[ind])])

        return [self.labels[i] for i in topk], dict(zip(self.labels, prod))
    
