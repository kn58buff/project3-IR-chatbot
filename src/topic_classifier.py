import pandas as pd
import numpy as np
import pickle
import ast

class TopicClassifier:
    """Implementation of a Multinomial Naive Bayes Classifier for topic classification. Uses Laplace smoothing and log probabilities.
    """
    def __init__(self):
        self.docs = pd.read_csv("./wikipedia_scraped_data.csv")
        self.docs["lemmatized_summary"] = self.docs["lemmatized_summary"].apply(ast.literal_eval)
        self.labels = list(self.docs['topic'].unique())
        self.wc = None # corpus word counts
    
    def _retrieve_word_counts(self):
        data = self.docs.explode("lemmatized_summary")
        self.wc = (data.groupby(["lemmatized_summary", "topic"]).size().unstack(fill_value = 0).reindex(columns = self.labels, fill_value = 0)) # word counts per topic
    
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
        """Classify a query into topics using Multinomial Naive Bayes with Laplace smoothing and log probabilities.
        
        Arguments:
        words: list <str>: lemmatized tokens of the query
        k: int: number of top topics to return (default k = 3)
        
        Returns:
        topk: list <str>: top k topics
        prod: dict <str, float>: log probabilities for each topic
        """
        vocab_size = np.array(self.wc.sum()).sum() # total number of words in the corpus
        prod = np.ones(len(self.labels)) # initialize array of 1s to store each topics' probability
        
        # iterate through each topic to compute probability of that class
        for idx, topic in enumerate(self.labels):
            prod[idx] = self.wc[f"{topic}"].sum() / vocab_size
        prod = np.log(prod) # take log of probabilities

        # iterate through each word in the query to update probabilities
        for word in words:
            # Check if word exists in the word count dataframe. If not, initialize it as an array of 0.
            w = self.wc.loc[word] if word in self.wc.index else np.zeros(shape = len(self.labels))

            # Apply Laplace smoothing and calculate probability
            p = (w+1) / (self.wc.sum() + len(self.wc))    

            # Add the log of p to the probability array
            prod += np.log(np.array(p))

        ind = np.argpartition(prod, -1 * k)[-1 * k:] # Get indices of top k topics
        topk = np.flip(ind[np.argsort(prod[ind])]) # Sort and return top k topics

        return [self.labels[i] for i in topk], dict(zip(self.labels, prod))
    
