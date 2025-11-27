from collections import OrderedDict
import pandas as pd
import tqdm
import pickle
import ast

"""
Adapts Preprocessor, Node, LinkedList, and Indexer class definitions from Project 2.
"""

class Indexer:
    """
    Class to build an inverted index.
    """
    def __init__(self):
        self.inverted_index = OrderedDict({})
        self.doc_lengths = {}
        self.inverted_topics = {}
        self.total_docs = 0
        self.avg_doc_length = 0.0

    def get_index(self):
        """ Function to get the index.
            Already implemented."""
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        """ This function adds each tokenized document to the index. This in turn uses the function add_to_index.

        Also calculates and updates corpus statistics at the end.
        """
        for t in tokenized_document:
            postings = self.inverted_index.setdefault(t, {})
            postings[doc_id] = postings.get(doc_id, 0) + 1

        self.doc_lengths[doc_id] = len(tokenized_document)
        self.total_docs += 1

    def sort_terms(self):
        """Sorting the index by terms.
            Already implemented."""
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def calculate_idf(self):
        """ Calculate tf-idf score for each document in the postings lists of the index.

        Calculated as:
        tf = freq of token in doc / total tokens in doc
        idf = total docs / length of postings list for token

        tfidf = tf * idf
        """
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs # calculate average doc length

        for term in self.inverted_index.keys():
            postings = self.inverted_index[term]
            postings_length = len(postings)
            postings["IDF"] = self.total_docs / postings_length # term in index -> length > 0
    
    def _save(self):
        with open("./inverted_index.pkl", "wb") as f:
            pickle.dump(self, f)
        print(f"Inverted Index successfuly saved.")

    @classmethod
    def load(cls, filepath):
        try:
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
            print(f"Successfully loaded Inverted Index from {filepath}")
            return obj
        except Exception as e:
            print(f"Error loading: {e}")
            return None

    def run_indexer(self):
        data = pd.read_csv("wikipedia_scraped_data.csv")
        tqdm.tqdm.pandas()

        data["lemmatized_summary"] = data["lemmatized_summary"].apply(ast.literal_eval)

        tokens = data["lemmatized_summary"].to_list()
        self.inverted_topics = data.groupby("topic")["page_id"].apply(list).to_dict()

        for page_id, t in tqdm.tqdm(zip(data["page_id"], tokens), total = len(data)):
            self.generate_inverted_index(page_id, t)
                                         
        self.sort_terms()
        self.calculate_idf()

        