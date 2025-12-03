from preprocessor import Preprocessor
from topic_classifier import TopicClassifier
import numpy as np
from collections import OrderedDict
from pathlib import Path

class QueryProcessor:
    def __init__(self, index):
        self.preprocessor = Preprocessor()

        # Check if classifier already fit
        fp = Path("./NB_classifier.pkl")

        if fp.exists():
            self.TC = TopicClassifier.load(fp)
        else:
            self.TC = TopicClassifier()
            self.TC._fit()
            self.TC._save()

        self.index = index

    def _preprocess_query(self, query):
        return self.preprocessor.tokenizer(query)
    
    def _classify_query(self, query):
        query_tokens = self._preprocess_query(query)
        pred_topics = self.TC.NB_classify(query_tokens)[0]
        print(pred_topics)
        return pred_topics
    
    def _compute_RSVBM25_score(self, query, k = 1.2, b = 0.75, topics = None):
        if topics is None:
            topics = set()
        else:
            topics = set(topics)
        pred_topics = set(self._classify_query(query))

        topics |= pred_topics

        docs_subset = set()
        for topic in topics:
            docs_subset |= set(self.index.inverted_topics[topic])

        query_tokens = self._preprocess_query(query)
        avdl = self.index.avg_doc_length

        scores = OrderedDict({})
        for token in query_tokens:
            postings = self.index.inverted_index.get(token, [])
            if not postings:
                continue
            postings = postings.copy()
            log_idf = np.log(postings.pop("IDF"))
            
            for doc in postings.keys():
                if doc not in docs_subset:
                    continue

                tf = postings[doc]
                dl = self.index.doc_lengths.get(doc, 0)

                scores[doc] = scores.get(doc, 0) + log_idf * (tf*(k+1)) / (k*((1 - b) + (b * (dl/avdl))) + tf)

        sorted_scores = sorted(scores.items(), key = lambda x: x[1], reverse=True)

        return sorted_scores
    
    def retrieve_rel_docs(self, query, top_k = 5, topics = None):
        scores = self._compute_RSVBM25_score(query, topics = topics)
        retrieved_docs = dict(scores[:top_k])
        return retrieved_docs
