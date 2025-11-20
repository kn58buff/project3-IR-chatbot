import wikipedia
import regex as re
import pandas as pd
import threading
import json
import numpy as np
import os
import pysolr
import requests
import concurrent.futures
import tqdm
import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

load_dotenv()

# Constant defs for scraping
API_URL = "https://en.wikipedia.org/w/api.php?"
PARAMS = {
    "action": "query",
    "titles": "",
    "prop": "extracts",
    "format": "json"
    }

API_KEY = os.getenv("API_KEY")

HEADERS = {
  'Authorization': API_KEY,
  'User-Agent': "CSE535Project3/1.0 (contact: kn58@buffalo.edu)"
}

# Constant defs for indexing
CORE_NAME = "IRF25P3"
VM_IP = "34.63.237.17"

class Scraper:
    """Class to handle scraping Wikipedia articles and exporting to CSV."""
    def __init__(self):
        self.base_url = API_URL
        self.params = PARAMS.copy()
        self.headers = HEADERS.copy()


    def _process_text(self, text):
        # Remove special characters and extra spaces
        if text is None or text == "":
            return ""
        else:
            text_clean = re.sub(r"[^a-zA-Z0-9 ]", "", text) # remove special chars
            text_clean = re.sub(r"\s+", " ", text_clean).strip(" ") # remove multiple spaces
            return text_clean

    
    def _create_chunks(self, seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]
    
    def _find_topics(self, queries, tol = 6000):
        results = []
        for q in queries:
            if len(results) >= tol:
                break

            res = wikipedia.search(q, results=1000)
            results.extend(res)
        unique_results = list(set(results))
        return unique_results

    def _retrieve_page(self, page_title):
        params = self.params.copy()
        # implement batching logic
        params ={
            "action": "query",
            "prop": "extracts",
            "format": "json",
            "exintro": True,
            "titles": "|".join(page_title),
            "explaintext": True
        }

        response = requests.get(f"{self.base_url}", params=params, headers=self.headers).json()

        pages = response["query"]["pages"]

        batched_data = []

        for k in pages.keys():
            page_id = k

            page_content = self._process_text(pages[page_id].get("extract", ""))
            title = pages[page_id]["title"]
            url = f"https://en.wikipedia.org/w/index.php?curid={page_id}"

            batched_data.append({
                "page_id": page_id,
                "title": title,
                "url": url,
                "summary": page_content,
            })

        return batched_data
    
    def _scrape_search_results(self, page_titles, max_workers=10):
        
        data = []
        batches = list(self._create_chunks(page_titles, 10))
        print(len(batches))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._retrieve_page, batch): batch for batch in batches}
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    batched_data = future.result()
                    data.extend(batched_data)
                except Exception as e:
                    print(f"Error retrieving batch: {e}")
                    continue

        return data
    
    def run_scraper(self, queries, save = False):
        to_scrape = [self._find_topics(qs, tol=10000) for qs in tqdm.tqdm(queries)]
        topics = ["health", "environment", "technology", "economy", "entertainment",
          "sports", "politics", "education", "travel", "food"]
        
        df = pd.DataFrame()
        for qs, topic in zip(to_scrape, topics):
            print(f"Scraping topic: {topic} with {len(qs)} queries")

            to_add = self._scrape_search_results(qs, max_workers=10)
            temp_df = pd.DataFrame(to_add).drop_duplicates(subset=["page_id"], keep="first").reset_index(drop=True)
            temp_df["topic"] = topic
            df = pd.concat([df, temp_df], ignore_index=True)

        df_cleaned = df.replace("", np.nan).dropna()
        true_df = df_cleaned.loc[df_cleaned["summary"].str.len() >= 200]
        if save:
            true_df.to_csv("wikipedia_data.csv", index=False)
        return true_df

class Indexer:
    """
    Class to handle indexing documents into a Solr collection.
    Reuses code from previous project (P1).
    """

    def __init__(self):
        self.docs = pd.read_csv("./wikipedia_data.csv")
        self.solr_url = f'http://{VM_IP}:8983/solr/'
        self.connection = pysolr.Solr(f'{self.solr_url}{CORE_NAME}', always_commit=True, timeout=500)
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")
        self.keywords_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")
    
    def _initialize_core(self):
      # Check if the core exists
      try:
        response = requests.get(f"{self.solr_url}{CORE_NAME}/admin/ping", timeout=5)
        return response.status_code == 200
      except requests.exceptions.ConnectionError:
        print(f"Could not find core, creating core: {CORE_NAME}")

        # Create the core if it does not exist
        print(os.system(
        'sudo su - solr -c "/opt/solr/bin/solr create -c {CORE_NAME} -n data_driven_schema_configs"'.format(
            core=CORE_NAME)))
        
      except requests.exceptions.Timeout:
        print(f"Error: Request to Solr timed out.")
        return False

    def _add_fields(self):
        data = {
            "add-field": [
                {
                    "name": "revision_id",
                    "type": "string",
                    "indexed": True,
                    "multiValued": False
                },
                {
                    "name": "title",
                    "type": "string",
                    "multiValued": False,
                    "indexed": True,
                    "stored": True
                },
                {
                    "name": "url",
                    "type": "string",
                    "multiValued": False
                },
                {
                    "name": "summary",
                    "type": "text_en",
                    "multiValued": False,
                    "stored": True,
                    "indexed": True
                },
                {
                    "name": "topic",
                    "type": "string",
                    "multiValued": False
                },
                {
                    "name": "keywords",
                    "type": "strings",
                    "multiValued": False,
                    "indexed": True,
                    "stored": True
                },
                {
                    "name": "embeddings",
                    "type": "knn_vector",
                    "indexed": True,
                    "stored": True
                }
            ]
        }

        print(requests.post(f"{self.solr_url}{CORE_NAME}/schema", json=data).json())

    def _embed_text(self, text):
        embedding = self.embeddings.encode([text])[0].tolist()
        return embedding

    def _extract_keywords(self, text, top_k=10):
        kw = self.keywords_model.extract_keywords(text, top_n = top_k, keyphrase_ngram_range=(1,3))

        return [w for w, score in kw]
    
    def _generate_kw_embeddings(self):
        self.docs["keywords"] = self.docs["summyary"].apply(self._extract_keywords)
        self.docs["embeddings"] = self.docs["summary"].apply(self._embed_text)

    def _index_documents(self):
        self.docs["summary"] = self.docs["summary"].str.lower()
        self._generate_kw_embeddings()
        
        records = self.docs.to_dict(orient='records')

        try:
            self.connection.add(records)
            print(f"Successfully indexed {len(records)} documents to Solr.")
        except pysolr.SolrError as e:
            print(f"Error indexing documents: {e}")
